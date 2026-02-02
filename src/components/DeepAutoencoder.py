import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from typing import List, Optional, Dict, Any
from pathlib import Path
from utils import Logger
from model import DeepAutoencoderConfig
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt


class AutoencoderModel(nn.Module):
    """PyTorch Deep Autoencoder Model"""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        encoding_dim: int,
        dropout_rates: List[float],
        l2_reg: float = 0.0001,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Encoder
        encoder_layers = []
        prev_size = input_dim
        for i, (size, dropout) in enumerate(zip(layer_sizes, dropout_rates)):
            encoder_layers.append(nn.Linear(prev_size, size))
            encoder_layers.append(nn.BatchNorm1d(size))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_size = size

        # Bottleneck
        encoder_layers.append(nn.Linear(prev_size, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_size = encoding_dim
        for i, (size, dropout) in enumerate(
            zip(reversed(layer_sizes), reversed(dropout_rates))
        ):
            decoder_layers.append(nn.Linear(prev_size, size))
            decoder_layers.append(nn.BatchNorm1d(size))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_size = size

        # Output layer
        decoder_layers.append(nn.Linear(prev_size, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Apply L2 regularization through weight decay in optimizer
        self.l2_reg = l2_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoencoderLightningModule(L.LightningModule):
    """PyTorch Lightning Module for Autoencoder Training"""

    def __init__(
        self,
        model: AutoencoderModel,
        learning_rate: float = 0.001,
        clipnorm: float = 1.0,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 8,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr
        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = torch.mean(torch.abs(x_hat - x))

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = torch.mean(torch.abs(x_hat - x))

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.model.l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class DeepAutoencoder:
    def __init__(self, config: Optional[DeepAutoencoderConfig] = None) -> None:
        self.benign_data: Optional[pd.DataFrame] = None
        self.attack_data: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.features: Optional[pd.DataFrame] = None
        self.binary_labels: Optional[pd.Series] = None

        self.benign_features: Optional[pd.DataFrame] = None
        self.test_features: Optional[pd.DataFrame] = None
        self.test_labels: Optional[pd.Series] = None

        self.scaler: Optional[StandardScaler] = None
        self.clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.benign_features_scaled: Optional[np.ndarray] = None
        self.test_features_scaled: Optional[np.ndarray] = None

        self.autoencoder_model: Optional[AutoencoderModel] = None
        self.lightning_module: Optional[AutoencoderLightningModule] = None
        self.random_forest_model: Optional[RandomForestClassifier] = None

        self.ae_normalization_params: Optional[Dict[str, float]] = None
        self.ae_mse_scores: Optional[np.ndarray] = None

        self.rf_probabilities: Optional[np.ndarray] = None

        self.ensemble_strategies: Optional[Dict[str, np.ndarray]] = None
        self.strategy_results: Optional[List[Dict[str, Any]]] = []
        self.best_strategy: Optional[Dict[str, Any]] = None

        self.training_history: Optional[Dict[str, List[float]]] = None

        self.config: Optional[DeepAutoencoderConfig] = config or DeepAutoencoderConfig()

        self.log: Logger = Logger("DeepAutoencoder")

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_tensorflow(self) -> None:
        """Check PyTorch and GPU availability"""
        self.log.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            self.log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.log.info(f"CUDA: {torch.version.cuda}")
        else:
            self.log.info("GPU: No GPU detected, using CPU")

    def load_data(self) -> None:
        self.log.info("Loading data from outputs/preprocessing_benign.csv...")
        self.benign_data = pd.read_csv("./outputs/preprocessing_benign.csv")
        self.benign_data.columns = self.benign_data.columns.str.strip()

        self.log.info("Loading data from outputs/preprocessing_attack.csv...")
        self.attack_data = pd.read_csv("./outputs/preprocessing_attack.csv")
        self.attack_data.columns = self.attack_data.columns.str.strip()

        print(f"BENIGN samples: {len(self.benign_data):,}")
        print(f"Attack samples: {len(self.attack_data):,}")

    def prepare_data(self) -> None:
        self.log.info("Preparing data...")

        exclude_cols = ["Label"]

        self.benign_features = self.benign_data.drop(
            columns=exclude_cols, errors="ignore"
        ).select_dtypes(include=[np.number])

        attack_features = self.attack_data.drop(
            columns=exclude_cols, errors="ignore"
        ).select_dtypes(include=[np.number])

        self.features = pd.concat(
            [self.benign_features, attack_features], ignore_index=True
        )
        self.labels = pd.concat(
            [self.benign_data["Label"], self.attack_data["Label"]], ignore_index=True
        )
        self.binary_labels = (~self.labels.isin(["BENIGN", "Benign"])).astype(int)

        self.test_features = self.features.copy()
        self.test_labels = self.binary_labels.copy()

        print(f"BENIGN training samples: {len(self.benign_features):,}")
        print(f"Total test samples: {len(self.test_features):,}")
        print(f"Number of features: {self.features.shape[1]}")

    def preprocess_data(self) -> None:
        self.log.info("Preprocessing data...")

        self.benign_features = self.benign_features.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(self.config.fill_value)
        self.test_features = self.test_features.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(self.config.fill_value)

        self.clip_params = {}
        for col in self.benign_features.columns:
            lower = self.benign_features[col].quantile(self.config.winsorize_lower)
            upper = self.benign_features[col].quantile(self.config.winsorize_upper)
            self.benign_features[col] = np.clip(self.benign_features[col], lower, upper)
            self.test_features[col] = np.clip(self.test_features[col], lower, upper)
            self.clip_params[col] = {"lower": float(lower), "upper": float(upper)}

        self.scaler = StandardScaler()
        self.benign_features_scaled = self.scaler.fit_transform(self.benign_features)
        self.test_features_scaled = self.scaler.transform(self.test_features)

        self.benign_features_scaled = np.clip(
            self.benign_features_scaled, self.config.clip_min, self.config.clip_max
        )
        self.test_features_scaled = np.clip(
            self.test_features_scaled, self.config.clip_min, self.config.clip_max
        )

        self.log.info("Preprocessing completed")

    def build_autoencoder(self) -> None:
        self.log.info("Building Deep Autoencoder...")

        input_dim = self.benign_features_scaled.shape[1]

        layer_info = " -> ".join(
            [str(input_dim)]
            + [str(s) for s in self.config.layer_sizes]
            + [str(self.config.encoding_dim)]
        )
        self.log.info(f"Architecture: {layer_info}")

        self.autoencoder_model = AutoencoderModel(
            input_dim=input_dim,
            layer_sizes=self.config.layer_sizes,
            encoding_dim=self.config.encoding_dim,
            dropout_rates=self.config.dropout_rates,
            l2_reg=self.config.l2_reg,
        )

        self.lightning_module = AutoencoderLightningModule(
            model=self.autoencoder_model,
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.clipnorm,
            reduce_lr_factor=self.config.reduce_lr_factor,
            reduce_lr_patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
        )

        total_params = sum(p.numel() for p in self.autoencoder_model.parameters())
        self.log.info(f"Total parameters: {total_params:,}")

    def train_autoencoder(self) -> None:
        self.log.info("Training Deep Autoencoder with PyTorch Lightning...")

        # Prepare data loaders
        train_size = int(len(self.benign_features_scaled) * (1 - self.config.validation_split))
        indices = np.random.permutation(len(self.benign_features_scaled))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_data = torch.FloatTensor(self.benign_features_scaled[train_indices])
        val_data = torch.FloatTensor(self.benign_features_scaled[val_indices])

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        # Use multiprocessing on Linux/Mac, single process on Windows
        num_workers = 4 if os.name != "nt" else 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Callbacks
        os.makedirs("./artifacts", exist_ok=True)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                min_delta=1e-6,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="./artifacts",
                filename="autoencoder_temp",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Trainer
        trainer = L.Trainer(
            max_epochs=self.config.epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            gradient_clip_val=self.config.clipnorm,
            log_every_n_steps=50,
            logger=CSVLogger("logs")
        )

        # Train
        trainer.fit(self.lightning_module, train_loader, val_loader)

        # Load best model
        best_model_path = callbacks[1].best_model_path
        if best_model_path:
            self.lightning_module = AutoencoderLightningModule.load_from_checkpoint(
                best_model_path,
                model=self.autoencoder_model,
            )
            self.log.info(f"Loaded best model from {best_model_path}")

        # Extract training history from trainer
        self.training_history = {
            "loss": [],
            "val_loss": [],
        }

        # Get metrics from trainer's logged metrics
        epochs_trained = trainer.current_epoch + 1 if trainer.current_epoch else self.config.epochs

        print(f"Training completed: {epochs_trained} epochs")
        print(f"Best validation loss: {callbacks[1].best_model_score:.6f}")

    def calculate_ae_normalization(self) -> None:
        self.log.info("Calculating AE normalization parameters...")

        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        with torch.no_grad():
            benign_tensor = torch.FloatTensor(self.benign_features_scaled).to(self.device)
            batch_size = 2048
            ae_reconstructions = []

            for i in range(0, len(benign_tensor), batch_size):
                batch = benign_tensor[i : i + batch_size]
                recon = self.lightning_module(batch)
                ae_reconstructions.append(recon.cpu().numpy())

            ae_reconstructions = np.vstack(ae_reconstructions)

        ae_mse_benign = np.mean(
            np.square(self.benign_features_scaled - ae_reconstructions), axis=1
        )

        self.ae_normalization_params = {
            "min": float(ae_mse_benign.min()),
            "max": float(ae_mse_benign.max()),
            "mean": float(ae_mse_benign.mean()),
            "std": float(ae_mse_benign.std()),
            "median": float(np.median(ae_mse_benign)),
            "p90": float(np.percentile(ae_mse_benign, 90)),
            "p95": float(np.percentile(ae_mse_benign, 95)),
            "p99": float(np.percentile(ae_mse_benign, 99)),
        }

        print(f"AE MSE statistics (BENIGN training set):")
        for key, value in self.ae_normalization_params.items():
            print(f"{key.upper()}: {value:.6f}")

    def predict_autoencoder(self) -> None:
        self.log.info("Calculating Deep AE anomaly scores...")

        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        batch_size = 2048
        n_samples = len(self.test_features_scaled)
        self.ae_mse_scores = np.zeros(n_samples, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_data = torch.FloatTensor(self.test_features_scaled[start:end]).to(
                    self.device
                )
                batch_pred = self.lightning_module(batch_data).cpu().numpy()
                self.ae_mse_scores[start:end] = np.mean(
                    np.square(self.test_features_scaled[start:end] - batch_pred), axis=1
                )
                if (start // batch_size) % 500 == 0:
                    print(f"Progress: {end:,}/{n_samples:,} ({end/n_samples*100:.1f}%)")

        ae_mse_benign = self.ae_mse_scores[self.test_labels == 0]
        ae_mse_attack = self.ae_mse_scores[self.test_labels == 1]

        separation = ae_mse_attack.mean() / ae_mse_benign.mean()

        print(f"AE MSE statistics (test set):")
        self.log.info(
            f"BENIGN: Mean={ae_mse_benign.mean():.6f}, Median={np.median(ae_mse_benign):.6f}"
        )
        self.log.info(
            f"Attack: Mean={ae_mse_attack.mean():.6f}, Median={np.median(ae_mse_attack):.6f}"
        )
        self.log.info(f"Separation: {separation:.2f}x")

        web_attack_mask = self.labels.str.contains("Web Attack", na=False)
        web_attack_scores = self.ae_mse_scores[web_attack_mask]

        print(f"Web Attack MSE stats:")
        print(f"  Mean: {web_attack_scores.mean():.6f}")
        print(f"  Median: {np.median(web_attack_scores):.6f}")
        print(f"  P99: {np.percentile(web_attack_scores, 99):.6f}")

        benign_scores = self.ae_mse_scores[self.test_labels == 0]
        print(f"BENIGN MSE P99: {np.percentile(benign_scores, 99):.6f}")

    def train_random_forest(self) -> None:
        self.log.info("Training Random Forest...")

        benign_indices = np.where(self.binary_labels == 0)[0]
        attack_indices = np.where(self.binary_labels == 1)[0]

        n_samples = min(self.config.rf_train_samples, len(attack_indices))
        benign_sample = np.random.choice(benign_indices, n_samples, replace=False)
        attack_sample = np.random.choice(attack_indices, n_samples, replace=False)

        train_indices = np.concatenate([benign_sample, attack_sample])
        np.random.shuffle(train_indices)

        rf_train_features = self.test_features_scaled[train_indices]
        rf_train_labels = self.test_labels.iloc[train_indices]

        self.log.info(
            f"RF training data: {len(rf_train_features):,} "
            f"(BENIGN: {n_samples:,}, Attack: {n_samples:,})"
        )

        self.random_forest_model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            max_features=self.config.rf_max_features,
            n_jobs=self.config.rf_n_jobs,
            random_state=self.config.rf_random_state,
            verbose=1,
        )

        self.random_forest_model.fit(rf_train_features, rf_train_labels)
        self.log.info("RF training completed")

        self.rf_probabilities = self.random_forest_model.predict_proba(
            self.test_features_scaled
        )[:, 1]
        self.log.info("RF prediction completed")

    def create_ensemble_strategies(self) -> None:
        self.log.info("Creating Ensemble strategies...")

        ae_scores_normalized = (
            self.ae_mse_scores - self.ae_normalization_params["min"]
        ) / (
            self.ae_normalization_params["max"]
            - self.ae_normalization_params["min"]
            + 1e-10
        )
        ae_scores_normalized = np.clip(ae_scores_normalized, 0, 1)
        rf_scores_normalized = self.rf_probabilities

        self.log.info(
            f"AE Score normalized range: "
            f"[{ae_scores_normalized.min():.4f}, {ae_scores_normalized.max():.4f}]"
        )
        self.log.info(
            f"RF Score range: "
            f"[{rf_scores_normalized.min():.4f}, {rf_scores_normalized.max():.4f}]"
        )

        self.ensemble_strategies = {}

        for ae_weight in self.config.ensemble_strategies:
            rf_weight = 1 - ae_weight
            name = f"W_{int(ae_weight * 10)}:{int(rf_weight * 10)}"
            self.ensemble_strategies[name] = (
                ae_weight * ae_scores_normalized + rf_weight * rf_scores_normalized
            )

        self.ensemble_strategies["Max"] = np.maximum(
            ae_scores_normalized, rf_scores_normalized
        )
        self.ensemble_strategies["Min"] = np.minimum(
            ae_scores_normalized, rf_scores_normalized
        )
        self.ensemble_strategies["Product"] = (
            ae_scores_normalized * rf_scores_normalized
        )
        self.ensemble_strategies["Average"] = (
            ae_scores_normalized + rf_scores_normalized
        ) / 2

    def evaluate_strategies(self) -> None:
        self.log.info("Evaluating strategies...")

        header = f"{'Strategy':<12} {'Threshold':>10} {'TPR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7}"
        print(header)
        print("-" * 60)

        for name, score in self.ensemble_strategies.items():
            thresholds = np.percentile(
                score[self.test_labels == 0], self.config.percentiles
            )

            best_f1 = 0
            best_threshold = None
            best_metrics = None

            for percentile, threshold in zip(self.config.percentiles, thresholds):
                pred = (score > threshold).astype(int)

                tp = ((self.test_labels == 1) & (pred == 1)).sum()
                fp = ((self.test_labels == 0) & (pred == 1)).sum()
                fn = ((self.test_labels == 1) & (pred == 0)).sum()
                tn = ((self.test_labels == 0) & (pred == 0)).sum()

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0

                estimate_fpr_limit = ((100 - percentile) / 100) * 1.5

                if (
                    f1 > best_f1
                    and prec > 0.5
                    and fpr < estimate_fpr_limit
                    and tpr > 0.90
                ):
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                        "tpr": float(tpr),
                        "fpr": float(fpr),
                        "precision": float(prec),
                        "f1": float(f1),
                    }

            if best_metrics:
                result_line = (
                    f"{name:<12} {best_threshold:>10.4f} {best_metrics['tpr']:>6.1%} "
                    f"{best_metrics['fpr']:>6.1%} {best_metrics['precision']:>6.2f} "
                    f"{best_metrics['f1']:>6.3f}"
                )
                print(result_line)

                self.strategy_results.append(
                    {
                        "name": name,
                        "score": score,
                        "threshold": best_threshold,
                        **best_metrics,
                    }
                )

        self.best_strategy = max(self.strategy_results, key=lambda x: x["f1"])

        print(f"\n{'=' * 60}")
        print(f"Best strategy: {self.best_strategy['name']}")
        print(f"Threshold: {self.best_strategy['threshold']:.4f}")
        print(f"TPR: {self.best_strategy['tpr']:.2%}")
        print(f"FPR: {self.best_strategy['fpr']:.2%}")
        print(f"Precision: {self.best_strategy['precision']:.3f}")
        print(f"F1: {self.best_strategy['f1']:.3f}")
        print(f"{'=' * 60}\n")

    def evaluate_attack_types(self) -> None:
        self.log.info("Attack type detection rates...")

        attack_labels = self.labels[
            ~self.labels.isin(["BENIGN", "Benign"]) & (self.labels.notna())
        ]

        for attack_type in sorted(attack_labels.unique()):
            mask = self.labels == attack_type
            detected = (
                self.best_strategy["score"][mask] > self.best_strategy["threshold"]
            ).sum()
            total = mask.sum()
            rate = detected / total if total > 0 else 0

            status = "GOOD" if rate > 0.5 else "WARN" if rate > 0.2 else "POOR"
            print(
                f"[{status}] {attack_type[:30]:<30} {detected:>6}/{total:<6} ({rate:>6.1%})"
            )

    def save_results(self) -> None:
        self.log.info("Saving results...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./artifacts", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        output = self.features.copy()
        output["deep_ae_mse"] = self.ae_mse_scores
        output["rf_proba"] = self.rf_probabilities
        output["ensemble_score"] = self.best_strategy["score"]
        output["ensemble_anomaly"] = (
            self.best_strategy["score"] > self.best_strategy["threshold"]
        ).astype(int)
        output["Label"] = self.labels.values

        attack_anomaly_mask = (output["ensemble_anomaly"] == 1) & (
            ~output["Label"].isin(["BENIGN", "Benign"])
        )

        output_filtered = output[attack_anomaly_mask]

        self.log.info(
            f"Filtered: {len(output_filtered):,} attack anomalies "
            f"(from {len(output):,} total samples)"
        )

        output_path = Path("outputs") / "deep_ae_ensemble.csv"
        output_filtered.to_csv(output_path, index=False)
        self.log.info(f"Saved: {output_path}")

        # Save PyTorch model
        model_ae_path = Path("artifacts") / "deep_autoencoder.pt"
        torch.save(
            {
                "model_state_dict": self.autoencoder_model.state_dict(),
                "input_dim": self.autoencoder_model.input_dim,
                "encoding_dim": self.autoencoder_model.encoding_dim,
                "layer_sizes": self.config.layer_sizes,
                "dropout_rates": self.config.dropout_rates,
                "l2_reg": self.config.l2_reg,
            },
            model_ae_path,
        )
        self.log.info(f"Saved: {model_ae_path}")

        model_rf_path = Path("artifacts") / "random_forest.pkl"
        joblib.dump(self.random_forest_model, model_rf_path)
        self.log.info(f"Saved: {model_rf_path}")

        config_data = {
            "scaler": self.scaler,
            "clip_params": self.clip_params,
            "best": self.best_strategy,
            "results": self.strategy_results,
            "encoding_dim": self.config.encoding_dim,
            "ae_normalization": self.ae_normalization_params,
        }
        config_path = Path("artifacts") / "deep_ae_ensemble_config.pkl"
        joblib.dump(config_data, config_path)

        self.log.info(f"Saved: {config_path}")

    def generate_visualizations(self) -> None:
        self.log.info("Generating visualizations...")

        os.makedirs("./plots", exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Plot 1: Training History (placeholder since Lightning handles this differently)
        ax = axes[0, 0]
        ax.text(
            0.5,
            0.5,
            "Training completed\nwith PyTorch Lightning",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Deep AE Training History")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        bins = 50
        ax.hist(
            self.best_strategy["score"][self.test_labels == 0],
            bins=bins,
            alpha=0.7,
            label="BENIGN",
            color="green",
            density=True,
        )
        ax.hist(
            self.best_strategy["score"][self.test_labels == 1],
            bins=bins,
            alpha=0.7,
            label="Attack",
            color="red",
            density=True,
        )
        ax.axvline(
            self.best_strategy["threshold"],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Threshold",
        )
        ax.set_xlabel("Ensemble Score")
        ax.set_title("Score Distribution")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[0, 2]
        top_strategies = sorted(
            self.strategy_results, key=lambda x: x["f1"], reverse=True
        )[:8]
        names = [r["name"] for r in top_strategies]
        f1s = [r["f1"] for r in top_strategies]
        colors = [
            "gold" if r["name"] == self.best_strategy["name"] else "steelblue"
            for r in top_strategies
        ]
        ax.barh(names, f1s, color=colors)
        ax.set_xlabel("F1-Score")
        ax.set_title("Ensemble Strategies")
        ax.grid(alpha=0.3, axis="x")

        ax = axes[1, 0]
        cm = np.array(
            [
                [self.best_strategy["tn"], self.best_strategy["fp"]],
                [self.best_strategy["fn"], self.best_strategy["tp"]],
            ]
        )
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(2):
            for j in range(2):
                text = f"{cm[i, j]:,}\n({cm[i, j] / cm.sum():.1%})"
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=10,
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticklabels(["Normal", "Attack"])
        ax.set_title(f'Confusion Matrix ({self.best_strategy["name"]})')

        ax = axes[1, 1]
        ae_score_norm = (self.ae_mse_scores - self.ae_normalization_params["min"]) / (
            self.ae_normalization_params["max"]
            - self.ae_normalization_params["min"]
            + 1e-10
        )
        ae_score_norm = np.clip(ae_score_norm, 0, 1)

        sample_size = min(10000, len(ae_score_norm))
        sample_idx = np.random.choice(len(ae_score_norm), sample_size, replace=False)
        colors_scatter = [
            "red" if self.test_labels.iloc[i] == 1 else "green" for i in sample_idx
        ]
        ax.scatter(
            ae_score_norm[sample_idx],
            self.rf_probabilities[sample_idx],
            c=colors_scatter,
            alpha=0.3,
            s=1,
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("Deep AE Score (normalized)")
        ax.set_ylabel("RF Score (probability)")
        ax.set_title("AE vs RF Scores")
        ax.grid(alpha=0.3)

        ax = axes[1, 2]
        feature_importance = self.random_forest_model.feature_importances_
        top_10_idx = np.argsort(feature_importance)[-10:]
        ax.barh(range(10), feature_importance[top_10_idx], color="teal")
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"F{i}" for i in top_10_idx], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Feature Importance (RF)")
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plot_path = Path("plots") / "deep_ae_ensemble_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()