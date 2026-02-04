import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import joblib
from utils import Logger
from model import MLPConfig
from typing import List, Optional, Dict, Any, Tuple


class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        layer_sizes: List[int],
        dropout_rates: List[float],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        layers = []
        prev_size = input_dim
        for size, dropout in zip(layer_sizes, dropout_rates):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size

        layers.append(nn.Linear(prev_size, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPLightningModule(L.LightningModule):
    def __init__(
        self,
        model: MLPModel,
        learning_rate: float = 0.001,
        class_weights: Optional[torch.Tensor] = None,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 7,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class MLP:
    def __init__(self, config: Optional[MLPConfig] = None) -> None:
        self.raw_data: Optional[pd.DataFrame] = None
        self.anomaly_data: Optional[pd.DataFrame] = None

        self.scaler: Optional[Any] = None
        self.clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.label_encoder: Optional[LabelEncoder] = None

        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.features_scaled: Optional[np.ndarray] = None
        self.labels_encoded: Optional[np.ndarray] = None

        self.train_features: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None

        self.smote_strategy: Optional[Dict[int, int]] = None

        self.train_features_balanced: Optional[np.ndarray] = None
        self.train_labels_balanced: Optional[np.ndarray] = None

        self.class_weights: Optional[Dict[int, float]] = None
        self.class_weights_tensor: Optional[torch.Tensor] = None

        self.mlp_model: Optional[MLPModel] = None
        self.lightning_module: Optional[MLPLightningModule] = None

        self.training_history: Optional[Dict[str, List[float]]] = None

        self.test_loss: Optional[float] = None
        self.test_accuracy: Optional[float] = None
        self.predictions: Optional[np.ndarray] = None

        self.config: MLPConfig = config or MLPConfig()

        self.log: Logger = Logger("MLP")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self) -> None:
        self.log.info("Loading data from outputs/deep_ae_ensemble.csv...")
        self.raw_data = pd.read_csv("./outputs/deep_ae_ensemble.csv")
        self.raw_data.columns = self.raw_data.columns.str.strip()

        self.log.info(
            "Loading preprocessing config from artifacts/deep_ae_ensemble_config.pkl..."
        )
        ensemble_config = joblib.load("./artifacts/deep_ae_ensemble_config.pkl")
        self.scaler = ensemble_config["scaler"]
        self.clip_params = ensemble_config["clip_params"]

        self.anomaly_data = self.raw_data.copy()
        self.log.info(f"Attack anomaly samples: {len(self.anomaly_data):,}")

    def prepare_features(self) -> None:
        self.log.info("Preparing features...")

        exclude_cols = [
            "Label",
            "deep_ae_mse",
            "rf_proba",
            "ensemble_score",
            "ensemble_anomaly",
        ]
        self.features = self.anomaly_data.drop(columns=exclude_cols, errors="ignore")
        self.labels = self.anomaly_data["Label"]

        min_samples_threshold = 50

        label_counts = self.labels.value_counts()
        rare_labels = label_counts[label_counts < min_samples_threshold].index.tolist()

        if rare_labels:
            self.log.warning(
                f"Removing {len(rare_labels)} classes with < {min_samples_threshold} samples:"
            )
            for label in rare_labels:
                count = label_counts[label]
                self.log.warning(f"  - {label}: {count} samples")

            mask = ~self.labels.isin(rare_labels)
            self.features = self.features[mask].reset_index(drop=True)
            self.labels = self.labels[mask].reset_index(drop=True)

        self.features = self.features.replace([np.inf, -np.inf], np.nan).fillna(
            self.config.fill_value
        )
        for col in self.features.columns:
            if col in self.clip_params:
                self.features[col] = np.clip(
                    self.features[col],
                    self.clip_params[col]["lower"],
                    self.clip_params[col]["upper"],
                )

        self.features_scaled = self.scaler.transform(self.features)
        self.features_scaled = np.clip(
            self.features_scaled, self.config.clip_min, self.config.clip_max
        )

        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

        self.log.info(f"Feature dimensions: {self.features_scaled.shape}")
        self.log.info(f"Number of classes: {len(self.label_encoder.classes_)}")

        print("\nFinal class distribution:")
        print("=" * 60)
        for idx, label in enumerate(self.label_encoder.classes_):
            count = (self.labels_encoded == idx).sum()
            print(f"{idx:2d}. {label:<35} {count:>7,}")
        print("=" * 60)

    def split_data(self) -> None:
        self.log.info("Splitting data...")

        self.train_features, self.test_features, self.train_labels, self.test_labels = (
            train_test_split(
                self.features_scaled,
                self.labels_encoded,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=self.labels_encoded,
            )
        )

        self.log.info(f"Training set: {self.train_features.shape[0]:,}")
        self.log.info(f"Test set: {self.test_features.shape[0]:,}")

    def apply_smote(self) -> None:
        self.log.info("SMOTE data augmentation...")

        unique, counts = np.unique(self.train_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))

        min_samples_for_smote = 100

        removed_classes = []
        for cls, count in class_counts.items():
            if count < min_samples_for_smote:
                removed_classes.append(cls)

        if removed_classes:
            self.log.warning(
                f"Removing {len(removed_classes)} classes with < {min_samples_for_smote} "
                "samples before SMOTE:"
            )
            for cls in removed_classes:
                label = self.label_encoder.classes_[cls]
                count = class_counts[cls]
                self.log.warning(f"  - {label}: {count} samples")

            mask = ~np.isin(self.train_labels, removed_classes)
            self.train_features = self.train_features[mask]
            self.train_labels = self.train_labels[mask]

            test_mask = ~np.isin(self.test_labels, removed_classes)
            self.test_features = self.test_features[test_mask]
            self.test_labels = self.test_labels[test_mask]

            remaining_labels = [
                self.label_encoder.classes_[i]
                for i in range(len(self.label_encoder.classes_))
                if i not in removed_classes
            ]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(remaining_labels)

            train_label_names = [
                self.label_encoder.classes_[i] for i in self.train_labels
            ]
            test_label_names = [
                self.label_encoder.classes_[i] for i in self.test_labels
            ]
            self.train_labels = self.label_encoder.transform(train_label_names)
            self.test_labels = self.label_encoder.transform(test_label_names)

            unique, counts = np.unique(self.train_labels, return_counts=True)
            class_counts = dict(zip(unique, counts))

        max_count = max(counts)
        self.smote_strategy = {}
        for cls, count in class_counts.items():
            if count < max_count * self.config.smote_ratio:
                self.smote_strategy[cls] = int(max_count * self.config.smote_ratio)

        if not self.smote_strategy:
            self.log.warning("No classes need SMOTE augmentation")
            self.train_features_balanced = self.train_features
            self.train_labels_balanced = self.train_labels
            return

        self.log.info("SMOTE strategy:")
        for cls in self.smote_strategy:
            label = self.label_encoder.classes_[cls]
            original = class_counts[cls]
            target = self.smote_strategy[cls]
            self.log.info(
                f"{label:<35} {original:>6,} -> {target:>6,} (+{target - original:,})"
            )

        min_samples = min([class_counts[c] for c in self.smote_strategy])
        k_neighbors = min(self.config.smote_k_neighbors, min_samples - 1)

        smote = SMOTE(
            sampling_strategy=self.smote_strategy,
            k_neighbors=k_neighbors,
            random_state=self.config.random_state,
        )

        self.train_features_balanced, self.train_labels_balanced = smote.fit_resample(
            self.train_features, self.train_labels
        )

        self.log.info(
            f"Training set size: {self.train_features.shape[0]:,} -> "
            f"{self.train_features_balanced.shape[0]:,}"
        )

        print("\nBalanced class distribution:")
        print("=" * 60)
        unique, counts = np.unique(self.train_labels_balanced, return_counts=True)
        for cls, count in zip(unique, counts):
            label = self.label_encoder.classes_[cls]
            print(f"{label:<35} {count:>7,}")
        print("=" * 60)

    def calculate_class_weights(self) -> None:
        self.log.info("Calculating class weights...")

        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_labels_balanced),
            y=self.train_labels_balanced,
        )
        self.class_weights = dict(enumerate(class_weights_array))
        self.class_weights_tensor = torch.FloatTensor(class_weights_array)

        print("Class weights (first 10):")
        for idx in range(min(10, len(class_weights_array))):
            label = self.label_encoder.classes_[idx]
            weight = class_weights_array[idx]
            print(f"{label:<35} {weight:.4f}")

    def build_model(self) -> None:
        self.log.info("Building Improved MLP model with PyTorch...")

        n_classes = len(self.label_encoder.classes_)
        input_dim = self.train_features_balanced.shape[1]

        self.mlp_model = MLPModel(
            input_dim=input_dim,
            n_classes=n_classes,
            layer_sizes=self.config.layer_sizes,
            dropout_rates=self.config.dropout_rates,
        )

        self.lightning_module = MLPLightningModule(
            model=self.mlp_model,
            learning_rate=self.config.learning_rate,
            class_weights=(
                self.class_weights_tensor.to(self.device)
                if self.class_weights_tensor is not None
                else None
            ),
            reduce_lr_factor=self.config.reduce_lr_factor,
            reduce_lr_patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
        )

        total_params = sum(p.numel() for p in self.mlp_model.parameters())
        self.log.info(f"Total parameters: {total_params:,}")

    def train_model(self) -> None:
        self.log.info("Training MLP with PyTorch Lightning...")

        train_data = torch.FloatTensor(self.train_features_balanced)
        train_labels = torch.LongTensor(self.train_labels_balanced)
        test_data = torch.FloatTensor(self.test_features)
        test_labels = torch.LongTensor(self.test_labels)

        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

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
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        os.makedirs("./artifacts", exist_ok=True)
        callbacks = [
            EarlyStopping(
                monitor="val_acc",
                patience=self.config.early_stopping_patience,
                min_delta=1e-4,
                mode="max",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="./artifacts",
                filename="classifier_temp",
                monitor="val_acc",
                save_top_k=1,
                mode="max",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = L.Trainer(
            max_epochs=self.config.epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=50,
            logger=True,
        )

        trainer.fit(self.lightning_module, train_loader, val_loader)

        best_model_path = callbacks[1].best_model_path
        if best_model_path:
            self.lightning_module = MLPLightningModule.load_from_checkpoint(
                best_model_path,
                model=self.mlp_model,
                class_weights=(
                    self.class_weights_tensor.to(self.device)
                    if self.class_weights_tensor is not None
                    else None
                ),
            )
            self.log.info(f"Loaded best model from {best_model_path}")

        self.log.info("Training completed")

    def evaluate_model(self) -> None:
        self.log.info("Evaluating model...")

        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        test_data = torch.FloatTensor(self.test_features).to(self.device)
        test_labels_tensor = torch.LongTensor(self.test_labels).to(self.device)

        with torch.no_grad():
            logits = self.lightning_module(test_data)
            loss = nn.CrossEntropyLoss()(logits, test_labels_tensor)
            self.predictions = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        self.test_loss = loss.item()
        self.test_accuracy = (self.predictions == self.test_labels).mean()

        self.log.info(f"Test accuracy: {self.test_accuracy:.4f}")
        self.log.info(f"Test loss: {self.test_loss:.4f}")

        print("Detailed classification report:")
        report = classification_report(
            self.test_labels,
            self.predictions,
            target_names=self.label_encoder.classes_,
            digits=4,
        )
        print(f"\n{report}")

    def save_results(self) -> None:
        self.log.info("Saving results...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./artifacts", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        with torch.no_grad():
            test_data = torch.FloatTensor(self.test_features).to(self.device)
            logits = self.lightning_module(test_data)
            prediction_probs = torch.softmax(logits, dim=1).cpu().numpy()

        output_df = pd.DataFrame(
            {
                "Label": self.label_encoder.inverse_transform(self.test_labels),
                "predicted_label": self.label_encoder.inverse_transform(
                    self.predictions
                ),
                "correct": self.test_labels == self.predictions,
            }
        )

        for idx, class_name in enumerate(self.label_encoder.classes_):
            output_df[f"prob_{class_name}"] = prediction_probs[:, idx]

        csv_path = Path("outputs") / "mlp.csv"
        output_df.to_csv(csv_path, index=False)
        self.log.info(f"Saved: {csv_path}")

        model_path = Path("artifacts") / "mlp.pt"
        torch.save(
            {
                "model_state_dict": self.mlp_model.state_dict(),
                "input_dim": self.mlp_model.input_dim,
                "n_classes": self.mlp_model.n_classes,
                "layer_sizes": self.config.layer_sizes,
                "dropout_rates": self.config.dropout_rates,
            },
            model_path,
        )
        self.log.info(f"Saved: {model_path}")

        encoder_path = Path("artifacts") / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        self.log.info(f"Saved: {encoder_path}")

        config_data = {
            "encoder": self.label_encoder,
            "scaler": self.scaler,
            "clip_params": self.clip_params,
            "class_weights": self.class_weights,
            "smote_strategy": self.smote_strategy,
            "test_accuracy": float(self.test_accuracy),
            "test_loss": float(self.test_loss),
        }
        config_path = Path("artifacts") / "mlp_config.pkl"
        joblib.dump(config_data, config_path)
        self.log.info(f"Saved: {config_path}")

    def generate_visualizations(self) -> None:
        self.log.info("Generating visualizations...")

        os.makedirs("./plots", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax = axes[0, 0]
        cm = confusion_matrix(self.test_labels, self.predictions)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_normalized,
            annot=False,
            cmap="Blues",
            ax=ax,
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        ax.set_title("Confusion Matrix (Normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax = axes[0, 1]
        ax.text(
            0.5,
            0.5,
            f"Training completed\nwith PyTorch Lightning\n\nTest Accuracy: {self.test_accuracy:.4f}",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Training History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        accuracies = []
        labels_list = []
        for idx, label in enumerate(self.label_encoder.classes_):
            mask = self.test_labels == idx
            if mask.sum() > 0:
                acc = (self.predictions[mask] == idx).sum() / mask.sum()
                accuracies.append(acc)
                labels_list.append(label[:20])

        y_pos = np.arange(len(labels_list))
        colors = [
            "red" if acc < 0.5 else "orange" if acc < 0.8 else "green"
            for acc in accuracies
        ]
        ax.barh(y_pos, accuracies, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_list, fontsize=8)
        ax.set_xlabel("Accuracy")
        ax.set_title("Per-Class Accuracy")
        ax.grid(alpha=0.3, axis="x")

        ax = axes[1, 1]
        train_dist = np.bincount(self.train_labels_balanced)
        test_dist = np.bincount(self.test_labels)
        x = np.arange(len(self.label_encoder.classes_))
        width = 0.35
        ax.bar(x - width / 2, train_dist, width, label="Train (SMOTE)", alpha=0.8)
        ax.bar(x + width / 2, test_dist, width, label="Test", alpha=0.8)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = Path("plots") / "mlp_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()
