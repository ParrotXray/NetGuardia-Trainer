import os

import pandas as pd
import tensorflow as tf
from typing import List, Optional, Dict, Any
from pathlib import Path
from utils import Logger
from model import DeepAutoencoderConfig
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt


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

        self.autoencoder_model: Optional[keras.Model] = None
        self.random_forest_model: Optional[RandomForestClassifier] = None

        self.ae_normalization_params: Optional[Dict[str, float]] = None
        self.ae_mse_scores: Optional[np.ndarray] = None

        self.rf_probabilities: Optional[np.ndarray] = None

        self.ensemble_strategies: Optional[Dict[str, np.ndarray]] = None
        self.strategy_results: Optional[List[Dict[str, Any]]] = []
        self.best_strategy: Optional[Dict[str, Any]] = None

        self.training_history: Optional[keras.callbacks.History] = None

        self.config: Optional[DeepAutoencoderConfig] = config or DeepAutoencoderConfig()

        self.log: Logger = Logger("DeepAutoencoder")

    def check_tensorflow(self) -> None:
        self.log.info(f"TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        self.log.info(f"GPU: {gpus if gpus else 'No GPU detected'}")

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

        exclude_cols = ["Label", "anomaly_if"]

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
        self.binary_labels = (self.labels != "BENIGN").astype(int)

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

        inputs = layers.Input(shape=(input_dim,))
        x = inputs

        for i, (size, dropout) in enumerate(
            zip(self.config.layer_sizes, self.config.dropout_rates)
        ):
            x = layers.Dense(size, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

        encoded = layers.Dense(
            self.config.encoding_dim,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.config.l2_reg),
            name="bottleneck",
        )(x)

        x = encoded
        for i, (size, dropout) in enumerate(
            zip(reversed(self.config.layer_sizes), reversed(self.config.dropout_rates))
        ):
            x = layers.Dense(size, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

        decoded = layers.Dense(input_dim, activation="linear")(x)

        self.autoencoder_model = models.Model(inputs, decoded, name="deep_autoencoder")

        self.autoencoder_model.compile(
            optimizer=Adam(
                learning_rate=self.config.learning_rate, clipnorm=self.config.clipnorm
            ),
            loss="mse",
            metrics=["mae"],
        )

        self.log.info(f"Total parameters: {self.autoencoder_model.count_params():,}")

    def train_autoencoder(self) -> None:
        self.log.info("Training Deep Autoencoder...")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=1,
            ),
        ]

        self.training_history = self.autoencoder_model.fit(
            self.benign_features_scaled,
            self.benign_features_scaled,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        epochs = len(self.training_history.history["loss"])
        final_train_loss = self.training_history.history["loss"][-1]
        final_val_loss = self.training_history.history["val_loss"][-1]

        print(f"Training completed: {epochs} epochs")
        print(f"Final train loss: {final_train_loss:.6f}")
        print(f"Final validation loss: {final_val_loss:.6f}")

    def calculate_ae_normalization(self) -> None:
        self.log.info("Calculating AE normalization parameters...")

        ae_reconstructions = self.autoencoder_model.predict(
            self.benign_features_scaled, batch_size=2048, verbose=1
        )
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

        predictions = self.autoencoder_model.predict(
            self.test_features_scaled, batch_size=2048, verbose=1
        )
        self.ae_mse_scores = np.mean(
            np.square(self.test_features_scaled - predictions), axis=1
        )

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

                if f1 > best_f1 and prec > 0.5 and fpr < estimate_fpr_limit and tpr > 0.90:
                    # self.log.info(f"Strict selection criteria: {fpr}, {tpr}")
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

        # self.best_strategy = min(
        #     [s for s in self.strategy_results if s["f1"] > 0.95],
        #     key=lambda x: x["fpr"]
        # )

        # self.best_strategy = max(
        #     self.strategy_results,
        #     key=lambda x: x["f1"] * 0.6 + (1 - x["fpr"]) * 0.4
        # )

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

        attack_labels = self.labels[(self.labels != "BENIGN") & (self.labels.notna())]
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
            output["Label"] != "BENIGN"
        )
        output_filtered = output[attack_anomaly_mask]

        self.log.info(
            f"Filtered: {len(output_filtered):,} attack anomalies "
            f"(from {len(output):,} total samples)"
        )

        output_path = Path("outputs") / "deep_ae_ensemble.csv"
        output_filtered.to_csv(output_path, index=False)
        self.log.info(f"Saved: {output_path}")

        model_ae_path = Path("artifacts") / "deep_autoencoder.keras"
        self.autoencoder_model.save(model_ae_path)
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

        ax = axes[0, 0]
        ax.plot(self.training_history.history["loss"], label="Train", linewidth=2)
        ax.plot(self.training_history.history["val_loss"], label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Deep AE Training History")
        ax.legend()
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