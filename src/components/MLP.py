import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
import seaborn as sns
import joblib
from utils import Logger
from model import MLPConfig
from typing import List, Optional, Dict, Any, Tuple


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

        self.mlp_model: Optional[keras.Model] = None

        self.training_history: Optional[keras.callbacks.History] = None

        self.test_loss: Optional[float] = None
        self.test_accuracy: Optional[float] = None
        self.predictions: Optional[np.ndarray] = None

        self.config: MLPConfig = config or MLPConfig()

        self.log: Logger = Logger("MLP")

    def load_data(self) -> None:
        self.log.info(f"Loading data from outputs/deep_ae_ensemble.csv...")
        self.raw_data = pd.read_csv("./outputs/deep_ae_ensemble.csv")
        self.raw_data.columns = self.raw_data.columns.str.strip()

        self.log.info(
            f"Loading preprocessing config from artifacts/deep_ae_ensemble_config.pkl..."
        )
        ensemble_config = joblib.load("./artifacts/deep_ae_ensemble_config.pkl")
        self.scaler = ensemble_config["scaler"]
        self.clip_params = ensemble_config["clip_params"]

        self.anomaly_data = self.raw_data[self.raw_data["ensemble_anomaly"] == 1].copy()
        self.log.info(f"Anomaly samples: {len(self.anomaly_data):,}")

    def prepare_features(self) -> None:
        self.log.info("Preparing features...")

        exclude_cols = [
            "Label",
            "deep_ae_mse",
            "rf_proba",
            "ensemble_score",
            "ensemble_anomaly",
            "anomaly_if",
        ]
        self.features = self.anomaly_data.drop(columns=exclude_cols, errors="ignore")
        self.labels = self.anomaly_data["Label"]

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

        print("Original class distribution:")
        for idx, label in enumerate(self.label_encoder.classes_):
            count = (self.labels_encoded == idx).sum()
            print(f"{idx:2d}. {label:<35} {count:>7,}")

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

        max_count = max(counts)
        self.smote_strategy = {}
        for cls, count in class_counts.items():
            if count < max_count * self.config.smote_ratio:
                self.smote_strategy[cls] = int(max_count * self.config.smote_ratio)

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

        print("Balanced class distribution:")
        unique, counts = np.unique(self.train_labels_balanced, return_counts=True)
        for cls, count in zip(unique, counts):
            label = self.label_encoder.classes_[cls]
            print(f"{label:<35} {count:>7,}")

    def calculate_class_weights(self) -> None:
        self.log.info("Calculating class weights...")

        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_labels_balanced),
            y=self.train_labels_balanced,
        )
        self.class_weights = dict(enumerate(class_weights_array))

        print("Class weights (first 10):")
        for idx in range(min(10, len(class_weights_array))):
            label = self.label_encoder.classes_[idx]
            weight = class_weights_array[idx]
            print(f"{label:<35} {weight:.4f}")

    def build_model(self) -> None:
        self.log.info("Building Improved MLP model...")

        n_classes = len(self.label_encoder.classes_)
        input_dim = self.train_features_balanced.shape[1]

        inputs = layers.Input(shape=(input_dim,), name="input")

        x = inputs

        for i, (size, dropout) in enumerate(
            zip(self.config.layer_sizes, self.config.dropout_rates)
        ):
            x = layers.Dense(size, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

        outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

        self.mlp_model = models.Model(
            inputs=inputs, outputs=outputs, name="mlp_improved"
        )

        self.mlp_model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.log.info(f"Total parameters: {self.mlp_model.count_params():,}")

    def train_model(self) -> None:
        self.log.info("Training MLP...")

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

        self.training_history = self.mlp_model.fit(
            self.train_features_balanced,
            self.train_labels_balanced,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(self.test_features, self.test_labels),
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1,
        )

        self.log.info("Training completed")

    def evaluate_model(self) -> None:
        self.log.info("Evaluating model...")

        self.test_loss, self.test_accuracy = self.mlp_model.evaluate(
            self.test_features, self.test_labels, verbose=0
        )

        self.log.info(f"Test accuracy: {self.test_accuracy:.4f}")
        self.log.info(f"Test loss: {self.test_loss:.4f}")

        self.predictions = np.argmax(
            self.mlp_model.predict(self.test_features, verbose=0), axis=1
        )

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

        if not (
            os.path.exists("./metadata")
            or os.path.exists("./artifacts")
            or os.path.exists("./outputs")
        ):
            os.makedirs("./metadata", exist_ok=True)
            os.makedirs("./artifacts", exist_ok=True)
            os.makedirs("./outputs", exist_ok=True)

        output_df = pd.DataFrame(
            {
                "Label": self.label_encoder.inverse_transform(self.test_labels),
                "predicted_label": self.label_encoder.inverse_transform(
                    self.predictions
                ),
                "correct": self.test_labels == self.predictions,
            }
        )

        prediction_probs = self.mlp_model.predict(self.test_features, verbose=0)
        for idx, class_name in enumerate(self.label_encoder.classes_):
            output_df[f"prob_{class_name}"] = prediction_probs[:, idx]

        csv_path = Path("outputs") / "mlp.csv"
        output_df.to_csv(csv_path, index=False)
        self.log.info(f"Saved: {csv_path}")

        model_path = Path("artifacts") / "mlp.keras"
        self.mlp_model.save(model_path)
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

        if not os.path.exists("./plots"):
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

        # Training history
        ax = axes[0, 1]
        ax.plot(self.training_history.history["accuracy"], label="Train", linewidth=2)
        ax.plot(self.training_history.history["val_accuracy"], label="Val", linewidth=2)
        ax.set_title("Training History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
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
