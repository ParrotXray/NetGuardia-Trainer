import os

import ujson
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from utils import Logger
from model import ExportConfig


class Exporter:
    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.deep_ae_model: Optional[tf.keras.Model] = None
        self.rf_model: Optional[Any] = None
        self.mlp_model: Optional[tf.keras.Model] = None
        self.label_encoder: Optional[Any] = None

        self.scaler: Optional[Any] = None
        self.clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.best_strategy: Optional[Dict[str, Any]] = None
        self.ae_normalization: Optional[Dict[str, float]] = None
        self.encoding_dim: Optional[int] = None

        self.deep_ae_onnx_path: Optional[Path] = None
        self.rf_onnx_path: Optional[Path] = None
        self.mlp_onnx_path: Optional[Path] = None

        self.full_config: Optional[Dict[str, Any]] = None
        self.inference_config: Optional[Dict[str, Any]] = None

        self.config: ExportConfig = config or ExportConfig()
        self.log: Logger = Logger("Exporter")

    def load_models(self) -> None:
        self.log.info("Loading models and configurations...")

        if not (
            os.path.exists("./metadata")
            or os.path.exists("./artifacts")
            or os.path.exists("./outputs")
        ):
            os.makedirs("./metadata", exist_ok=True)
            os.makedirs("./artifacts", exist_ok=True)
            os.makedirs("./outputs", exist_ok=True)

        model_path = Path("artifacts")

        try:
            ae_path = model_path / "deep_autoencoder.keras"
            self.deep_ae_model = tf.keras.models.load_model(ae_path)
            self.log.info("Deep Autoencoder loaded")
        except Exception as e:
            self.log.error(f"Failed to load Deep Autoencoder: {e}")
            raise

        try:
            rf_path = model_path / "random_forest.pkl"
            self.rf_model = joblib.load(rf_path)
            self.log.info("Random Forest loaded")
        except Exception as e:
            self.log.error(f"Failed to load Random Forest: {e}")
            raise

        try:
            mlp_path = model_path / "mlp.keras"
            encoder_path = model_path / "label_encoder.pkl"
            self.mlp_model = tf.keras.models.load_model(mlp_path)
            self.label_encoder = joblib.load(encoder_path)
            self.log.info("MLP Improved loaded")
        except Exception as e:
            self.log.error(f"Failed to load MLP: {e}")
            raise

        try:
            config_path = model_path / "deep_ae_ensemble_config.pkl"
            ensemble_config = joblib.load(config_path)
            self.scaler = ensemble_config["scaler"]
            self.clip_params = ensemble_config["clip_params"]
            self.best_strategy = ensemble_config["best"]
            self.ae_normalization = ensemble_config.get("ae_normalization", None)
            self.encoding_dim = ensemble_config.get("encoding_dim", 16)
            self.log.info("Ensemble configuration loaded")

            if self.ae_normalization:
                print("AE normalization parameters found:")
                self.log.info(f"Min: {self.ae_normalization['min']:.6f}")
                self.log.info(f"Max: {self.ae_normalization['max']:.6f}")
                self.log.info(f"Mean: {self.ae_normalization['mean']:.6f}")
                self.log.info(f"Std: {self.ae_normalization['std']:.6f}")
            else:
                self.log.warning(
                    "Missing ae_normalization in config, please re-run ensemble training"
                )
        except Exception as e:
            self.log.error(f"Failed to load configuration: {e}")
            raise

    def export_deep_ae_onnx(self) -> None:
        self.log.info("Converting Deep Autoencoder to ONNX...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        input_dim = self.deep_ae_model.input.shape[1]
        spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)

        model_proto, _ = tf2onnx.convert.from_keras(
            self.deep_ae_model, input_signature=spec, opset=self.config.opset_version
        )

        self.deep_ae_onnx_path = Path("exports") / "deep_autoencoder.onnx"
        onnx.save(model_proto, self.deep_ae_onnx_path)
        self.log.info(f"Saved: {self.deep_ae_onnx_path}")

        onnx_model = onnx.load(self.deep_ae_onnx_path)
        onnx.checker.check_model(onnx_model)
        self.log.info("ONNX model validation passed")

    def export_rf_onnx(self) -> None:
        self.log.info("Converting Random Forest to ONNX...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        n_features = len(self.clip_params)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]

        onx = convert_sklearn(
            self.rf_model,
            initial_types=initial_type,
            target_opset=self.config.opset_version,
        )

        self.rf_onnx_path = Path("exports") / "random_forest.onnx"
        with open(self.rf_onnx_path, "wb") as f:
            f.write(onx.SerializeToString())

        self.log.info(f"Saved: {self.rf_onnx_path}")

        onnx_model = onnx.load(self.rf_onnx_path)
        onnx.checker.check_model(onnx_model)
        self.log.info("ONNX model validation passed")

    def export_mlp_onnx(self) -> None:
        self.log.info("Converting MLP Classifier to ONNX...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        mlp_input_dim = self.mlp_model.input_shape[1]
        spec = (tf.TensorSpec((None, mlp_input_dim), tf.float32, name="input"),)

        model_proto, _ = tf2onnx.convert.from_keras(
            self.mlp_model, input_signature=spec, opset=self.config.opset_version
        )

        self.mlp_onnx_path = Path("exports") / "mlp.onnx"
        onnx.save(model_proto, self.mlp_onnx_path)
        self.log.info(f"Saved: {self.mlp_onnx_path}")

        onnx_model = onnx.load(self.mlp_onnx_path)
        onnx.checker.check_model(onnx_model)
        self.log.info("ONNX model validation passed")

    def build_config_json(self) -> None:
        self.log.info("Building configuration JSON...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        scaler_params = {
            "mean": self.scaler.mean_.tolist(),
            "std": self.scaler.scale_.tolist(),
            "feature_names": (
                self.scaler.feature_names_in_.tolist()
                if hasattr(self.scaler, "feature_names_in_")
                else []
            ),
        }

        clip_params_json = {}
        for col, params in self.clip_params.items():
            clip_params_json[col] = {
                "lower": float(params["lower"]),
                "upper": float(params["upper"]),
            }

        ensemble_params = {
            "strategy_name": self.best_strategy["name"],
            "threshold": float(self.best_strategy["threshold"]),
            "tpr": float(self.best_strategy["tpr"]),
            "fpr": float(self.best_strategy["fpr"]),
            "precision": float(self.best_strategy["precision"]),
            "f1": float(self.best_strategy["f1"]),
        }

        if self.ae_normalization:
            ae_normalization_json = {
                "min": float(self.ae_normalization["min"]),
                "max": float(self.ae_normalization["max"]),
                "mean": float(self.ae_normalization["mean"]),
                "std": float(self.ae_normalization["std"]),
                "median": float(self.ae_normalization.get("median", 0.0)),
                "p90": float(self.ae_normalization.get("p90", 0.0)),
                "p95": float(self.ae_normalization.get("p95", 0.0)),
                "p99": float(self.ae_normalization.get("p99", 0.0)),
            }
        else:
            self.log.warning(
                "Using default AE normalization parameters (not recommended)"
            )
            ae_normalization_json = {
                "min": 0.0,
                "max": 1.0,
                "mean": 0.0,
                "std": 1.0,
                "median": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        attack_labels = {
            str(i): label for i, label in enumerate(self.label_encoder.classes_)
        }

        self.full_config = {
            "created_at": pd.Timestamp.now().isoformat(),
            "model": {
                "deep_autoencoder": {
                    "file": "deep_autoencoder.onnx",
                    "input_dim": int(self.deep_ae_model.input.shape[1]),
                    "encoding_dim": int(self.encoding_dim),
                },
                "random_forest": {
                    "file": "random_forest.onnx",
                    "n_estimators": int(self.rf_model.n_estimators),
                    "n_features": int(len(self.clip_params)),
                },
                "mlp_classifier": {
                    "file": "mlp.onnx",
                    "input_dim": int(self.mlp_model.input_shape[1]),
                    "n_classes": int(len(self.label_encoder.classes_)),
                },
            },
            "preprocessing": {
                "clip_params": clip_params_json,
                "scaler": scaler_params,
                "post_scaling_clip": {
                    "min": self.config.post_scaling_clip_min,
                    "max": self.config.post_scaling_clip_max,
                },
            },
            "ensemble": ensemble_params,
            "ae_normalization": ae_normalization_json,
            "attack_labels": attack_labels,
            "feature_order": scaler_params["feature_names"],
        }

        self.inference_config = {
            "threshold": ensemble_params["threshold"],
            "strategy_name": ensemble_params["strategy_name"],
            "clip_params": clip_params_json,
            "scaler_mean": scaler_params["mean"],
            "scaler_std": scaler_params["std"],
            "post_clip_min": self.config.post_scaling_clip_min,
            "post_clip_max": self.config.post_scaling_clip_max,
            "ae_normalization": ae_normalization_json,
            "attack_labels": attack_labels,
            "feature_names": scaler_params["feature_names"],
        }

    def save_config_json(self) -> None:
        self.log.info("Saving configuration JSON...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        full_config_path = Path("exports") / "full_config.json"
        with open(full_config_path, "w", encoding="utf-8") as f:
            ujson.dump(self.full_config, f, indent=2, ensure_ascii=False)
        self.log.info(f"Saved: {full_config_path}")

        inference_config_path = Path("exports") / "inference_config.json"
        with open(inference_config_path, "w", encoding="utf-8") as f:
            ujson.dump(self.inference_config, f, indent=2, ensure_ascii=False)
        self.log.info(f"Saved: {inference_config_path} (inference only)")

    def verify_onnx_models(self) -> None:
        self.log.info("Verifying ONNX models...")

        n_features = len(self.clip_params)
        test_input = np.random.randn(1, n_features).astype(np.float32)

        scaler_params = self.full_config["preprocessing"]["scaler"]
        clip_params_json = self.full_config["preprocessing"]["clip_params"]

        for i, col in enumerate(scaler_params["feature_names"]):
            if col in clip_params_json:
                test_input[0, i] = np.clip(
                    test_input[0, i],
                    clip_params_json[col]["lower"],
                    clip_params_json[col]["upper"],
                )

        test_input_scaled = (test_input - np.array(scaler_params["mean"])) / np.array(
            scaler_params["std"]
        )
        test_input_scaled = np.clip(
            test_input_scaled,
            self.config.post_scaling_clip_min,
            self.config.post_scaling_clip_max,
        ).astype(np.float32)

        print("Testing Deep Autoencoder:")
        session_ae = ort.InferenceSession(str(self.deep_ae_onnx_path))
        ae_output = session_ae.run(None, {"input": test_input_scaled})[0]
        ae_mse = np.mean((test_input_scaled - ae_output) ** 2)
        print(f"AE MSE: {ae_mse:.6f}")

        print()
        print("Testing Random Forest:")
        session_rf = ort.InferenceSession(str(self.rf_onnx_path))
        rf_output = session_rf.run(None, {"float_input": test_input_scaled})
        rf_proba = rf_output[1][0][1]
        print(f"RF Attack Probability: {rf_proba:.6f}")

        print()
        print("Testing MLP Classifier:")
        session_mlp = ort.InferenceSession(str(self.mlp_onnx_path))
        mlp_output = session_mlp.run(None, {"input": test_input_scaled})[0]
        predicted_class = np.argmax(mlp_output[0])
        confidence = mlp_output[0][predicted_class]
        print(
            f"Predicted Class: {predicted_class} ({self.label_encoder.classes_[predicted_class]})"
        )
        print(f"Confidence: {confidence:.6f}")

        print()
        print("Testing Ensemble:")
        ae_normalization_json = self.full_config["ae_normalization"]
        ae_score_norm = (ae_mse - ae_normalization_json["min"]) / (
            ae_normalization_json["max"] - ae_normalization_json["min"] + 1e-10
        )
        ae_score_norm = np.clip(ae_score_norm, 0, 1)
        rf_score_norm = rf_proba

        print(f"AE Score (normalized): {ae_score_norm:.6f}")
        print(f"RF Score: {rf_score_norm:.6f}")

        ensemble_params = self.full_config["ensemble"]
        strategy_name = ensemble_params["strategy_name"]

        if strategy_name.startswith("W_"):
            parts = strategy_name.split("_")[1].split(":")
            w1, w2 = int(parts[0]) / 10, int(parts[1]) / 10

            if "ae" in strategy_name.lower() or w1 > w2:
                ensemble_score = w1 * ae_score_norm + w2 * rf_score_norm
            else:
                ensemble_score = w1 * rf_score_norm + w2 * ae_score_norm
        else:
            ensemble_score = (rf_score_norm + ae_score_norm) / 2.0

        is_anomaly = ensemble_score > ensemble_params["threshold"]

        print(f"Ensemble Score: {ensemble_score:.6f}")
        print(f"Threshold: {ensemble_params['threshold']:.6f}")
        print(f"Is Anomaly: {is_anomaly}")

        if is_anomaly:
            print(
                f"Predicted as Attack: {self.label_encoder.classes_[predicted_class]} (Confidence: {confidence:.2%})"
            )
        else:
            print(f"Predicted as Normal Traffic")
        print()

    def print_summary(self) -> None:
        self.log.info("Export Summary...")

        print("Model Information:")
        input_dim = self.deep_ae_model.input.shape[1]
        print(f"Deep AE: {input_dim} dim -> {self.encoding_dim} dim bottleneck")
        print(f"Random Forest: {self.rf_model.n_estimators} trees")
        print(f"MLP: {len(self.label_encoder.classes_)} attack classes")

        ensemble_params = self.full_config["ensemble"]
        print(
            f"Ensemble: {ensemble_params['strategy_name']} (threshold={ensemble_params['threshold']:.4f})"
        )

        print()
        print("Performance Metrics:")
        print(f"TPR: {ensemble_params['tpr']:.2%}")
        print(f"FPR: {ensemble_params['fpr']:.2%}")
        print(f"Precision: {ensemble_params['precision']:.3f}")
        print(f"F1-Score: {ensemble_params['f1']:.3f}")

        ae_norm = self.full_config["ae_normalization"]
        print()
        print("AE Normalization Parameters:")
        print(f"Min: {ae_norm['min']:.6f}")
        print(f"Max: {ae_norm['max']:.6f}")
        print(f"Mean: {ae_norm['mean']:.6f}")
        print(f"Std: {ae_norm['std']:.6f}")
