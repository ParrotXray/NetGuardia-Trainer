import os

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
import ujson
from typing import List, Optional, Dict, Any
from utils import Logger
from model import PreprocessConfig


class DataPreprocess:
    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:

        self.datasets: List[pd.DataFrame] = []
        self.combined_data: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.clf: Optional[IsolationForest] = None
        self.detection_result: Optional[Dict[str, Any]] = None

        self.config: Optional[PreprocessConfig] = config or PreprocessConfig()

        self.log: Logger = Logger("DataPreprocess")

    def load_dataset(self, file: str) -> None:
        try:
            self.log.info(f"Loading {file}")
            df = pd.read_csv(file, encoding="utf-8", encoding_errors="replace")
            df.columns = df.columns.str.strip()
            self.datasets.append(df)
            self.log.info(f"Shape: {df.shape}, Label: {df['Label'].nunique()} Class")
        except FileNotFoundError:
            self.log.warning(f"File does not exist: {file}")
        except Exception as e:
            self.log.error(f"Error: {e}")

    def load_datasets(self, csv_dir: str) -> None:
        self.log.info(f"Loading datasets from {csv_dir}...")

        csv_files = list(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            self.log.warning(f"No CSV files found in {csv_dir}")
            return

        for file_path in csv_files:
            self.load_dataset(str(file_path))

        if not self.datasets:
            raise ValueError("No dataset was successfully loaded!")

        self.log.info(f"Successfully loaded {len(self.datasets)} datasets")

    def merge_dataset(self) -> None:
        if not self.datasets:
            raise ValueError("No datasets to merge!")

        self.log.info(f"Merge dataset...")
        self.combined_data = pd.concat(self.datasets, ignore_index=True)

        self.labels = (
            self.combined_data["Label"].str.replace("�", "-", regex=False).copy()
        )

        web_attack_mapping = {
            "Web Attack - Brute Force": "Web Attack",
            "Web Attack - Sql Injection": "Web Attack",
            "Web Attack - XSS": "Web Attack",
            "Infiltration": "Web Attack",
            "Heartbleed": "DoS Slowhttptest",
        }

        self.labels = self.labels.replace(web_attack_mapping)

        self.log.info(f"Combined data: {self.combined_data.shape}")

        print("Tag distribution:")
        print(self.labels.value_counts())

    def feature_preparation(self) -> None:
        if self.combined_data is None:
            raise ValueError("No combined data available. Call merge_dataset() first!")

        self.log.info("Feature preparation...")

        selected_features = [
            # 最重要 (Rank 1-5)
            "Destination Port",
            "Protocol",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            # 重要 (Rank 6-10)
            "Total Length of Bwd Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Fwd Packet Length Mean",
            "Bwd Packet Length Mean",
            # 次要重要 (Rank 11-20)
            "Flow IAT Mean",
            "Fwd IAT Mean",
            "Bwd IAT Mean",
            "Fwd PSH Flags",
            "Bwd PSH Flags",
            "FIN Flag Count",
            "SYN Flag Count",
            "RST Flag Count",
            "ACK Flag Count",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "min_seg_size_forward",
            "Packet Length Mean",
            "Packet Length Std",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
        ]

        available_features = [
            f for f in selected_features if f in self.combined_data.columns
        ]
        missing_features = set(selected_features) - set(available_features)

        if missing_features:
            self.log.warning(f"Missing features: {missing_features}")

        self.feature_matrix = self.combined_data[available_features].copy()
        self.log.info(f"Selected {len(available_features)} features")

        self.feature_matrix = self.feature_matrix.replace([np.inf, -np.inf], np.nan)
        self.feature_matrix = self.feature_matrix.fillna(self.config.fill_value)
        self.feature_matrix = self.feature_matrix.clip(
            self.config.clip_min, self.config.clip_max
        )
        self.log.info(f"Feature matrix shape: {self.feature_matrix.shape}")

    def anomaly_detection(self):
        if self.feature_matrix is None:
            raise ValueError(
                "No feature matrix available. Call feature_preparation() first!"
            )

        self.log.info("IsolationForest anomaly detection...")
        self.clf = IsolationForest(
            contamination=self.config.contamination_rate,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.if_verbose,
        )

        self.log.info(
            f"Training IsolationForest (contamination={self.config.contamination_rate})..."
        )
        self.clf.fit(self.feature_matrix)

        predictions = self.clf.predict(self.feature_matrix)
        anomaly_if = np.where(predictions == 1, 0, 1)

        anomaly_count = int(anomaly_if.sum())
        anomaly_ratio = float(anomaly_count / len(self.combined_data) * 100)

        self.log.info(
            f"Number of anomalies: {anomaly_count:,} / {len(self.combined_data):,} "
            f"({anomaly_ratio:.2f}%)"
        )

        self.detection_result = {
            "anomaly_if": anomaly_if,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": anomaly_ratio,
            "predictions": predictions,
        }

    def output_result(self) -> None:
        if self.detection_result is None:
            raise ValueError(
                "No detection result available. Call anomaly_detection() first!"
            )

        self.log.info("Saving processed data...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./artifacts", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        output: pd.DataFrame = self.feature_matrix.copy()
        output["anomaly_if"] = self.detection_result["anomaly_if"]
        output["Label"] = self.labels.values

        invalid_labels = ["Unknown", "0", "", "nan"]
        benign_mask = output["Label"] == "BENIGN"
        attack_mask = (output["Label"] != "BENIGN") & (
            ~output["Label"].isin(invalid_labels)
        )
        output_benign = output[benign_mask]
        output_attack = output[attack_mask]

        dropped_count = len(output) - len(output_benign) - len(output_attack)
        if dropped_count > 0:
            self.log.warning(f"Dropped {dropped_count:,} rows with invalid labels")

        benign_path = Path("outputs") / "preprocessing_benign.csv"
        attack_path = Path("outputs") / "preprocessing_attack.csv"

        output_benign.to_csv(benign_path, index=False)
        output_attack.to_csv(attack_path, index=False)

        self.log.info(f"BENIGN samples: {len(output_benign):,} -> {benign_path}")
        self.log.info(f"Attack samples: {len(output_attack):,} -> {attack_path}")

        stats = {
            "total_samples": len(self.combined_data),
            "total_features": self.feature_matrix.shape[1],
            "anomaly_if_count": self.detection_result["anomaly_count"],
            "anomaly_if_ratio": self.detection_result["anomaly_ratio"],
            "contamination_rate": self.config.contamination_rate,
            "label_distribution": self.labels.value_counts().to_dict(),
        }

        stats_path = Path("metadata") / "preprocessing_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            ujson.dump(stats, f, indent=2, ensure_ascii=False)
        self.log.info(f"Statistics save: {stats_path}")

        model_path = Path("artifacts") / "isolation_forest_model.joblib"
        joblib.dump(self.clf, model_path)
        self.log.info(f"Model save: {model_path}")