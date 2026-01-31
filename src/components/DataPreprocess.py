import os

import pandas as pd
import numpy as np
from pathlib import Path
import ujson
from typing import List, Optional, Dict, Any
from utils import Logger
from model import PreprocessConfig
from model import UnsupportedDatasetError


class DataPreprocess:
    def __init__(self, year: str, config: Optional[PreprocessConfig] = None) -> None:

        self.datasets: List[pd.DataFrame] = []
        self.combined_data: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.config: Optional[PreprocessConfig] = config or PreprocessConfig()

        self.year: int = year
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

    def load_datasets(self) -> None:
        if self.year not in ["2017", "2018"]:
            raise UnsupportedDatasetError(
                f"Unsupported dataset year: {self.year}. "
                f"Only 2017 and 2018 are supported."
            )

        csv_dir = "./rawdata/2017" if self.year == "2017" else "./rawdata/2018"

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

        _mapping = {
            "Web Attack - Brute Force": "Web Attack",
            "Web Attack - Sql Injection": "Web Attack",
            "Web Attack - XSS": "Web Attack",
            "Infiltration": "Web Attack",
            "Heartbleed": "DoS Slowhttptest",
            "Brute Force -Web": "Web Attack",
            "Brute Force -XSS": "Web Attack",
            "SQL Injection": "Web Attack",
            "Infilteration": "Web Attack",
        }
        self.labels = self.labels.replace(_mapping)

        self.log.info(f"Combined data: {self.combined_data.shape}")

        print("Tag distribution:")
        print(self.labels.value_counts())

    def feature_preparation(self) -> None:
        if self.combined_data is None:
            raise ValueError("No combined data available. Call merge_dataset() first!")

        self.log.info("Feature preparation...")

        if self.year not in ["2017", "2018"]:
            raise UnsupportedDatasetError(
                f"Unsupported dataset year: {self.year}. "
                f"Only 2017 and 2018 are supported."
            )

        selected_features = (
            self.config.cic_2017_selected_features
            if self.year == "2017"
            else self.config.cic_2018_selected_features
        )

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

    def output_result(self) -> None:
        if self.feature_matrix is None:
            raise ValueError(
                "No feature matrix available. Call feature_preparation() first!"
            )

        self.log.info("Saving processed data...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        output: pd.DataFrame = self.feature_matrix.copy()
        output["Label"] = self.labels.values

        invalid_labels = ["Unknown", "0", "", "nan"]
        benign_mask = (output["Label"] == "BENIGN") | (output["Label"] == "Benign")
        attack_mask = ~benign_mask & (~output["Label"].isin(invalid_labels))
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
            "benign_samples": len(output_benign),
            "attack_samples": len(output_attack),
            "label_distribution": self.labels.value_counts().to_dict(),
        }

        stats_path = Path("metadata") / "preprocessing_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            ujson.dump(stats, f, indent=2, ensure_ascii=False)
        self.log.info(f"Statistics save: {stats_path}")
