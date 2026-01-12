import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
import ujson
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from Logger import Logger


@dataclass
class PreprocessingConfig:
    contamination_rate: float = 0.05
    random_state: int = 42
    n_jobs: int = -1
    if_verbose: int = 1

    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0

    output_csv_name: str = "output_anomaly.csv"
    output_stats_name: str = "preprocessing_stats.json"
    output_model_name: str = "isolation_forest_model.joblib"

class DataPreprocessing:
    def __init__(self,
                 config: Optional[PreprocessingConfig] = None
        ) -> None:

        self.datasets: List[pd.DataFrame] = []
        self.combined_data: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.clf: Optional[IsolationForest] = None
        self.detection_result: Optional[Dict[str, Any]] = None

        self.config: Optional[PreprocessingConfig] = config

        self.log: Logger = Logger("DataPreprocessing")

    def load_dataset(self, file: str) -> None:
        try:
            self.log.info(f"Loading {file}")
            df = pd.read_csv(file, encoding='utf-8', encoding_errors='replace')
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

        self.labels = self.combined_data['Label'].str.replace('�', '-', regex=False).copy()
        self.log.info(f"Combined data: {self.combined_data.shape}")
        self.log.info(f"Tag distribution:\n{self.labels.value_counts()}")

    def feature_preparation(self) -> None:
        if self.combined_data is None:
            raise ValueError("No combined data available. Call merge_dataset() first!")

        self.log.info(f"Feature preparation...")
        non_feature_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
        df_features = self.combined_data.drop(columns=non_feature_cols, errors='ignore')

        self.feature_matrix = df_features.select_dtypes(include=[np.number])
        self.log.info(f"Original feature dimension: {self.feature_matrix.shape}")

        self.feature_matrix = self.feature_matrix.replace([np.inf, -np.inf], np.nan)
        self.feature_matrix = self.feature_matrix.fillna(self.config.fill_value)
        # self.feature_matrix = np.clip(self.feature_matrix, -1e9, 1e9)
        self.feature_matrix = self.feature_matrix.clip(self.config.clip_min, self.config.clip_max)
        self.log.info(f"Cleaned feature dimensions: {self.feature_matrix.shape}")

    def anomaly_detection(self):
        if self.feature_matrix is None:
            raise ValueError("No feature matrix available. Call feature_preparation() first!")

        self.log.info("IsolationForest anomaly detection...")
        self.clf = IsolationForest(
            contamination=self.config.contamination_rate,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.if_verbose
        )

        self.log.info(f"Training IsolationForest (contamination={self.config.contamination_rate})...")
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
            'anomaly_if': anomaly_if,
            'anomaly_count': anomaly_count,
            'anomaly_ratio': anomaly_ratio,
            'predictions': predictions
        }

    def output_result(self, output_dir: str = "..") -> None:
        if self.detection_result is None:
            raise ValueError("No detection result available. Call anomaly_detection() first!")

        self.log.info("Saving processed data...")

        output: Dict[str, Any] = self.feature_matrix.copy()
        output['anomaly_if'] = self.detection_result['anomaly_if']
        output['Label'] = self.labels.values

        output_path = Path(output_dir) / self.config.output_csv_name
        output.to_csv(output_path, index=False)
        self.log.info(f"Stored: {output_path}")

        stats = {
            'total_samples': len(self.combined_data),
            'total_features': self.feature_matrix.shape[1],
            'anomaly_if_count': self.detection_result['anomaly_count'],
            'anomaly_if_ratio': self.detection_result['anomaly_ratio'],
            'contamination_rate': self.config.contamination_rate,
            'label_distribution': self.labels.value_counts().to_dict()
        }

        stats_path = Path(output_dir) / self.config.output_stats_name
        with open(stats_path, 'w', encoding='utf-8') as f:
            ujson.dump(stats, f, indent=2, ensure_ascii=False)
        self.log.info(f"Statistics stored: {stats_path}")

        model_path = Path(output_dir) / self.config.output_model_name
        joblib.dump(self.clf, model_path)
        self.log.info(f"Model stored: {model_path}")