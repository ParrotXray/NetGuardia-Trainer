from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessConfig:
    contamination_rate: float = 0.05
    random_state: int = 42
    n_jobs: int = -1
    if_verbose: int = 1

    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0

    cic_2018_selected_features: List[str] = field(
        default_factory=lambda: [
            "Dst Port",
            "Protocol",
            "Flow Duration",
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "TotLen Fwd Pkts",
            "TotLen Bwd Pkts",
            "Flow Byts/s",
            "Flow Pkts/s",
            "Init Fwd Win Byts",
            "Init Bwd Win Byts",
            "Fwd Pkt Len Mean",
            "Bwd Pkt Len Mean",
            "Flow IAT Mean",
            "Fwd IAT Mean",
            "Bwd IAT Mean",
            "PSH Flag Cnt",
            "ACK Flag Cnt",
            "SYN Flag Cnt",
            "FIN Flag Cnt",
            "RST Flag Cnt",
            "Pkt Len Mean",
            "Pkt Len Std",
            "Fwd Pkt Len Std",
            "Bwd Pkt Len Std",
            "Fwd Seg Size Min",
            "Fwd Act Data Pkts",
        ]
    )

    cic_2017_selected_features: List[str] = field(
        default_factory=lambda: [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "Fwd Packet Length Mean",
            "Bwd Packet Length Mean",
            "Flow IAT Mean",
            "Fwd IAT Mean",
            "Bwd IAT Mean",
            "PSH Flag Count",
            "ACK Flag Count",
            "SYN Flag Count",
            "FIN Flag Count",
            "RST Flag Count",
            "Packet Length Mean",
            "Packet Length Std",
            "Fwd Packet Length Std",
            "Bwd Packet Length Std",
            "min_seg_size_forward",
            "act_data_pkt_fwd",
        ]
    )

    # output_csv_name: str = "output_anomaly"
    # output_stats_name: str = "preprocessing_stats"
    # output_model_name: str = "isolation_forest_model"
