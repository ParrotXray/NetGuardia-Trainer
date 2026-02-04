"""
Data Preprocessing Configuration Module.

This module defines the configuration dataclass for preprocessing network
traffic data from CIC-IDS datasets (2017 and 2018). It includes settings
for Isolation Forest anomaly detection and feature selection.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessConfig:
    """
    Configuration for Data Preprocessing with Isolation Forest.

    This configuration controls the preprocessing pipeline for network
    traffic data, including feature selection for CIC-IDS datasets and
    Isolation Forest parameters for initial anomaly filtering.

    Attributes:
        Isolation Forest Parameters:
            contamination_rate: Expected proportion of anomalies in the dataset.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 uses all processors).
            if_verbose: Verbosity level for Isolation Forest training.

        Data Cleaning Parameters:
            clip_min: Minimum value for clipping extreme values.
            clip_max: Maximum value for clipping extreme values.
            fill_value: Value used to fill missing or invalid entries.

        Feature Selection:
            cic_2018_selected_features: Selected feature names for CIC-IDS-2018 dataset.
                These features capture key network flow characteristics including:
                - Port and protocol information
                - Packet counts and sizes (forward/backward)
                - Flow timing and inter-arrival times
                - TCP flags (PSH, ACK, SYN, FIN, RST)
                - Statistical measures (mean, std)

            cic_2017_selected_features: Selected feature names for CIC-IDS-2017 dataset.
                Equivalent features to 2018 but with different column naming conventions.
    """

    # Isolation Forest Parameters
    contamination_rate: float = 0.05
    random_state: int = 42
    n_jobs: int = -1
    if_verbose: int = 1

    # Data Cleaning Parameters
    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0

    # CIC-IDS-2018 Dataset Feature Selection
    cic_2018_selected_features: List[str] = field(
        default_factory=lambda: [
            "Dst Port",           # Destination port number
            "Protocol",           # Network protocol identifier
            "Flow Duration",      # Total duration of the flow
            "Tot Fwd Pkts",       # Total forward packets
            "Tot Bwd Pkts",       # Total backward packets
            "TotLen Fwd Pkts",    # Total length of forward packets
            "TotLen Bwd Pkts",    # Total length of backward packets
            "Flow Byts/s",        # Flow bytes per second
            "Flow Pkts/s",        # Flow packets per second
            "Init Fwd Win Byts",  # Initial forward window bytes
            "Init Bwd Win Byts",  # Initial backward window bytes
            "Fwd Pkt Len Mean",   # Mean forward packet length
            "Bwd Pkt Len Mean",   # Mean backward packet length
            "Flow IAT Mean",      # Mean flow inter-arrival time
            "Fwd IAT Mean",       # Mean forward inter-arrival time
            "Bwd IAT Mean",       # Mean backward inter-arrival time
            "PSH Flag Cnt",       # PSH flag count
            "ACK Flag Cnt",       # ACK flag count
            "SYN Flag Cnt",       # SYN flag count
            "FIN Flag Cnt",       # FIN flag count
            "RST Flag Cnt",       # RST flag count
            "Pkt Len Mean",       # Mean packet length
            "Pkt Len Std",        # Packet length standard deviation
            "Fwd Pkt Len Std",    # Forward packet length standard deviation
            "Bwd Pkt Len Std",    # Backward packet length standard deviation
            "Fwd Seg Size Min",   # Minimum forward segment size
            "Fwd Act Data Pkts",  # Forward packets with data payload
        ]
    )

    # CIC-IDS-2017 Dataset Feature Selection
    cic_2017_selected_features: List[str] = field(
        default_factory=lambda: [
            "Destination Port",           # Destination port number
            "Flow Duration",              # Total duration of the flow
            "Total Fwd Packets",          # Total forward packets
            "Total Backward Packets",     # Total backward packets
            "Total Length of Fwd Packets",  # Total length of forward packets
            "Total Length of Bwd Packets",  # Total length of backward packets
            "Flow Bytes/s",               # Flow bytes per second
            "Flow Packets/s",             # Flow packets per second
            "Init_Win_bytes_forward",     # Initial forward window bytes
            "Init_Win_bytes_backward",    # Initial backward window bytes
            "Fwd Packet Length Mean",     # Mean forward packet length
            "Bwd Packet Length Mean",     # Mean backward packet length
            "Flow IAT Mean",              # Mean flow inter-arrival time
            "Fwd IAT Mean",               # Mean forward inter-arrival time
            "Bwd IAT Mean",               # Mean backward inter-arrival time
            "PSH Flag Count",             # PSH flag count
            "ACK Flag Count",             # ACK flag count
            "SYN Flag Count",             # SYN flag count
            "FIN Flag Count",             # FIN flag count
            "RST Flag Count",             # RST flag count
            "Packet Length Mean",         # Mean packet length
            "Packet Length Std",          # Packet length standard deviation
            "Fwd Packet Length Std",      # Forward packet length standard deviation
            "Bwd Packet Length Std",      # Backward packet length standard deviation
            "min_seg_size_forward",       # Minimum forward segment size
            "act_data_pkt_fwd",           # Forward packets with data payload
        ]
    )

    # output_csv_name: str = "output_anomaly"
    # output_stats_name: str = "preprocessing_stats"
    # output_model_name: str = "isolation_forest_model"