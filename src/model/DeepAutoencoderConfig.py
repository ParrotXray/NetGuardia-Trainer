"""
Deep Autoencoder Configuration Module.

This module defines the configuration dataclass for training a Deep Autoencoder
combined with Random Forest ensemble for network traffic anomaly detection.
The autoencoder learns compressed representations of normal traffic patterns,
while the Random Forest provides additional classification capability.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepAutoencoderConfig:
    """
    Configuration for Deep Autoencoder with Random Forest Ensemble.

    This configuration controls the training parameters for a hybrid anomaly
    detection model that combines deep autoencoder reconstruction error with
    Random Forest classification.

    Attributes:
        Data Preprocessing:
            clip_min: Minimum value for clipping normalized features.
            clip_max: Maximum value for clipping normalized features.
            winsorize_lower: Lower percentile for winsorization to handle outliers.
            winsorize_upper: Upper percentile for winsorization to handle outliers.
            fill_value: Value used to fill missing or invalid entries.

        Autoencoder Architecture:
            encoding_dim: Dimension of the bottleneck (latent) layer.
            layer_sizes: List of hidden layer sizes for encoder (decoder mirrors this).
            dropout_rates: Dropout rate for each layer to prevent overfitting.
            l2_reg: L2 regularization factor for weight decay.

        Autoencoder Training:
            learning_rate: Initial learning rate for Adam optimizer.
            clipnorm: Gradient clipping norm to prevent exploding gradients.
            batch_size: Number of samples per training batch.
            epochs: Maximum number of training epochs.
            validation_split: Fraction of data used for validation.
            early_stopping_patience: Epochs to wait before early stopping.
            reduce_lr_patience: Epochs to wait before reducing learning rate.
            reduce_lr_factor: Factor to reduce learning rate by.
            min_lr: Minimum learning rate threshold.

        Random Forest Parameters:
            rf_n_estimators: Number of trees in the forest.
            rf_max_depth: Maximum depth of each tree.
            rf_min_samples_split: Minimum samples required to split a node.
            rf_min_samples_leaf: Minimum samples required at a leaf node.
            rf_max_features: Number of features to consider for best split.
            rf_n_jobs: Number of parallel jobs (-1 uses all processors).
            rf_random_state: Random seed for reproducibility.
            rf_train_samples: Number of samples used for training Random Forest.

        Weighted Voting Ensemble Configuration:
            ae_threshold_high_percentile: Percentile for "very anomalous" AE threshold.
            ae_threshold_medium_percentile: Percentile for "anomalous" AE threshold.
            ae_threshold_low_percentile: Percentile for "suspicious" AE threshold.
            rf_threshold_high: RF attack probability threshold for high confidence.
            rf_threshold_medium: RF attack probability threshold for medium confidence.
            ae_voting_weight: AE weight in fallback weighted voting.
            rf_voting_weight: RF weight in fallback weighted voting.
    """

    # Data Preprocessing Parameters
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    fill_value: float = 0.0

    # Autoencoder Architecture Parameters
    encoding_dim: int = 16
    layer_sizes: List[int] = field(default_factory=lambda: [1024, 512, 256, 128, 64])
    dropout_rates: List[float] = field(
        default_factory=lambda: [0.3, 0.25, 0.2, 0.15, 0.0]
    )
    l2_reg: float = 0.0001

    # Autoencoder Training Parameters
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    batch_size: int = 1024
    epochs: int = 350
    validation_split: float = 0.15
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Random Forest Parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 20
    rf_min_samples_split: int = 10
    rf_min_samples_leaf: int = 5
    rf_max_features: str = "sqrt"
    rf_n_jobs: int = -1
    rf_random_state: int = 42
    rf_train_samples: int = 50000

    # Weighted Voting Ensemble Configuration
    # AE threshold percentiles (computed from benign training MSE)
    ae_threshold_high_percentile: float = 99.5  # Very anomalous
    ae_threshold_medium_percentile: float = 99.0  # Anomalous
    ae_threshold_low_percentile: float = 97.0  # Suspicious

    # RF confidence thresholds (applied to attack probability)
    rf_threshold_high: float = 0.85  # High confidence
    rf_threshold_medium: float = 0.6  # Medium confidence

    # Fallback voting weights (when decision matrix is inconclusive)
    ae_voting_weight: float = 0.4
    rf_voting_weight: float = 0.6

    # output_csv_name: str = "output_deep_ae_ensemble"
    # output_model_ae: str = "deep_autoencoder"
    # output_model_rf: str = "random_forest"
    # output_config: str = "deep_ae_ensemble_config"
    # output_plot: str = "deep_ae_ensemble_analysis"
