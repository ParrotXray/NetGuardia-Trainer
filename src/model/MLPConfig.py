"""
Multi-Layer Perceptron (MLP) Configuration Module.

This module defines the configuration dataclass for training a Multi-Layer
Perceptron neural network for network traffic classification. The MLP is
used for supervised multi-class classification of network attack types.
"""

from typing import List
from dataclasses import dataclass, field


@dataclass
class MLPConfig:
    """
    Configuration for Multi-Layer Perceptron Classifier.

    This configuration controls the training parameters for a supervised
    MLP classifier that categorizes network traffic into different attack
    types or normal traffic.

    Attributes:
        Data Split Parameters:
            test_size: Fraction of data reserved for testing (0.0 to 1.0).
            random_state: Random seed for reproducible train/test splits.
            fill_value: Value used to fill missing or invalid entries.

        SMOTE Oversampling Parameters:
            smote_ratio: Target ratio of minority to majority class samples.
            smote_k_neighbors: Number of nearest neighbors for SMOTE synthesis.

        MLP Architecture:
            layer_sizes: List of hidden layer sizes (neurons per layer).
            dropout_rates: Dropout rate for each layer to prevent overfitting.

        Training Parameters:
            learning_rate: Initial learning rate for Adam optimizer.
            batch_size: Number of samples per training batch.
            epochs: Maximum number of training epochs.
            validation_split: Fraction of training data used for validation.
            early_stopping_patience: Epochs to wait before early stopping.
            reduce_lr_patience: Epochs to wait before reducing learning rate.
            reduce_lr_factor: Factor to reduce learning rate by.
            min_lr: Minimum learning rate threshold.

        Data Preprocessing:
            clip_min: Minimum value for clipping normalized features.
            clip_max: Maximum value for clipping normalized features.
    """

    # Data Split Parameters
    test_size: float = 0.2
    random_state: int = 42
    fill_value: float = 0.0

    # SMOTE Oversampling Parameters
    smote_ratio: float = 0.5
    smote_k_neighbors: int = 5

    # MLP Architecture Parameters
    layer_sizes: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])

    # Training Parameters
    learning_rate: float = 0.001
    batch_size: int = 512
    epochs: int = 100
    validation_split: float = 0.0
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Data Preprocessing Parameters
    clip_min: float = -5.0
    clip_max: float = 5.0

    # output_model_name: str = "mlp_improved"
    # output_encoder_name: str = "label_encoder_improved"
    # output_config_name: str = "mlp_improved_config"
    # output_plot_name: str = "mlp_improved_analysis"
    # output_csv_name: str = "output_mlp_improved"
