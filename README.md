# NetGuardia-Trainer

A machine learning training pipeline for network intrusion detection models. This project trains Deep Autoencoder and MLP-based classifiers on CIC-IDS datasets and exports them to ONNX format for deployment.

## Overview

NetGuardia-Trainer provides a complete pipeline for training network anomaly detection and classification models:

1. **Data Preprocessing** - Loads and preprocesses CIC-IDS-2017 or CIC-IDS-2018 datasets with feature selection and normalization
2. **Deep Autoencoder** - Trains an autoencoder for anomaly detection combined with Random Forest ensemble
3. **MLP Classifier** - Trains a Multi-Layer Perceptron for multi-class attack classification
4. **ONNX Export** - Exports trained models to ONNX format for cross-platform deployment

## Supported Datasets

- [CIC-IDS-2017](https://github.com/ParrotXray/NetGuardia-Trainer/releases/download/2.0.0/CIC-IDS2017.zip) - Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017
- [CIC-IDS-2018](https://github.com/ParrotXray/NetGuardia-Trainer/releases/download/2.0.0/CIC-IDS2018.zip) - Canadian Institute for Cybersecurity Intrusion Detection Dataset 2018

## Requirements

- Python >= 3.10
- Linux operating system
- NVIDIA GPU with CUDA support (recommended for training)

### Dependencies

- PyTorch and PyTorch Lightning for deep learning
- scikit-learn for preprocessing and Random Forest
- imbalanced-learn for SMOTE oversampling
- ONNX and onnxruntime for model export
- pandas and numpy for data processing

See `requirements.txt` for the complete list of dependencies.

## Installation

### Clone the Repository

```bash
git clone https://github.com/ParrotXray/NetGuardia-Trainer.git
cd NetGuardia-Trainer
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Usage

### Prepare Dataset

Place your CIC-IDS dataset CSV files in the appropriate directory:

```
src/rawdata/2017/    # For CIC-IDS-2017 dataset
src/rawdata/2018/    # For CIC-IDS-2018 dataset
```

### Run Training Pipeline

Navigate to the source directory and run the main script:

```bash
cd src
chmod +x main.py
```

#### Run Complete Pipeline

```bash
./main.py -s 2017 -a    # For CIC-IDS-2017
./main.py -s 2018 -a    # For CIC-IDS-2018
```

#### Run Individual Components

```bash
# Data preprocessing only
./main.py -s 2017 -dp

# Deep Autoencoder training only
./main.py -da

# MLP classifier training only
./main.py -mp

# Export models to ONNX only
./main.py -ep
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-s, --set` | Select dataset year (2017 or 2018) |
| `-a, --all` | Run complete training pipeline |
| `-dp, --datapreprocess` | Run data preprocessing |
| `-da, --deepautoencoder` | Train Deep Autoencoder model |
| `-mp, --mlp` | Train MLP classifier |
| `-ep, --export` | Export models to ONNX format |

## Docker

### Pull Pre-built Image

```bash
docker pull ghcr.io/parrotxray/netguardia-train:master
```

### Run with Docker

```bash
cd NetGuardia-Trainer

docker run --gpus all \
  -v ./src/rawdata:/app/src/rawdata \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/metadata:/app/src/metadata \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -e DATASET="2017" \
  -e ALL=true \
  ghcr.io/parrotxray/netguardia-trainer:master
```
### Other ENV
```bash
# Data preprocessing only
-e DATAPREPROCESS=true

# Deep Autoencoder training only
-e DEEPAUTOENCODER=true

# MLP classifier training only
-e MLP=true

# Export models to ONNX only
-e EXPORT=true
```

### Build Docker Image Locally

```bash
docker build -t netguardia-trainer .
```

## Output Directories

After training, the following directories will contain outputs:

| Directory | Contents |
|-----------|----------|
| `outputs/` | Processed CSV files and training outputs |
| `artifacts/` | Trained model files |
| `metadata/` | Model configurations and label encoders |
| `plots/` | Training visualizations and analysis plots |
| `exports/` | Exporting ONNX models |

This project is provided for educational and research purposes.

## Acknowledgments

- Canadian Institute for Cybersecurity for the CIC-IDS datasets
- PyTorch and PyTorch Lightning teams
