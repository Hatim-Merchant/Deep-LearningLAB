<div align="center">

# Vision Transformer from Scratch – Deep Learning LAB

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Experimental-yellow)]()

Implementing and training a Vision Transformer (ViT) for image classification using PyTorch Lightning. The project focuses on understanding the ViT architecture, training process, experiment tracking, and evaluation.

</div>

---

This repository contains a PyTorch implementation of a Vision Transformer (ViT) trained on **CIFAR-10**. The training pipeline uses PyTorch Lightning and includes configurable experiments, model checkpoints, metric logging, and visualisation of training results.

**Key features:**
- Vision Transformer (ViT) implementation in PyTorch.
- CIFAR-10 image classification.
- Patch-based image representation using configurable patch size.
- Multi-head self-attention with configurable number of heads.
- Configurable number of encoder blocks.
- Training with PyTorch Lightning.
- Automatic checkpointing and early stopping.
- Experiment configuration and metric storage.
- Training and validation loss/accuracy visualisation.
- CPU and GPU execution support.

---

## Installation

This project uses Python 3.9+ and depends on the libraries listed in `requirements.txt` and `requirements-cuda.txt`.

We recommend using a virtual environment (e.g., `venv` or `conda`) to isolate dependencies.

### For CPU / MPS (Apple Silicon)

```bash
pip install -r requirements.txt
```

### For CUDA (GPU)

The CUDA requirements file contains the versions used for the GPU environment:

```bash
pip install -r requirements-cuda.txt
```

Make sure that the installed PyTorch build is compatible with the CUDA version available on your system.

## Usage

The main training script is `train.py`.

To start a training run with the default configuration:

```bash
python train.py
```

To see the available command-line arguments:

```bash
python train.py --help
```

The training script currently provides options for:
- `--exp-name`
- `--batch-size`
- `--lr`
- `--epochs`
- `--no-early-stopping`
- `--resume`

For example:

```bash
python train.py --exp-name my_experiment --batch-size 128 --lr 0.001 --epochs 100
```

### Default model configuration

The current implementation uses the following configuration:

| Parameter | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Patch size | 4 × 4 |
| Image size | 32 × 32 |
| Hidden size | 48 |
| Number of patches | 64 |
| Attention heads | 8 |
| Encoder blocks | 6 |
| Dropout | 0.1 |
| Embedding dropout | 0.1 |
| Weight decay | 1e-6 |
| Default batch size | 128 |
| Default learning rate | 1e-3 |
| Default maximum epochs | 100 |

The training script automatically uses a GPU when CUDA is available and otherwise falls back to CPU. GPU training uses BF16 precision.

## Overview

The project is built around the following components:

- **PyTorch** – model implementation and tensor operations.
- **Torchvision** – CIFAR-10 dataset and image processing.
- **PyTorch Lightning** – training loop, validation, checkpointing, and callbacks.
- **NumPy / pandas** – supporting numerical and data-processing functionality.
- **TensorBoard** – experiment logging.
- **Matplotlib** – plotting training metrics.

The training pipeline is organised around a Lightning `DataModule`, a ViT model, experiment utilities, and metric callbacks. The main training script creates an experiment directory, saves the configuration, trains the model, stores checkpoints, and saves the final model state.

## Project Layout

The repository contains the following main components:

```text
Deep-LearningLAB/
├── train.py                    # Main training script and experiment configuration
├── classify.py                 # Example CIFAR-10 classification/inference script
├── plot_metrics.py             # Generates training metric plots
├── src/
│   ├── dataset.py              # CIFAR-10 data module and patch transformation
│   ├── models/
│   │   └── basic.py            # Vision Transformer implementation
│   ├── experiment_utils.py     # Experiment directory and configuration utilities
│   └── metrics.py              # Training metric callback
├── experiments/                # Experiment outputs, checkpoints, configs and metrics
├── requirements.txt            # Standard dependencies
├── requirements-cuda.txt       # CUDA/GPU environment dependencies
└── README.md                   # This file
```

## Dataset Preparation

The project currently uses **CIFAR-10**.

The training script creates a `CIFAR10DataModule`, which handles the dataset used for training and validation. The images are RGB images of size 32 × 32, and the model divides each image into 4 × 4 patches.

With a patch size of 4, a 32 × 32 image produces:

```text
(32 / 4) × (32 / 4) = 8 × 8 = 64 patches
```

## Training and Experiments

Each training run is stored under the `experiments/` directory.

The training script creates an experiment directory and stores information including:

- Experiment configuration (`config.json`)
- Training metrics (`metrics.json`)
- Model checkpoints
- Final model state
- PyTorch Lightning training outputs

The training process uses:
- **ModelCheckpoint** to save the best checkpoints based on validation loss.
- **EarlyStopping** with validation loss as the monitored metric.
- **LearningRateMonitor** to track the learning rate.
- A custom `MetricsCallback` to store training and validation metrics.

Early stopping can be disabled with:

```bash
python train.py --no-early-stopping
```

A previous run can be resumed with:

```bash
python train.py --resume
```

## Visualising Metrics

`plot_metrics.py` can be used to visualise the metrics of an existing experiment.

Run:

```bash
python plot_metrics.py
```

The script asks for the name of an experiment directory inside `experiments/`.

For example:

```text
Which folder in experiments do you want to plot? my_experiment
```

The script reads:

```text
experiments/
└── my_experiment/
    ├── metrics.json
    └── config.json
```

and generates:

```text
training_overview.png
```

The resulting figure contains:
- Validation accuracy over epochs.
- Training loss over epochs.
- Validation loss over epochs.

## Reproducibility

To reproduce an experiment:

1. Create and activate a Python virtual environment.
2. Install the appropriate dependencies.
3. Run `train.py` with the desired hyperparameters.
4. Keep the generated experiment directory containing the configuration, metrics, and checkpoints.

For example:

```bash
python train.py \
    --exp-name vit_cifar10 \
    --batch-size 128 \
    --lr 0.001 \
    --epochs 100
```

The exact results can depend on the hardware, software versions, random state, and training configuration.

## Acknowledgements

This project was developed as part of the Deep Learning LAB and uses the open-source PyTorch, Torchvision, and PyTorch Lightning ecosystems.
