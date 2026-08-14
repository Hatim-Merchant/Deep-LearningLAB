<div align="center">

# Vision Transformer from Scratch – Deep Learning LAB

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-yellow)]()

Implementing and training a Vision Transformer (ViT) from scratch to understand how self‑attention works on images without relying on high‑level transformer libraries.

</div>

---

This repository contains a self‑contained implementation of the Vision Transformer model, following the tutorial [*Build and Train a Vision Transformer from Scratch*](https://towardsai.net/p/l/build-and-train-vision-transformer-from-scratch) and based on the [original code](https://github.com/MikhailKravets/vision_transformer).

**Key features:**
- Full Vision Transformer (ViT) implementation in pure PyTorch.
- Modular code for patch embedding, multi‑head self‑attention, and MLP blocks.
- Support for CIFAR‑10 and CIFAR‑100.
- Configurable hyperparameters (patch size, number of heads, layers, etc.).
- Training loop with logging of loss and accuracy.

---

## Installation

This project uses Python 3.9+ and depends on the libraries listed in `requirements.txt`. To set up the environment, run the appropriate commands below.

We recommend using a virtual environment (e.g., `venv` or `conda`) to isolate dependencies.

### For CPU / MPS (Apple Silicon)

```bash
pip install -r requirements.txt
```

### For CUDA (GPU)

First install PyTorch with CUDA support (follow the command from the official PyTorch website):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining packages:

```bash
pip install -r requirements-cuda.txt
```

## Usage

The main training script is train.py. For a quick start with the CIFAR‑10 dataset:

```bash
python train.py
```
To see all available command‑line arguments (if your script supports them) or to inspect the configurable parameters:

```bash
python train.py --help
```
You can also modify the model architecture and training settings directly inside the configuration dictionary in train.py (e.g., batch_size, learning_rate, epochs, patch_size, num_heads, num_layers).

## Overview

The project is built with the following key libraries:
```
PyTorch and Torchvision – for model definition, data loading, and training.
NumPy – for numerical computations.
tqdm – for progress bars during training and evaluation.
Matplotlib (optional) – for visualising attention maps and loss curves.
The code is organised into separate modules for the model architecture, data pipeline, and training utilities, making it easy to extend or replace components.
```

## Project Layout
```
Deep-LearningLAB/
├── train.py              # Main training script (contains config and loop)
├── model.py              # Vision Transformer architecture (ViT)
├── data_loader.py        # Data loading and preprocessing (CIFAR-10/100)
├── utils.py              # Helper functions (metrics, visualisation, seed setting)
├── requirements.txt      # CPU / MPS dependencies
├── requirements-cuda.txt # GPU dependencies (with CUDA PyTorch)
└── README.md             # This file
```

## Dataset Preparation

The project supports CIFAR‑10 and CIFAR‑100 out of the box. These datasets are automatically downloaded by `torchvision` when you run the script for the first time – no manual download is required.

If you want to use a custom dataset, organise your images into the following directory structure:

```plaintext
DATA_ROOT/
├── train/
│   ├── class_1/
│   │   ├── image_1.jpg
│   │   └── image_2.jpg
│   ├── class_2/
│   │   └── ...
│   └── ...
└── val/
    ├── class_1/
    │   ├── image_1.jpg
    │   └── ...
    ├── class_2/
    │   └── ...
    └── ...
```
## Reproducibility

To reproduce the results:

* Install dependencies from requirements.txt (or requirements-cuda.txt).
* Run the training script with a fixed random seed. You can set the seed inside train.py (e.g., torch.manual_seed(42)) or pass it as an argument if you have added CLI support.
* Record the final test accuracy and loss curves from the output.
* All experiments in the original tutorial use a fixed random seed for weight initialisation and data shuffling, ensuring consistent results across runs.

## Acknowledgements

The original tutorial and code by Mikhail Kravets.
The open‑source community for PyTorch, torchvision, and the many tools that make deep learning research accessible.
