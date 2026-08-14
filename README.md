<div align="center">

# Attention Retention for Continual Learning with Vision Transformers

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)
[![Timm](https://img.shields.io/badge/timm-0.9.12-1a1a2e)](https://github.com/rwightman/pytorch-image-models)

🚀 Investigating whether identifying and retaining important attention heads can help Vision Transformers preserve knowledge during continual learning. 🚀

</div>

---

This repository contains the implementation and experimental code for studying attention‑head importance in Vision Transformers (ViTs) for continual learning. The project explores a simple yet effective idea: estimate the importance of each self‑attention head using gradient‑based scores, then freeze the most important heads when learning new tasks to mitigate catastrophic forgetting.

**Key features:**
- Gradient‑based importance scoring for attention heads in ViT‑B/16.
- Class‑incremental learning setup on CIFAR‑100, ImageNet‑R, and DomainNet.
- Support for multiple task splits (10, 20 tasks).
- Scripts for dataset preparation, training, and evaluation.

---

## Installation

This project uses Python 3.11 and depends on the libraries listed in `requirements.txt`. To set up the environment, run:

```bash
pip install -r requirements.txt
```

We recommend using a virtual environment (e.g., `venv` or `conda`) to isolate dependencies.

---

## Usage

The main training and evaluation script is `train_eval.py`. For a quick start with the CIFAR‑100 experiment:

```bash
bash train_cifar100.sh
```

Before running any script, update the dataset root path in the script (e.g., `--data_root /path/to/dataset`). You can also change the random seed with `--seed`.

To see all available command‑line arguments:

```bash
python train_eval.py --help
```

---

## Overview

The project is built with the following key libraries:

- **PyTorch** and **Torchvision** – for model definition, data loading, and training.
- **timm** – provides the Vision Transformer implementation (ViT‑B/16).
- **NumPy / SciPy** – for numerical computations and gradient analysis.
- **Einops** – for tensor operations.
- **tqdm** – for progress bars.

The code is organized to support multiple datasets and task splits, making it easy to extend to other continual‑learning scenarios.

---

## Method

Vision Transformers contain multiple self‑attention heads in every Transformer layer. While all heads contribute to the model, their contribution to a particular task is not necessarily equal. In continual learning, updating the entire model for every new task can lead to catastrophic forgetting. Our approach works as follows:

1. **Train** the ViT on the current task.
2. **Compute** a gradient‑based importance score for each of the 144 attention heads (12 layers × 12 heads).
3. **Rank** heads by importance and select the most important ones.
4. **Freeze** the selected heads when learning subsequent tasks.
5. **Proceed** to the next task and repeat the process, updating importance scores as needed.

The pipeline is illustrated below:

```
              ┌─────────────────────┐
              │   Current Task      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Train ViT-B/16      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Compute Gradients   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Head Importance     │
              │ Estimation          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Rank Attention      │
              │ Heads               │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Retain / Freeze     │
              │ Important Heads     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │      Next Task      │
              └─────────────────────┘
```

---

## Project Layout

```
.
├── train_eval.py               # Main training and evaluation script
├── train_cifar100.sh           # CIFAR‑100 experiment (10 tasks)
├── train_imagenet_r_s10.sh     # ImageNet‑R, 10‑task split
├── train_imagenet_r_s20.sh     # ImageNet‑R, 20‑task split
├── train_domainnet.sh          # DomainNet, 10‑task split
├── load_data_on_gpu_cluster.py # Dataset preparation / loading utilities
├── tools/                      # Dataset preprocessing and splitting
├── utils/                      # Supporting training and model utilities
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Dataset Preparation

The project supports three datasets:

- **CIFAR‑100**
- **ImageNet‑R**
- **DomainNet**

Download the datasets from their respective sources and organise them into the following directory structure:

```
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

Dataset‑specific preprocessing scripts are available in the `tools/` directory. Update the dataset root paths in those scripts before running them.

---

## Running Experiments

### CIFAR‑100 (10 tasks)

```bash
bash train_cifar100.sh
```

Script sets the appropriate dataset path, task split, and random seed. You can customise these by editing the script or passing arguments directly to `train_eval.py`.

---

## Reproducibility

To reproduce the results:

1. Install dependencies from `requirements.txt`.
2. Download and prepare the required dataset.
3. Set the `--data_root` argument to the correct path.
4. Run the corresponding training script with a fixed random seed (e.g., `--seed 42`).
5. Record evaluation metrics (accuracy, forgetting, etc.) from the output.

We provide scripts with fixed seeds for convenience; you can also run multiple seeds to assess variability.

---

## Citation

If you use this repository or the associated implementation, please cite the original work:

```bibtex
@inproceedings{lu2026arcl,
  title     = {Attention Retention for Continual Learning with Vision Transformers},
  author    = {Lu, Yue and Zhou, Xiangyu and Zhang, Shizhou and Xing, Yinghui and Liang, Guoqiang and Zhang, Wencong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

---

## Acknowledgements

This project was developed as part of the Deep Learning Laboratory work on continual learning and Vision Transformers. We thank the contributors and the open‑source community for providing the building blocks (PyTorch, timm, etc.) that made this research possible.

---

**Status:** This branch (`gradient-head-importance`) implements the gradient‑based attention‑head importance approach and is intended for experimentation and comparison with other attention‑retention strategies.
