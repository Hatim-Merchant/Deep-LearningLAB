# Continual Learning with Attention Head Freezing (ViT / CIFAR-10)

This project studies **catastrophic forgetting** in a Vision Transformer (ViT) trained sequentially on CIFAR-10 (https://cave.cs.toronto.edu/kriz/cifar.html) task splits, and evaluates attention head freezing as a mitigation strategy. It builds on the "Vision Transformer from Scratch" implementation (`vit.py`) and extends it with a continual learning pipeline (`continual_learning.py`).
## Codebase

The base ViT implementation (`vit.py`, `train.py`, `data.py`, `utils.py`) is built on top of [tintn/vision-transformer-from-scratch](https://github.com/tintn/vision-transformer-from-scratch). The continual learning pipeline and attention head freezing mechanism (`continual_learning.py`) are extensions on top of this original codebase. The head-importance measure is based on https://github.com/pmichel31415/pytorch-pretrained-BERT/tree/paul/examples.

## Who is this branch for?

This branch is intended for users who want to **demonstrate or reproduce continual learning experiments with a Vision Transformer (ViT) in a small, lightweight setting**.

There are two main use cases:

### 1. Demonstrating catastrophic forgetting

If your goal is to **demonstrate and study catastrophic forgetting** without any mitigation method, use our baseline:

```bash
bash no_freezing.sh
```

This provides the continual learning baseline by sequentially training the ViT on the CIFAR-10 task splits without freezing any attention heads.

### 2. Reproducing attention head freezing

If your goal is to **reproduce our attention head freezing experiment**, use:

```bash
bash freeze.sh
```

This extends the baseline with attention head freezing based on **[Michel et al.'s attention head importance scoring](https://github.com/pmichel31415/pytorch-pretrained-BERT/tree/paul)**. The most important attention heads are identified after each task and frozen before training on the next task.

The experiment is designed as a **small-scale, lightweight setting that does not require a GPU**, making it suitable for reproducing the basic effect of attention head freezing without the computational requirements of larger Vision Transformer experiments.

This setup can also be used as a framework for **implementing and evaluating alternative attention-head importance or freezing methods**.

## Overview

The continual learning experiment (`continual_learning.py`) works as follows:

1. Load a pre-trained ViT (trained on Task 1 and Task 2, i.e. classes 0–3).
2. Sequentially train on Task 3, Task 4, and Task 5 (classes 4–5, 6–7, 8–9).
3. After each task (except the last one), optionally freeze the most important attention heads identified for previous tasks.
4. Track performance on all tasks after every new task to measure forgetting.

**Task structure (CIFAR-10):**

| Task | Classes |
|------|-------------------|
| Task 1 | plane (0), car (1) |
| Task 2 | bird (2), cat (3) |
| Task 3 | deer (4), dog (5) |
| Task 4 | frog (6), horse (7) |
| Task 5 | ship (8), truck (9) |

The original pre-trained model files are never modified — the script always operates on a copy.

## Installation

Dependencies:
- Python 3.10
- PyTorch 1.13.1
- torchvision 0.14.1
- matplotlib 3.7.1
- numpy < 2.0.0
- tqdm 4.68.3

### 1. Create conda environment (Python 3.10)

```bash
conda create -n your_env_name python=3.10
```

### 2. Activate environment

```bash
conda activate your_env_name
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

The main entry point for the continual learning experiments is `continual_learning.py`.

To see all available options:

```bash
python continual_learning.py --help
```


### Available CLI arguments (`continual_learning.py`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pretrained-dir` | str | required | Directory containing the pre-trained model |
| `--exp-name` | str | `CL_Tasks_{pretrained_name}` | Experiment name |
| `--batch-size` | int | `256` | Batch size |
| `--epochs-per-task` | int | `20` | Number of epochs to train on each new task |
| `--lr` | float | `1e-3` | Learning rate |
| `--device` | str | `None` | Device to use (auto-detected if not set) |
| `--save-every` | int | `0` | Save a checkpoint every N epochs |
| `--save-dir` | str | `CL` | Base directory for saving continual learning experiments |
| `--freeze-heads` | flag | `False` | Enable attention head freezing after each task |
| `--freeze-ratio` | float | `0.3` | Ratio of attention heads to freeze |
| `--seed` | int | `42` | Random seed |
| `--save-head-importance` | flag | `False` | Save head importance scores to a JSON file |

### Base ViT training (`train.py`)

To pre-train the base ViT model (e.g. before running continual learning):

```bash
python train.py --exp-name vit-with-10-epochs --epochs 10 --batch-size 32
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--exp-name` | str | required | Experiment name |
| `--batch-size` | int | `256` | Batch size |
| `--epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `1e-2` | Learning rate |
| `--device` | str | — | Device to use |
| `--save-model-every` | int | `0` | Save a checkpoint every N epochs |

## Results

We pretrained the base ViT model on task 1 and 2 for 30 epochs with a batch size of 128 and a learning rate of 0.001, achieving 82.65% validation accuracy.   See `experiments/vit_pretrained_T1and2_cifar10_60epochs/` for more details and the plot in `plots_performance_vit/vit_metrics_vit_pretrained_T1and2_cifar10_60epochs.png`

Continual learning results (Baseline (no head freezing) and head freezing with different ratios) are stored per experiment in `/experiments/method_paper_v1/upload/`. We used the pretrained base ViT model, trained it on the remaining tasks (3, 4, and 5), and froze 20%, 30%, or 40% of the attention heads after each respective task (tasks 2, 3 and 4), which corresponds to 3, 4, or 6 heads per task.

We also added a JSON file showing the importance scores of the attention heads for the trained task in `/experiments/method_paper_v1/head_importance_per_task.json`.

## Project layout

```
requirements.txt              # Project dependencies
train.py                      # Base ViT training entry-point
continual_learning.py         # Continual learning + attention head freezing entry-point
no_freezing.sh                # Baseline run script (no freezing)
freeze.sh                     # Run script with attention head freezing enabled
vit.py                        # ViT model implementation (ViTForClassfication)
data.py                       # Dataset loading and per-task class filtering (SubsetByClass)
utils.py                      # Checkpointing and helper utilities
plot_metrics.py               # Plotting for base training metrics
plot_cl_metrics.py            # Plotting for continual learning metrics/forgetting
inspect.ipynb                 # Pre-existing Notebook for model/attention inspection
vision_transformers.ipynb     # ViT walkthrough notebook
experiments/                  # Saved experiment runs (models, configs, metrics)
CL/                           # plots to demonstrate catastrophic forgetting
CL_ViT_60_30Epochs/           # Example continual learning to demonstrate catastrophic forgetting on CIFAR-10 (config, metrics, trained model)
                                (trained on the first 5 classes for 60 epochs, then on the remaining 5 classes for 30 epochs)
ViT_60_Epochs/                # ViT trained on the first five classes of CIFAR-10 for 60 epochs (config, metrics, trained model)
ViT_100_Epochs/               # ViT trained on the first five classes of CIFAR-10 for 100 epochs (config, metrics, trained model)
assets/                       # Pre-existing result images from the base codebase (metrics.png, attention.png)
plots_performance_vit/        # Base training performance plots
```
