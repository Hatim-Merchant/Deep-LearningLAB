<div align="center">

# Continual Learning with Attention Head Freezing (ViT / CIFAR-10)

A class-incremental continual learning framework built on Vision Transformers, using attention-head importance scoring to selectively freeze heads and retain previously learned knowledge.

</div>

This is the official implementation of **Heads Already Know: Zero-Parameter Continual Learning via Attention Head Routing in Vision Transformers**. The core idea is to identify and freeze the attention heads that matter most for previously seen tasks, so that a Vision Transformer can keep learning new classes without forgetting old ones.

Please note: This is research code released alongside our paper. Some paths and scripts assume the datasets described below have been downloaded locally.

## Installation

This project uses standard `pip` + `requirements.txt` for dependency management.

```bash
git clone <this-repository-url>
cd Deep-LearningLAB-big-vit-attentionRetentionCL
pip install -r requirements.txt
```

Use this command to see all available options:

```bash
python train_eval.py --help
```

## Environment

- Python: 3.11.5

```
numpy==1.26.4
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
Pillow==11.3.0
scipy==1.16.1
scikit-image==0.22.0
scikit-learn==1.3.2
huggingface-hub==0.18.0
einops==0.7.0
tqdm==4.66.1
```

## Overview

This project makes use of the following libraries and ideas:

- [timm](https://github.com/huggingface/pytorch-image-models) provides the pretrained Vision Transformer (`vit_base_patch16_224.augreg_in21k`) used as the backbone.
- A custom **head-importance scoring** module (Michel-style and Voita-style scoring) ranks attention heads by their contribution to a task.
- A **head-freezing** mechanism cumulatively locks a fraction of the most important heads after each task, protecting them from being overwritten while new heads remain trainable.
- A **modified Adam optimizer** (`ModAdam`) applies per-parameter learning-rate scaling (e.g. slower updates on `qkv` weights) alongside the freezing mask.
- A lightweight **JSON run logger** (`RunLogger`) records `config.json` and `metrics.json` for every experiment, making results easy to compare and reproduce.

## Who is this for? Why should you use this repo?

This repo is for researchers working on **class-incremental continual learning** with transformer backbones who want to:

- Reproduce the ARCL results on CIFAR-100, ImageNet-R, and DomainNet.
- Compare different attention-head selection strategies (`michel`, `michel_current`, `random`) and scoring methods (`michel`, `voita`).
- Extend the head-freezing mechanism to new datasets, backbones, or scoring functions.

## Dataset preparation

### 1. Download the datasets and uncompress them:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- DomainNet: https://ai.bu.edu/M3SDA/

### 2. Rearrange the directory structure:

We use a unified directory structure for all datasets:

```
DATA_ROOT
    |- train
    |    |- class_folder_1
    |    |    |- image_file_1
    |    |    |- image_file_2
    |    |- class_folder_2
    |         |- image_file_2
    |         |- image_file_3
    |- val
         |- class_folder_1
         |    |- image_file_5
         |    |- image_file_6
         |- class_folder_2
              |- image_file_7
              |- image_file_8
```

We provide the scripts `split_[dataset].py` in the `tools` folder to rearrange the directory structure. Please change the `root_dir` in each script to the path of the uncompressed dataset.

## Usage

### Training and evaluation

10-split ImageNet-R: `train_imagenet_r_s10.sh`

20-split ImageNet-R: `train_imagenet_r_s20.sh`

10-split CIFAR-100: `train_cifar100.sh`

10-split DomainNet: `train_domainnet.sh`

Please specify the `--data_root` argument in the above bash scripts to the location of the datasets. Change the `--seed` argument to use different seeds (e.g., 2026, 2027).

### Head-freezing ablations

The numbered scripts in the project root reproduce the head-freezing ablation studies on CIFAR-100:

| Script | Head selection | Description |
| --- | --- | --- |
| `1train_freeze_michel.sh` | `michel` | Considers the aggregated importance across all seen tasks (global importance freezing) |
| `2train_freeze_michel_current.sh` | `michel_current` | Considers solely the most recently learned task (local task importance freezing) |
| `3train_freeze_random.sh` | `random` | Freeze a random subset of heads (baseline) |
| `4train_freeze_no_heads.sh` | — | No head freezing (baseline) |

Key arguments (see `python train_eval.py --help` for the full list):

```
-d, --dataset            cifar100 | imagenet_r | sdomainet
-t, --num_tasks          number of incremental tasks (default: 10)
--head_freeze            enable attention-head freezing
--freeze_ratio           fraction of all heads frozen per task (cumulative)
--freeze_subset          fraction/count of task samples used for importance estimation
--scoring_method         michel | voita
--head_selection         michel | michel_current | random
--log_dir                directory for config.json / metrics.json logs
```

Results and logs for each run are written to `results/<run_name>/`, containing `config.json`, `metrics.json`, and a console log `.txt` file.

## Project layout

```
requirements.txt                  # Project dependencies
train_eval.py                     # main entry-point (training + evaluation loop)
train_cifar100.sh                 # baseline CIFAR-100 training script
train_imagenet_r_s10.sh           # 10-split ImageNet-R training script
train_imagenet_r_s20.sh           # 20-split ImageNet-R training script
train_domainnet.sh                # DomainNet training script
1train_freeze_michel.sh           # head-freezing ablation: Michel scoring
2train_freeze_michel_current.sh   # head-freezing ablation: current-task Michel scoring
3train_freeze_random.sh           # head-freezing ablation: random head freezing
4train_freeze_no_heads.sh         # head-freezing ablation: no freezing (baseline)
load_data_on_gpu_cluster.py       # helper for staging data on a GPU cluster
utils/
    vit_builder.py                 # Vision Transformer backbone construction
    head_freeze.py                 # head-importance scoring & freezing masks
    mod_adam.py                    # modified Adam optimizer with per-parameter LR scaling
    dataset_builder.py             # dataset loading and task splitting
    continual_manager.py           # continual/incremental task management
    trainer.py                     # training loop utilities
    run_logger.py                  # JSON config/metrics logging
    gvm.py                         # global variables manager
    misc.py                        # miscellaneous helpers
tools/                             # dataset-splitting scripts and class-name lists
results/                           # experiment outputs (config.json, metrics.json, logs)
```
