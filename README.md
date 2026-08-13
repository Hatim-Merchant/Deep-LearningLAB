# Attention Retention for Continual Learning with Vision Transformers

This repository contains the implementation and experimental code for studying attention-head importance in Vision Transformers (ViTs) for continual learning.

The project investigates whether identifying and retaining important attention heads can help a Vision Transformer preserve previously learned knowledge while learning a sequence of new tasks.

The main experiments use ViT-B/16 in a class-incremental learning setting and estimate attention-head importance using gradient-based scores.

## Overview

Vision Transformers contain multiple self-attention heads in every Transformer layer. While all heads contribute to the model, their contribution to a particular task is not necessarily equal.

In continual learning, updating the entire model for every new task can lead to catastrophic forgetting of previously learned knowledge.

This project therefore explores a simple idea:

Identify the attention heads that are most important for the current task and retain/freeze them while learning subsequent tasks.

For ViT-B/16, the model contains:

* 12 Transformer layers
* 12 attention heads per layer
* 144 attention heads in total

The importance of individual heads is estimated from their gradients and used to determine which heads should be retained.

## Method

The continual-learning pipeline is organized around the following procedure:

1. Train the Vision Transformer on the current task.
2. Compute an importance score for each attention head using gradients.
3. Rank all attention heads according to their importance.
4. Select the most important heads.
5. Retain/freeze the selected heads when learning subsequent tasks.
6. Continue training on the next task.
7. Evaluate performance across the sequence of tasks.

The gradient-based importance score provides a way of estimating how strongly each attention head contributes to the current objective.

## Continual Learning Setup

The experiments divide datasets into multiple sequential tasks.

For example, the CIFAR-100 experiment uses a 10-task split, where the model learns the tasks sequentially rather than training on all classes simultaneously.

### At each task:

Task 1 → Train → Estimate head importance
                    ↓
              Retain important heads
                    ↓
Task 2 → Train → Update importance
                    ↓
Task 3 → Train → ...
                    ↓
                   ...
Task 10 → Evaluate

This setup allows us to investigate whether retaining important attention heads helps preserve knowledge from earlier tasks.

## Repository Structure
```
Deep-LearningLAB/
│
├── train_eval.py
│   └── Main training and evaluation implementation
│
├── train_cifar100.sh
│   └── CIFAR-100 continual-learning experiments
│
├── train_imagenet_r_s10.sh
│   └── 10-task ImageNet-R experiments
│
├── train_imagenet_r_s20.sh
│   └── 20-task ImageNet-R experiments
│
├── train_domainnet.sh
│   └── DomainNet continual-learning experiments
│
├── load_data_on_gpu_cluster.py
│   └── Dataset preparation/loading utilities
│
├── tools/
│   └── Dataset preprocessing and splitting utilities
│
├── utils/
│   └── Supporting training and model utilities
│
└── requirements.txt
    └── Python dependencies
```
The repository currently provides training scripts for 10-split CIFAR-100, 10-split ImageNet-R, 20-split ImageNet-R and 10-split DomainNet experiments. 

## Requirements

The experiments were developed and tested with:

* Python 3.11.5
* PyTorch 2.1.0
* Torchvision 0.16.0
* timm 0.9.12
* NumPy 2.3.2
* SciPy 1.16.1
* scikit-image 0.22.0
* scikit-learn 1.3.2
* einops 0.7.0
* tqdm 4.66.1

Install the dependencies with:

pip install -r requirements.txt

## Dataset Preparation

The project supports:

* CIFAR-100
* ImageNet-R
* DomainNet

Download the datasets from their respective sources and arrange them using the directory structure expected by the training pipeline.

The expected structure is:
```
DATA_ROOT/
├── train/
│   ├── class_1/
│   │   ├── image_1.jpg
│   │   └── image_2.jpg
│   ├── class_2/
│   │   └── ...
│   └── ...
│
└── val/
    ├── class_1/
    │   ├── image_1.jpg
    │   └── ...
    ├── class_2/
    │   └── ...
    └── ...
```
Dataset-specific preprocessing scripts are available in the tools/ directory. The dataset root paths should be updated in those scripts before running them. (⁠GitHub)

## Running Experiments

### CIFAR-100

The 10-task CIFAR-100 experiment can be launched using:

bash train_cifar100.sh

Before running, update the dataset path in the script:

--data_root /path/to/CIFAR100

The training scripts expose the dataset location through the --data_root argument. Different random seeds can be tested by changing the --seed argument.

## Training Pipeline

The main implementation is contained in train_eval.py.

At a high level, the training process is:
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
## Experimental Goal

The main research question is:

Can gradient-based identification of important attention heads improve knowledge retention in Vision Transformers during continual learning?

The experiments compare different strategies for handling attention heads as new tasks are introduced.

The goal is not simply to reduce the number of parameters, but to investigate whether task-relevant attention heads can act as a form of knowledge retention.


## The current branch:

gradient-head-importance

focuses on the gradient-based attention-head importance approach.

This branch is intended primarily for experimentation and comparison with other attention-retention strategies developed during the project.

## Reproducibility

For reproducible experiments:

1. Install the dependencies from requirements.txt.
2. Prepare the required dataset.
3. Set the appropriate --data_root.
4. Use a fixed random seed.
5. Run the corresponding dataset training script.
6. Record the resulting task-wise evaluation metrics.

For example:

bash train_cifar100.sh

## Related Work

The implementation is based on the idea that Transformer attention heads can have different levels of functional importance. Previous work has shown that attention heads can be analyzed and ranked according to their contribution to model behavior, motivating the use of head-level importance for retention and pruning strategies.

This project applies the idea specifically to continual learning with Vision Transformers.

## Citation

If you use this repository or the associated implementation, please cite the original Attention Retention work:

@inproceedings{lu2026arcl,
  title     = {Attention Retention for Continual Learning with Vision Transformers},
  author    = {Lu, Yue and Zhou, Xiangyu and Zhang, Shizhou and Xing, Yinghui and Liang, Guoqiang and Zhang, Wencong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}

## Acknowledgements

This project was developed as part of the Deep Learning Laboratory work on continual learning and Vision Transformers.

The implementation builds upon PyTorch and the timm Vision Transformer implementations.

Status

This branch represents an experimental implementation of gradient-based attention-head importance for continual learning.
