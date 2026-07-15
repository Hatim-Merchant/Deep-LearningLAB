# Deep-LearningLAB

Branch: "first-repo": ViT 1 - https://github.com/MikhailKravets/vision_transformer/tree/main  
Branch: "Second-ViT-tintn": ViT 2 - https://github.com/tintn/vision-transformer-from-scratch  
Branch: "big-vit-attentionRetentionCL": Big ViT - https://github.com/zugexiaodui/AttentionRetentionCL.git  
Used CIFAR-10 dataset for branch ViT 1 and 2 and CIFA-100 dataset for Big ViT: https://www.cs.toronto.edu/~kriz/cifar.html

Alex: Alex198464  
Hatim: Hatim-Merchant and st192505@uni-stuttgart.de    
Lisa: Lisa807649 and Lisa  

## Quick Start for branch "Second-ViT-tint"

1. Create conda environment (Python 3.10)  
conda create -n your_env_name python=3.10
2. Activate environment  
conda activate your_env_name
3. Install dependencies  
pip install -r requirements.txt
4. Run the baseline (adjust parameters to your needs)  
bash no_freezing.sh
5. Run the code (adjust parameters to your needs)  
bash freeze.sh

## Quick Start for branch "big-vit-attentionRetentionCL"

1. Create conda environment (Python 3.11)  
conda create -n your_env_name python=3.11
2. Activate environment  
conda activate your_env_name
3. Install dependencies  
pip install -r requirements.txt
4. Load CIFAR-100  
python load_data_on_gpu_cluster.py
6. Run the state of the art  
bash train_cifar100.sh
7. Run our methods, random freezing and no freezing (adjust parameters to your needs)  
bash 1train_freeze_michel.sh  
bash 2train_freeze_michel_current.sh  
bash 3train_freeze_random.sh  
bash 4train_freeze_no_heads.sh  



## Quick Start for gradient-head-importance

1. Create a virtual environment (Python 3.11)

```bash
python3.11 -m venv .venv
```

2. Activate the environment

```bash
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Download CIFAR-100

```bash
python load_data_on_gpu_cluster.py
```

5. Run training

```bash
bash train_cifar100.sh
```

## Outputs

During training, the following are generated:

- `outputs/head_importance_task_*.pt` – 12×12 head importance matrices
- `outputs/head_importance_heatmaps/` – Heatmaps for each task

To generate heatmaps:

```bash
python tools/plot_head_importance.py
```
