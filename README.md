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
6. Run the baseline  
bash train_cifar100.sh
