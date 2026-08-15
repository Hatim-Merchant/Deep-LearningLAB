mkdir -p results/false-no_freeze && \python train_eval.py  \
   -d cifar100    \
    --data_root data/cifar100    \
    --optimizer adam   \
        --shuffle_classes false \
         --log_dir ./results/false-no_freeze    2>&1 | tee results/false-no_freeze/false-no_freeze.txt