import os
import torchvision
from PIL import Image
import numpy as np

# Create the output directory if it does not exist
os.makedirs("data/cifar100", exist_ok=True)

# Download the raw CIFAR-100 dataset
train = torchvision.datasets.CIFAR100(root="data/cifar100_raw", train=True, download=True)
val = torchvision.datasets.CIFAR100(root="data/cifar100_raw", train=False, download=True)

# Load class names
classes = train.classes

# Create ImageFolder structure
for split_name, dataset in [("train", train), ("val", val)]:
    for cls in classes:
        os.makedirs(f"data/cifar100/{split_name}/{cls}", exist_ok=True)

    counters = {cls: 0 for cls in classes}

    for img, label in zip(dataset.data, dataset.targets):
        cls = classes[label]
        path = f"data/cifar100/{split_name}/{cls}/{counters[cls]:05d}.png"
        Image.fromarray(img).save(path)
        counters[cls] += 1

print("Finished.")