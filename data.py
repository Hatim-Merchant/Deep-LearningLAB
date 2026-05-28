# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms


class SubsetByClass(torch.utils.data.Dataset):
    """
    Dataset wrapper that filters samples to only include specified classes.
    Optionally remaps class labels to be contiguous (0, 1, 2, ...).
    """
    def __init__(self, dataset, classes_to_keep, remap_labels=True):
        self.dataset = dataset
        self.classes_to_keep = classes_to_keep
        self.remap_labels = remap_labels
        
        # Get all targets
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]
        
        # Filter indices belonging to the classes we want to keep
        self.indices = [i for i, target in enumerate(targets) if target in classes_to_keep]
        
        # Create label mapping for remapping to contiguous indices
        if remap_labels:
            self.label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(classes_to_keep))}
        else:
            self.label_map = {label: label for label in classes_to_keep}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, self.label_map[label]
    
    @property
    def targets(self):
        """Return targets for compatibility with subset operations"""
        if hasattr(self.dataset, 'targets'):
            return [self.label_map[self.dataset.targets[i]] for i in self.indices]
        return [self.dataset[i][1] for i in self.indices]


def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None,
                 classes_to_keep=None, remap_labels=True):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    
    # Filter by class if specified
    if classes_to_keep is not None:
        trainset = SubsetByClass(trainset, classes_to_keep, remap_labels=remap_labels)
    
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    
    # Filter by class if specified
    if classes_to_keep is not None:
        testset = SubsetByClass(testset, classes_to_keep, remap_labels=remap_labels)
    
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    all_classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Return only the classes being used
    if classes_to_keep is not None:
        classes = tuple(all_classes[i] for i in sorted(classes_to_keep))
    else:
        classes = all_classes
    
    return trainloader, testloader, classes