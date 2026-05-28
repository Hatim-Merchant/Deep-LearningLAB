"""
Continual Learning Experiment for ViT on CIFAR-10

This script demonstrates catastrophic forgetting by:
1. Loading a pre-trained ViT (trained on first 5 classes: 0-4)
2. Expanding the classification head to 10 classes
3. Training ONLY on the new 5 classes (5-9)
4. Tracking performance on both old and new classes to show forgetting

The original model files are NEVER modified - we always load a copy.
"""

import os
import time
import json
import copy
import argparse
import numpy as np
import torch
from torch import nn, optim
from collections import defaultdict

from utils import save_checkpoint
from data import prepare_data
from vit import ViTForClassfication


# CIFAR-10 class names for reference
CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Original classes (first 5) - these were used for pre-training
OLD_CLASSES = [0, 1, 2, 3, 4]

# New classes (remaining 5) - these will be used for continual learning
NEW_CLASSES = [5, 6, 7, 8, 9]


def expand_model_for_new_classes(model, config, num_total_classes=10):
    """
    Expand the model's classification head to accommodate new classes.
    Preserves the weights for old classes and initializes new class weights randomly.
    
    Args:
        model: Pre-trained ViT model
        config: Model configuration
        num_total_classes: Total number of classes after expansion (default 10)
    
    Returns:
        Expanded model with 10-class output head
    """
    # Create a deep copy to avoid modifying the original
    expanded_model = copy.deepcopy(model)
    
    old_num_classes = config['num_classes']
    hidden_size = config['hidden_size']
    
    # Create new classifier with expanded output dimension
    new_classifier = nn.Linear(hidden_size, num_total_classes)
    
    # Copy weights and biases for old classes (0-4)
    with torch.no_grad():
        new_classifier.weight[:old_num_classes, :] = model.classifier.weight
        new_classifier.bias[:old_num_classes] = model.classifier.bias
    
    # Replace the classifier in the expanded model
    expanded_model.classifier = new_classifier
    
    # Update config for the new model
    new_config = config.copy()
    new_config['num_classes'] = num_total_classes
    new_config['original_num_classes'] = old_num_classes
    new_config['old_classes'] = OLD_CLASSES
    new_config['new_classes'] = NEW_CLASSES
    
    return expanded_model, new_config


def evaluate_with_detailed_metrics(model, testloader, device, label_offset=0):
    """
    Evaluate model with detailed per-class and overall metrics.
    
    Args:
        model: The model to evaluate
        testloader: DataLoader for test data
        device: Device to run evaluation on
        label_offset: Offset to add to remapped labels to get original class indices
                      (e.g., 0 for old classes, 5 for new classes)
    
    Returns:
        Dictionary with overall and per-class metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class tracking (using original class indices)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_loss = defaultdict(float)
    
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            logits, _ = model(images)
            
            # Calculate loss (using remapped labels as is for training consistency)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # Calculate predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Track accuracy per class (mapping remapped labels back to original indices)
            for pred, true_label in zip(predictions, labels):
                true_label_remapped = true_label.item()
                # Map back to original class index
                true_label_original = true_label_remapped + label_offset
                class_total[true_label_original] += 1
                if pred == true_label:
                    class_correct[true_label_original] += 1
                
    # Calculate overall metrics
    total_samples = sum(class_total.values())
    accuracy = sum(class_correct.values()) / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    # Calculate per-class accuracies
    per_class_accuracy = {}
    for cls in sorted(class_total.keys()):
        if class_total[cls] > 0:
            per_class_accuracy[cls] = class_correct[cls] / class_total[cls]
        else:
            per_class_accuracy[cls] = 0.0
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'per_class_accuracy': per_class_accuracy,
        'class_correct': dict(class_correct),
        'class_total': dict(class_total)
    }


class ContinualLearningTrainer:
    """
    Trainer for continual learning experiments that tracks forgetting.
    """
    
    def __init__(self, model, optimizer, loss_fn, exp_name, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.config = config
        
        # Metrics storage
        self.metrics = {
            'train_losses': [],
            'new_class_accuracy': [],  # Accuracy on new classes (training objective)
            'old_class_accuracy': [],  # Accuracy on old classes (forgetting metric)
            'overall_accuracy': [],    # Accuracy on all classes
            'per_class_accuracies': defaultdict(list),  # Per-class accuracy over time
            'epoch_times': []
        }
    
    def train(self, trainloader_new, testloader_old, testloader_new, testloader_all, 
              epochs, save_model_every_n_epochs=0, save_dir="CL"):
        """
        Train the model on new classes while tracking performance on old and new classes.
        
        Args:
            trainloader_new: DataLoader for new classes (training data)
            testloader_old: DataLoader for old classes (to measure forgetting)
            testloader_new: DataLoader for new classes (training objective)
            testloader_all: DataLoader for all classes (overall performance)
            epochs: Number of training epochs
            save_model_every_n_epochs: Save checkpoint every N epochs
            save_dir: Base directory for saving results
        """
        print("=" * 70)
        print("Starting Continual Learning Training")
        print("=" * 70)
        print(f"Training on NEW classes: {[CIFAR10_CLASSES[i] for i in NEW_CLASSES]}")
        print(f"Tracking forgetting on OLD classes: {[CIFAR10_CLASSES[i] for i in OLD_CLASSES]}")
        print(f"Experiment: {self.exp_name}")
        print("=" * 70)
        
        # Initial evaluation before any training (baseline)
        print("\n--- Baseline Evaluation (Before Training on New Classes) ---")
        # Old classes (0-4) have offset 0, new classes (5-9) have offset 5
        old_metrics = evaluate_with_detailed_metrics(self.model, testloader_old, self.device, label_offset=0)
        new_metrics = evaluate_with_detailed_metrics(self.model, testloader_new, self.device, label_offset=5)
        all_metrics = evaluate_with_detailed_metrics(self.model, testloader_all, self.device, label_offset=0)
        
        print(f"Old Classes Accuracy: {old_metrics['accuracy']:.4f}")
        print(f"New Classes Accuracy: {new_metrics['accuracy']:.4f}")
        print(f"Overall Accuracy: {all_metrics['accuracy']:.4f}")
        
        # Store baseline metrics
        self.metrics['baseline'] = {
            'old_class_accuracy': old_metrics['accuracy'],
            'new_class_accuracy': new_metrics['accuracy'],
            'overall_accuracy': all_metrics['accuracy'],
            'per_class_accuracy': {**old_metrics['per_class_accuracy'], **new_metrics['per_class_accuracy']}
        }
        
        # Start training timer
        total_start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train on new classes only
            train_loss = self.train_epoch(trainloader_new)
            
            # Evaluate on all test sets
            old_metrics = evaluate_with_detailed_metrics(self.model, testloader_old, self.device, label_offset=0)
            new_metrics = evaluate_with_detailed_metrics(self.model, testloader_new, self.device, label_offset=5)
            all_metrics = evaluate_with_detailed_metrics(self.model, testloader_all, self.device, label_offset=0)
            
            epoch_time = time.time() - epoch_start
            
            # Store metrics
            self.metrics['train_losses'].append(train_loss)
            self.metrics['old_class_accuracy'].append(old_metrics['accuracy'])
            self.metrics['new_class_accuracy'].append(new_metrics['accuracy'])
            self.metrics['overall_accuracy'].append(all_metrics['accuracy'])
            self.metrics['epoch_times'].append(epoch_time)
            
            # Store per-class accuracies
            for cls, acc in {**old_metrics['per_class_accuracy'], **new_metrics['per_class_accuracy']}.items():
                self.metrics['per_class_accuracies'][cls].append(acc)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Old Classes: {old_metrics['accuracy']:.4f} " + 
                  f"(forgetting: {self.metrics['baseline']['old_class_accuracy'] - old_metrics['accuracy']:.4f})")
            print(f"  New Classes: {new_metrics['accuracy']:.4f}")
            print(f"  Overall: {all_metrics['accuracy']:.4f}")
            
            # Save checkpoint if requested
            if save_model_every_n_epochs > 0 and (epoch+1) % save_model_every_n_epochs == 0:
                checkpoint_dir = os.path.join(save_dir, self.exp_name, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(self.exp_name, self.model, f"epoch_{epoch+1}", base_dir=save_dir)
                print(f"  [Saved checkpoint at epoch {epoch+1}]")
        
        # Calculate total training time
        total_duration = time.time() - total_start_time
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        
        print("\n" + "=" * 70)
        print(f"Training Complete! Total Time: {minutes}m {seconds}s")
        print("=" * 70)
        
        # Final evaluation summary
        final_old = self.metrics['old_class_accuracy'][-1]
        final_new = self.metrics['new_class_accuracy'][-1]
        baseline_old = self.metrics['baseline']['old_class_accuracy']
        
        print("\n--- Final Results ---")
        print(f"Old Class Accuracy: {baseline_old:.4f} → {final_old:.4f} " +
              f"(Δ = {final_old - baseline_old:+.4f})")
        print(f"New Class Accuracy: 0.0000 → {final_new:.4f}")
        print(f"Forgetting (Old Classes): {baseline_old - final_old:.4f}")
        
        # Save final results
        self.save_results(save_dir)
        
        return self.metrics
    
    def train_epoch(self, trainloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for batch in trainloader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(images)
            num_samples += len(images)
        
        return total_loss / num_samples
    
    def save_results(self, base_dir="CL"):
        """Save all experiment results including metrics and model."""
        outdir = os.path.join(base_dir, self.exp_name)
        os.makedirs(outdir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(outdir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, sort_keys=True)
        
        # Save metrics
        metrics_path = os.path.join(outdir, 'cl_metrics.json')
        
        # Convert per_class_accuracies to regular dict for JSON serialization
        metrics_to_save = {
            'train_losses': self.metrics['train_losses'],
            'old_class_accuracy': self.metrics['old_class_accuracy'],
            'new_class_accuracy': self.metrics['new_class_accuracy'],
            'overall_accuracy': self.metrics['overall_accuracy'],
            'per_class_accuracies': {k: v for k, v in self.metrics['per_class_accuracies'].items()},
            'epoch_times': self.metrics['epoch_times'],
            'baseline': self.metrics['baseline'],
            'class_names': CIFAR10_CLASSES
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4, sort_keys=True)
        
        # Save the model
        model_path = os.path.join(outdir, 'model_final.pt')
        torch.save(self.model.state_dict(), model_path)
        
        print(f"\nResults saved to: {outdir}")
        print(f"  - Config: {config_path}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Model: {model_path}")


def load_pretrained_model(model_path, config_path):
    """
    Load a pre-trained ViT model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pt file)
        config_path: Path to the config file (.json)
    
    Returns:
        model, config tuple
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = ViTForClassfication(config)
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning Experiment - Demonstrates Catastrophic Forgetting'
    )
    parser.add_argument('--pretrained-dir', type=str, required=True,
                        help='Directory containing pre-trained model (e.g., ViT_60_Epochs or ViT_100_Epochs)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: CL_{pretrained_dir})')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train on new classes')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (lower than original training to prevent catastrophic forgetting)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=0,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save-dir', type=str, default='CL',
                        help='Base directory for saving CL experiments')
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set experiment name
    if args.exp_name is None:
        pretrained_name = os.path.basename(os.path.normpath(args.pretrained_dir))
        args.exp_name = f"CL_{pretrained_name}"
    
    print("=" * 70)
    print("Continual Learning - Catastrophic Forgetting Experiment")
    print("=" * 70)
    print(f"Pre-trained model: {args.pretrained_dir}")
    print(f"Experiment name: {args.exp_name}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Load pre-trained model
    model_path = os.path.join(args.pretrained_dir, 'model_final.pt')
    config_path = os.path.join(args.pretrained_dir, 'config.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"\nLoading pre-trained model from {args.pretrained_dir}...")
    pretrained_model, config = load_pretrained_model(model_path, config_path)
    print(f"  Original model trained for {config.get('num_classes', 5)} classes")
    
    # Expand model to handle 10 classes
    print("\nExpanding model to 10 classes...")
    expanded_model, new_config = expand_model_for_new_classes(pretrained_model, config, num_total_classes=10)
    print(f"  Expanded classifier: {config['hidden_size']} → 10 classes")
    print(f"  Preserved weights for classes 0-4, initialized classes 5-9")
    
    # Prepare data loaders
    print("\nPreparing data loaders...")
    
    # Training data: new classes only (5-9) with remapped labels (0-4)
    trainloader_new, _, _ = prepare_data(
        batch_size=args.batch_size,
        classes_to_keep=NEW_CLASSES,
        remap_labels=True
    )
    
    # Test data: new classes only (5-9) with remapped labels (0-4)
    # We use remapped labels because the model was trained on remapped labels
    _, testloader_new, _ = prepare_data(
        batch_size=args.batch_size,
        classes_to_keep=NEW_CLASSES,
        remap_labels=True
    )
    
    # Test data: old classes only (0-4) with remapped labels (0-4)
    # Same remapping as original training
    _, testloader_old, _ = prepare_data(
        batch_size=args.batch_size,
        classes_to_keep=OLD_CLASSES,
        remap_labels=True
    )
    
    # Test data: all classes (0-9) - use remapped labels for consistent evaluation
    # Note: overall accuracy will be affected since model predicts 0-4 for all
    _, testloader_all, _ = prepare_data(
        batch_size=args.batch_size,
        classes_to_keep=list(range(10)),
        remap_labels=True
    )
    
    print(f"  Training samples (new classes): {len(trainloader_new.dataset)}")
    print(f"  Test samples (old classes): {len(testloader_old.dataset)}")
    print(f"  Test samples (new classes): {len(testloader_new.dataset)}")
    print(f"  Test samples (all classes): {len(testloader_all.dataset)}")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(expanded_model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create trainer and run experiment
    trainer = ContinualLearningTrainer(
        expanded_model, optimizer, loss_fn, 
        args.exp_name, device, new_config
    )
    
    metrics = trainer.train(
        trainloader_new=trainloader_new,
        testloader_old=testloader_old,
        testloader_new=testloader_new,
        testloader_all=testloader_all,
        epochs=args.epochs,
        save_model_every_n_epochs=args.save_every,
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 70)
    print("Continual Learning Experiment Complete!")
    print(f"Results saved to: {os.path.join(args.save_dir, args.exp_name)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
