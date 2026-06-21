"""
Continual Learning Experiment for ViT on CIFAR-10 with Attention Head Freezing

This script demonstrates catastrophic forgetting and mitigation by:
1. Loading a pre-trained ViT (trained on first 2 tasks = classes 0-3)
2. Sequentially learning tasks 3, 4, 5 (classes 4-5, 6-7, 8-9)
3. After each task, freezing the most important attention heads for previous tasks
4. Tracking performance on ALL tasks after each new task to show forgetting

Task Structure (CIFAR-10):
- Task 1: plane (0), car (1)
- Task 2: bird (2), cat (3)
- Task 3: deer (4), dog (5)
- Task 4: frog (6), horse (7)
- Task 5: ship (8), truck (9)

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

# Define 5 tasks with 2 classes each
TASKS = {
    1: {'classes': [0, 1], 'name': 'Task 1: Plane & Car'},
    2: {'classes': [2, 3], 'name': 'Task 2: Bird & Cat'},
    3: {'classes': [4, 5], 'name': 'Task 3: Deer & Dog'},
    4: {'classes': [6, 7], 'name': 'Task 4: Frog & Horse'},
    5: {'classes': [8, 9], 'name': 'Task 5: Ship & Truck'},
}

# Pre-trained tasks (first 2 tasks = 4 classes)
PRETRAINED_TASKS = [1, 2]
PRETRAINED_CLASSES = [0, 1, 2, 3]

# Continual learning tasks (remaining 3 tasks = 6 classes)
#CL_TASKS = [3, 4, 5] #train all remaining tasks
CL_TASKS = [3] #just train task 3

def get_classes_for_tasks(task_ids):
    """Get all class indices for a list of task IDs."""
    classes = []
    for task_id in task_ids:
        classes.extend(TASKS[task_id]['classes'])
    return sorted(classes)


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

    # Copy weights and biases for old classes
    with torch.no_grad():
        new_classifier.weight[:old_num_classes, :] = model.classifier.weight
        new_classifier.bias[:old_num_classes] = model.classifier.bias

    # Replace the classifier in the expanded model
    expanded_model.classifier = new_classifier

    # Update config for the new model
    new_config = config.copy()
    new_config['num_classes'] = num_total_classes
    new_config['original_num_classes'] = old_num_classes

    return expanded_model, new_config

def calculate_head_importance(
    model,
    data,
    batch_size,
    device=None,
    normalize_scores_by_layer=True,
    subset_size=1.0,
    task_classes=None,
    verbose=True,
    disable_progress_bar=False,
):

    """
    Compute attention-head importance scores according to Michel et al. Section 4.1.

    The importance score for one attention head h is:

        I_h = E_x | Att_h(x)^T * dL(x) / dAtt_h(x) |

    Intuition:
        - Att_h(x) is the output produced by head h.
        - dL(x) / dAtt_h(x) tells us how strongly the loss reacts to this head output.
        - A large absolute dot product means that the head has a strong influence on the loss.
        - A small value means that the head is probably less important.

    Requirements:
        The attention module must store the per-head attention output before merging the heads:

            self.context_layer_val = attention_output
            self.context_layer_val.retain_grad()

        The tensor must have shape:

            [batch_size, num_attention_heads, sequence_length, attention_head_size]

    Args: 
        model: ViT model whose attention modules store context_layer_val
        data: Dataset (not a DataLoader) used to estimate importance
        batch_size: Batch size for the internally built DataLoader
        device: Compute device; defaults to the model's device
        normalize_scores_by_layer: If True, normalize head scores inside each layer by L2 norm
        subset_size: Fraction (<=1) or absolute number of examples to use
        task_classes: Optional list mapping task-local labels back to their global dataset class indices (e.g. [4, 5] turns label 0 into class 4 and label 1 into class 5); None if labels are not remapped
        verbose: If True, print a short summary (number of examples, batch size, number of steps) before the computation
        disable_progress_bar: If True, suppress the tqdm progress bar

    Returns:
        Dictionary mapping (layer_idx, head_idx) to the corresponding importance score
    """

    # Store whether the model was originally in training mode.
    # We restore this state at the end.
    model_was_training = model.training

    # Use eval mode to disable dropout during importance estimation.
    # Important: eval() is okay, but torch.no_grad() must NOT be used.
    model.eval()

    device = device or next(model.parameters()).device

    # Access transformer blocks.
    # This assumes your model structure is: model.encoder.blocks
    blocks = model.encoder.blocks

    # Read number of layers and number of heads from the model.
    num_layers = len(blocks)
    num_heads = blocks[0].attention.num_attention_heads 

    #fix numbers of examples instead of number of batches, code from paper´s repo
    if subset_size <= 1:
        subset_size *= len(data)
    n_steps = int(np.ceil(int(subset_size) / batch_size))


    # Prepare data loader
    sampler = RandomSampler(data)
    dataloader = islice(
        DataLoader(
        data, 
        sampler=sampler, 
        batch_size=batch_size), 
    n_steps
    )
    prune_iterator = tqdm(
        dataloader, 
        desc="Iteration",
        disable=disable_progress_bar, 
        total=n_steps,
    )

    if verbose:
        print("***** Calculating head importance *****")
        print(f"  Num examples = {len(data)}")
        print(f"  Batch size   = {batch_size}")
        print(f"  Num steps    = {n_steps}")
    # Tensor that stores one importance value per layer and per head.
    # Shape: [num_layers, num_heads]
    head_importance = torch.zeros(num_layers, num_heads, device=device) #more efficient than the paper´s repo, create tensor on the used device, not first on cpu and then move to used device

    # Standard classification loss, paper uses sum for loss
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    # Convert task_classes once, not inside every batch.
    # Example: task_classes = [4, 5]
    # If labels are remapped to 0/1, then:
    # label 0 -> class 4
    # label 1 -> class 5
    if task_classes is not None:
        task_classes = torch.tensor(task_classes, device=device, dtype=torch.long)

    for _, batch in enumerate(prune_iterator):
        images, labels = (t.to(device) for t in batch)   # ViT batch = (image, label)
        labels = labels.long()
        if task_classes is not None:                      # only if labels are remapped
            labels = task_classes[labels]

        # ViT returns (logits, attn) 
        loss = loss_fn(model(images, output_attentions=False)[0], labels)
        loss.backward()

        for layer_idx, block in enumerate(blocks):
            attention_module = block.attention

            # Check whether the attention module stored the per-head output.
            if not hasattr(attention_module, "context_layer_val"):
                raise RuntimeError(
                    "context_layer_val was not found in the attention module. "
                    "Add self.context_layer_val = attention_output inside FasterMultiHeadAttention.forward() "
                    "directly after attention_output = torch.matmul(attention_probs, value)."
                )

            # ctx is the per-head attention output:
            # Shape: [batch_size, num_heads, sequence_length, head_dim]
            ctx = attention_module.context_layer_val

            # grad_ctx is dL / dctx:
            # Shape: [batch_size, num_heads, sequence_length, head_dim]
            grad_ctx = ctx.grad

            # If this is None, retain_grad() was not called or gradients were disabled.
            if grad_ctx is None:
                raise RuntimeError(
                    "context_layer_val.grad is None. "
                    "Check that retain_grad() is called on context_layer_val "
                    "and that the forward pass is not inside torch.no_grad()."
                )

            # Safety check: for the Michel score, ctx must still contain a separate head dimension.
            if ctx.dim() != 4:
                raise RuntimeError(
                    f"context_layer_val must have 4 dimensions "
                    f"[batch, heads, tokens, head_dim], but got shape {tuple(ctx.shape)}."
                )

            # Compute the dot product between head output and its gradient.
            #
            # ctx and grad_ctx shape:
            #     [batch, heads, tokens, head_dim]
            #
            # dot shape:
            #     [batch, heads, tokens]
            #
            # For every sample, head, and token:
            #     dot[b, h, l] = sum_d grad_ctx[b, h, l, d] * ctx[b, h, l, d]
            dot = torch.einsum("bhld,bhld->bhl", grad_ctx, ctx)

            # Take absolute value and sum over batch and tokens.
            #
            # dot.abs().sum(dim=(0, 2)) changes:
            #     [batch, heads, tokens] -> [heads]
            #
            # The result is accumulated for this layer.
            head_importance[layer_idx] += dot.abs().sum(dim=(0, 2)).detach()

    # Normalize importance scores inside each layer by L2 norm.
    # This follows the layer-wise normalization used in Michel et al.
    if normalize_scores_by_layer:
        norm_by_layer = head_importance.norm(p=2, dim=1, keepdim=True)
        head_importance = head_importance / (norm_by_layer + 1e-20)

    # Convert the tensor into the dictionary format used by the rest of the code.
    importance_scores = {}

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            importance_scores[(layer_idx, head_idx)] = head_importance[
                layer_idx, head_idx
            ].item()

    # Clear gradients after importance computation.
    model.zero_grad(set_to_none=True)

    # Restore training mode if the model was training before.
    if model_was_training:
        model.train()

    return importance_scores

def compute_attention_head_importance(model, dataloader, device, task_classes):
    """
    Compute the importance of each attention head for a specific task.
    Uses gradient-based importance estimation.

    Args:
        model: ViT model
        dataloader: DataLoader for the task
        device: Device to run on
        task_classes: List of class indices for this task

    Returns:
        Dictionary mapping (layer_idx, head_idx) to importance score
    """
    model.eval()
    importance_scores = defaultdict(float)

    # Enable gradients for attention parameters
    for name, param in model.named_parameters():
        if 'attention' in name or 'qkv' in name:
            param.requires_grad = True

    # Compute gradients on a batch of data
    num_batches = 0
    max_batches = 10  # Limit number of batches for efficiency

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass with attention output
        logits, all_attentions = model(images, output_attentions=True)

        # Compute loss only for task-specific classes
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward pass
        loss.backward()

        # Accumulate gradient magnitudes for attention heads
        # For models using FasterMultiHeadAttention, we need to look at the qkv_projection
        for layer_idx, block in enumerate(model.encoder.blocks):
            if hasattr(block.attention, 'heads'):
                # MultiHeadAttention with separate heads
                for head_idx, head in enumerate(block.attention.heads):
                    # Compute importance as sum of gradient magnitudes
                    grad_mag = 0.0
                    for param in [head.query, head.key, head.value]:
                        if param.weight.grad is not None:
                            grad_mag += param.weight.grad.abs().mean().item()
                    importance_scores[(layer_idx, head_idx)] += grad_mag
            elif hasattr(block.attention, 'qkv_projection'):
                # FasterMultiHeadAttention - compute per-head importance from qkv
                qkv_grad = block.attention.qkv_projection.weight.grad
                if qkv_grad is not None:
                    num_heads = block.attention.num_attention_heads
                    head_size = block.attention.attention_head_size
                    all_head_size = block.attention.all_head_size

                    # Split qkv_grad into q, k, v gradients
                    grad_chunks = torch.chunk(qkv_grad, 3, dim=0)

                    for head_idx in range(num_heads):
                        head_grad = 0.0
                        for grad_chunk in grad_chunks:
                            # Extract gradients for this head
                            head_start = head_idx * head_size
                            head_end = (head_idx + 1) * head_size
                            head_grad += grad_chunk[head_start:head_end, :].abs().mean().item()

                        importance_scores[(layer_idx, head_idx)] += head_grad

        num_batches += 1

    # Average over batches
    for key in importance_scores:
        importance_scores[key] /= num_batches

    return importance_scores


def freeze_attention_heads_for_tasks(model, task_importance_scores, freeze_ratio=0.3):
    """
    Freeze the most important attention heads for previously learned tasks.

    Args:
        model: ViT model
        task_importance_scores: Dict mapping task_id -> importance_scores dict
        freeze_ratio: Ratio of heads to freeze (default 0.3 = 30%)

    Returns:
        Set of frozen (layer_idx, head_idx) tuples
    """
    frozen_heads = set()

    # Collect all importance scores across tasks
    all_scores = defaultdict(list)
    for task_id, scores in task_importance_scores.items():
        for (layer_idx, head_idx), score in scores.items():
            all_scores[(layer_idx, head_idx)].append((task_id, score))

    # For each layer, freeze the top heads by cumulative importance
    num_heads = model.encoder.blocks[0].attention.num_attention_heads
    num_layers = len(model.encoder.blocks)
    heads_to_freeze_per_layer = max(1, int(num_heads * freeze_ratio))

    for layer_idx in range(num_layers):
        # Get scores for this layer
        layer_scores = []
        for head_idx in range(num_heads):
            key = (layer_idx, head_idx)
            if key in all_scores:
                # Sum importance across all tasks
                total_importance = sum(score for _, score in all_scores[key])
                layer_scores.append((head_idx, total_importance))

        # Sort by importance and freeze top heads
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        for head_idx, _ in layer_scores[:heads_to_freeze_per_layer]:
            frozen_heads.add((layer_idx, head_idx))

            # Freeze this head's parameters
            block = model.encoder.blocks[layer_idx]
            if hasattr(block.attention, 'heads'):
                # MultiHeadAttention
                for param in block.attention.heads[head_idx].parameters():
                    param.requires_grad = False
            elif hasattr(block.attention, 'qkv_projection'):
                # For FasterMultiHeadAttention, we freeze the corresponding slice
                # This is approximate - we freeze the entire qkv_projection for now
                # A more refined approach would freeze only specific weight slices
                pass  # Will handle with masks in training

    return frozen_heads


def apply_head_freezing_mask(model, frozen_heads):
    """
    Apply masks to zero out gradients for frozen attention heads.
    Covers both qkv_projection AND output_projection so that frozen heads
    are truly protected: neither their Q/K/V projections nor the output
    path through which they contribute to the residual stream can change.
    """
    for layer_idx, block in enumerate(model.encoder.blocks):
        if not hasattr(block.attention, 'qkv_projection'):
            continue

        attn = block.attention
        num_heads = attn.num_attention_heads
        head_size = attn.attention_head_size
        all_head_size = attn.all_head_size
        device = attn.qkv_projection.weight.device

        # mask for qkv_projection.weight : shape [all_head_size*3, hidden_size]
        qkv_mask = torch.ones(all_head_size * 3, 1, device=device)

        # mask for output_projection.weight : shape [hidden_size, all_head_size]
        out_mask = torch.ones(1, all_head_size, device=device)

        for head_idx in range(num_heads):
            if (layer_idx, head_idx) not in frozen_heads:
                continue
            for qkv_idx in range(3):
                start = qkv_idx * all_head_size + head_idx * head_size
                end = start + head_size
                qkv_mask[start:end, :] = 0.0

            col_start = head_idx * head_size
            col_end = col_start + head_size
            out_mask[:, col_start:col_end] = 0.0

        attn._frozen_qkv_mask = qkv_mask
        attn._frozen_out_mask = out_mask

        if attn.qkv_bias and attn.qkv_projection.bias is not None:
            attn._frozen_qkv_bias_mask = qkv_mask.squeeze(1)


def evaluate_with_detailed_metrics(model, testloaders_by_task, device):
    """
    Evaluate model with detailed per-task metrics.

    Args:
        model: The model to evaluate
        testloaders_by_task: Dict mapping task_id -> testloader for that task
        device: Device to run evaluation on

    Returns:
        Dictionary with per-task and overall metrics
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    task_metrics = {}
    total_correct = 0
    total_samples = 0

    for task_id, testloader in testloaders_by_task.items():
        task_correct = 0
        task_total = 0
        task_loss = 0.0

        # Per-class tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for batch in testloader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                # Get predictions
                logits, _ = model(images)

                # Calculate loss
                loss = loss_fn(logits, labels)
                task_loss += loss.item()

                # Calculate predictions
                predictions = torch.argmax(logits, dim=1)

                # Track accuracy
                for pred, true_label in zip(predictions, labels):
                    true_label_item = true_label.item()
                    class_total[true_label_item] += 1
                    if pred == true_label:
                        class_correct[true_label_item] += 1
                        task_correct += 1
                    task_total += 1

        # Calculate task accuracy
        task_accuracy = task_correct / task_total if task_total > 0 else 0
        avg_task_loss = task_loss / task_total if task_total > 0 else 0

        # Calculate per-class accuracies
        per_class_accuracy = {}
        for cls in sorted(class_total.keys()):
            if class_total[cls] > 0:
                per_class_accuracy[cls] = class_correct[cls] / class_total[cls]
            else:
                per_class_accuracy[cls] = 0.0

        task_metrics[task_id] = {
            'accuracy': task_accuracy,
            'loss': avg_task_loss,
            'correct': task_correct,
            'total': task_total,
            'per_class_accuracy': per_class_accuracy
        }

        total_correct += task_correct
        total_samples += task_total

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    return {
        'task_metrics': task_metrics,
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_samples': total_samples
    }


class TaskBasedContinualLearningTrainer:
    """
    Trainer for task-based continual learning that tracks forgetting across all tasks.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.config = config

        # Metrics storage - tracks performance on all tasks after each new task
        self.metrics_history = {
            'after_task_2': {},  # Baseline (pre-trained on tasks 1-2)
            'after_task_3': {},
            'after_task_4': {},
            'after_task_5': {},
        }

        # Track frozen heads after each task
        self.frozen_heads_history = {}
        self.task_importance_scores = {}

        # Track which tasks have been learned
        self.learned_tasks = []

    def train_on_task(self, task_id, trainloader, testloaders_by_task, epochs,
                      save_model_every_n_epochs=0, save_dir="CL",
                      freeze_heads_after=False, freeze_ratio=0.3):
        """
        Train on a specific task while tracking performance on all tasks.

        Args:
            task_id: The task ID being learned
            trainloader: DataLoader for the current task's training data
            testloaders_by_task: Dict of testloaders for all tasks
            epochs: Number of training epochs
            save_model_every_n_epochs: Save checkpoint every N epochs
            save_dir: Base directory for saving results
            freeze_heads_after: Whether to freeze important heads after training
            freeze_ratio: Ratio of heads to freeze
        """
        print("=" * 80)
        print(f"Learning {TASKS[task_id]['name']}")
        print(f"Classes: {[CIFAR10_CLASSES[c] for c in TASKS[task_id]['classes']]}")
        print(f"Already learned: {[TASKS[t]['name'] for t in self.learned_tasks]}")
        print("=" * 80)

        # Add current task to learned tasks
        self.learned_tasks.append(task_id)

        # Training loop
        total_start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train on current task only
            train_loss = self.train_epoch(trainloader, task_classes=TASKS[task_id]['classes'])

            epoch_time = time.time() - epoch_start

            # Evaluate on all tasks every few epochs and at the end
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                metrics = evaluate_with_detailed_metrics(
                    self.model, testloaders_by_task, self.device
                )

                # Print progress
                print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - Train Loss: {train_loss:.4f}")
                for tid in sorted(metrics['task_metrics'].keys()):
                    acc = metrics['task_metrics'][tid]['accuracy']
                    if tid in self.learned_tasks:
                        print(f"  Task {tid}: {acc:.4f} ✓")
                    else:
                        print(f"  Task {tid}: {acc:.4f}")
                print(f"  Overall: {metrics['overall_accuracy']:.4f}")

            # Save checkpoint if requested
            if save_model_every_n_epochs > 0 and (epoch+1) % save_model_every_n_epochs == 0:
                checkpoint_dir = os.path.join(save_dir, self.exp_name, f"checkpoints_task{task_id}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(self.exp_name, self.model, f"task{task_id}_epoch_{epoch+1}", base_dir=save_dir)

        # Calculate total training time
        total_duration = time.time() - total_start_time
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)

        print(f"\nTask {task_id} Training Complete! Time: {minutes}m {seconds}s")

        # Final evaluation after this task
        final_metrics = evaluate_with_detailed_metrics(
            self.model, testloaders_by_task, self.device
        )

        # Store metrics
        key = f'after_task_{task_id}'
        self.metrics_history[key] = final_metrics

        # Compute attention head importance for this task
        print(f"\nComputing attention head importance for Task {task_id}...")
        importance_scores = calculate_head_importance(
            self.model, trainloader, self.device, task_classes=None
        )
        self.task_importance_scores[task_id] = importance_scores

        # Freeze important heads for previous tasks if requested
        #if freeze_heads_after and task_id > 2:  # Only freeze after learning beyond pre-trained tasks
        if freeze_heads_after and task_id > 3:
            print(f"\nFreezing most important attention heads (freeze_ratio={freeze_ratio})...")
            frozen_heads = freeze_attention_heads_for_tasks(
                self.model, self.task_importance_scores, freeze_ratio
            )
            apply_head_freezing_mask(self.model, frozen_heads)
            self.frozen_heads_history[task_id] = frozen_heads
            print(f"  Frozen {len(frozen_heads)} attention heads")

        # Print summary of forgetting
        self.print_forgetting_summary(task_id)

        return final_metrics

    def train_epoch(self, trainloader, task_classes=None):
        """Train for one epoch.

        Args:
            trainloader: DataLoader for the current task.
            task_classes: List of CIFAR-10 class indices for the current task
                          (e.g. [4, 5] for deer & dog).  When provided, the loss
                          is computed only over those output neurons so that the
                          softmax does not suppress neurons belonging to previously
                          learned tasks.
        """
        self.model.train()
        total_loss = 0
        num_samples = 0

        # Build a tensor of class indices for logit slicing (done once per epoch)
        if task_classes is not None:
            class_indices = torch.tensor(task_classes, device=self.device)

        for batch in trainloader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits, _ = self.model(images)

            if task_classes is not None:
                # Slice out only the logits for the current task's classes.
                # This prevents CrossEntropyLoss from suppressing the output
                # neurons of previously learned tasks via its softmax denominator.
                # Labels are already the original CIFAR-10 indices (remap_labels=False),
                # so we remap them to local positions [0, len(task_classes)) here.
                task_logits = logits[:, class_indices]
                local_labels = torch.zeros_like(labels)
                for local_idx, global_cls in enumerate(task_classes):
                    local_labels[labels == global_cls] = local_idx
                loss = self.loss_fn(task_logits, local_labels)
            else:
                loss = self.loss_fn(logits, labels)

            # Backward pass
            loss.backward()

            # Apply gradient masks for frozen heads (qkv_projection + output_projection + bias)
            for block in self.model.encoder.blocks:
                attn = block.attention
                if not hasattr(attn, '_frozen_qkv_mask'):
                    continue
                if attn.qkv_projection.weight.grad is not None:
                    attn.qkv_projection.weight.grad.mul_(
                        attn._frozen_qkv_mask.expand_as(attn.qkv_projection.weight.grad)
                    )
                if hasattr(attn, '_frozen_qkv_bias_mask') and attn.qkv_projection.bias is not None \
                        and attn.qkv_projection.bias.grad is not None:
                    attn.qkv_projection.bias.grad.mul_(attn._frozen_qkv_bias_mask)
                if attn.output_projection.weight.grad is not None:
                    attn.output_projection.weight.grad.mul_(
                        attn._frozen_out_mask.expand_as(attn.output_projection.weight.grad)
                    )

            self.optimizer.step()

            total_loss += loss.item() * len(images)
            num_samples += len(images)

        return total_loss / num_samples

    def print_forgetting_summary(self, current_task_id):
        """Print a summary of forgetting across tasks."""
        print("\n" + "=" * 80)
        print(f"FORGETTING SUMMARY AFTER LEARNING TASK {current_task_id}")
        print("=" * 80)

        # Compare with baseline (after_task_2)
        baseline_key = 'after_task_2'

        for task_id in range(1, 6):
            if task_id in [1, 2]:
                # Pre-trained tasks - show forgetting
                current_key = f'after_task_{current_task_id}'
                if baseline_key in self.metrics_history and current_key in self.metrics_history:
                    baseline_acc = self.metrics_history[baseline_key]['task_metrics'][task_id]['accuracy']
                    current_acc = self.metrics_history[current_key]['task_metrics'][task_id]['accuracy']
                    forgetting = baseline_acc - current_acc
                    print(f"Task {task_id} ({TASKS[task_id]['name']}): "
                          f"{baseline_acc:.4f} → {current_acc:.4f} "
                          f"(forgetting: {forgetting:+.4f})")
            else:
                # New tasks - show learning progress
                current_key = f'after_task_{current_task_id}'
                if current_key in self.metrics_history:
                    current_acc = self.metrics_history[current_key]['task_metrics'][task_id]['accuracy']
                    status = "✓ learned" if task_id <= current_task_id else "not yet learned"
                    print(f"Task {task_id} ({TASKS[task_id]['name']}): "
                          f"{current_acc:.4f} ({status})")

        print("=" * 80)

    def save_results(self, base_dir="CL"):
        """Save all experiment results including metrics and model."""
        outdir = os.path.join(base_dir, self.exp_name)
        os.makedirs(outdir, exist_ok=True)

        # Save config
        config_path = os.path.join(outdir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, sort_keys=True)

        # Save metrics history
        metrics_path = os.path.join(outdir, 'cl_metrics.json')

        # Convert metrics to JSON-serializable format
        metrics_to_save = {}
        for key, metrics in self.metrics_history.items():
            if metrics:  # Skip empty entries
                metrics_to_save[key] = {
                    'overall_accuracy': metrics.get('overall_accuracy', 0),
                    'task_metrics': {
                        str(tid): {
                            'accuracy': tm['accuracy'],
                            'loss': tm['loss'],
                            'per_class_accuracy': {str(k): v for k, v in tm['per_class_accuracy'].items()}
                        }
                        for tid, tm in metrics.get('task_metrics', {}).items()
                    }
                }

        # Add task names and frozen heads history
        metrics_to_save['tasks'] = {str(k): v for k, v in TASKS.items()}
        metrics_to_save['class_names'] = CIFAR10_CLASSES
        metrics_to_save['frozen_heads'] = {
            str(k): [list(h) for h in v] for k, v in self.frozen_heads_history.items()
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
        description='Task-Based Continual Learning with Attention Head Freezing'
    )
    parser.add_argument('--pretrained-dir', type=str, required=True,
                        help='Directory containing pre-trained model (e.g., ViT_60_Epochs)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: CL_Tasks_{pretrained_dir})')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs-per-task', type=int, default=20,
                        help='Number of epochs to train on each new task')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=0,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save-dir', type=str, default='CL',
                        help='Base directory for saving CL experiments')
    parser.add_argument('--freeze-heads', action='store_true',
                        help='Enable attention head freezing after each task')
    parser.add_argument('--freeze-ratio', type=float, default=0.3,
                        help='Ratio of attention heads to freeze (default: 0.3)')

    args = parser.parse_args()

    # Determine device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Set experiment name
    if args.exp_name is None:
        pretrained_name = os.path.basename(os.path.normpath(args.pretrained_dir))
        args.exp_name = f"CL_Tasks_{pretrained_name}"

    print("=" * 80)
    print("Task-Based Continual Learning Experiment")
    print("=" * 80)
    print(f"Pre-trained model: {args.pretrained_dir}")
    print(f"Experiment name: {args.exp_name}")
    print(f"Device: {device}")
    print(f"Freeze heads: {args.freeze_heads} (ratio: {args.freeze_ratio})")
    print("=" * 80)
    print("\nTask Structure:")
    for task_id, task_info in TASKS.items():
        classes_str = ', '.join([CIFAR10_CLASSES[c] for c in task_info['classes']])
        status = "PRE-TRAINED" if task_id in PRETRAINED_TASKS else "TO LEARN"
        print(f"  Task {task_id}: {classes_str} ({status})")
    print("=" * 80)

    # Load pre-trained model (trained on tasks 1-2)
    model_path = os.path.join(args.pretrained_dir, 'model_final.pt')
    config_path = os.path.join(args.pretrained_dir, 'config.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"\nLoading pre-trained model from {args.pretrained_dir}...")
    pretrained_model, config = load_pretrained_model(model_path, config_path)
    print(f"  Original model trained for {config.get('num_classes', 4)} classes")
    print(f"  (Tasks 1-2: {PRETRAINED_CLASSES})")

    # Expand model to handle 10 classes
    print("\nExpanding model to 10 classes...")
    expanded_model, new_config = expand_model_for_new_classes(
        pretrained_model, config, num_total_classes=10
    )
    print(f"  Expanded classifier: {config['hidden_size']} → 10 classes")
    print(f"  Preserved weights for classes 0-3 (Tasks 1-2)")

    # Move model to device
    expanded_model = expanded_model.to(device)

    # Prepare test data loaders for ALL tasks
    print("\nPreparing test data loaders for all tasks...")
    testloaders_by_task = {}
    for task_id in range(1, 6):
        _, testloader, _ = prepare_data(
            batch_size=args.batch_size,
            classes_to_keep=TASKS[task_id]['classes'],
            remap_labels=False  # keep original CIFAR-10 indices so they match the 10-class model output
        )
        testloaders_by_task[task_id] = testloader
        print(f"  Task {task_id}: {len(testloader.dataset)} test samples")

    # Evaluate baseline (pre-trained on tasks 1-2)
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION (Pre-trained on Tasks 1-2)")
    print("=" * 80)

    baseline_metrics = evaluate_with_detailed_metrics(
        expanded_model, testloaders_by_task, device
    )

    for task_id in range(1, 6):
        acc = baseline_metrics['task_metrics'][task_id]['accuracy']
        if task_id in PRETRAINED_TASKS:
            print(f"Task {task_id}: {acc:.4f} (pre-trained)")
        else:
            print(f"Task {task_id}: {acc:.4f} (random - not yet learned)")

    print(f"\nOverall Accuracy: {baseline_metrics['overall_accuracy']:.4f}")

    # Setup optimizer and loss
    optimizer = optim.AdamW(expanded_model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # Create trainer
    trainer = TaskBasedContinualLearningTrainer(
        expanded_model, optimizer, loss_fn,
        args.exp_name, device, new_config
    )

    # Store baseline metrics
    trainer.metrics_history['after_task_2'] = baseline_metrics
    trainer.learned_tasks = [1, 2]  # Mark tasks 1-2 as learned

    # we freeze heads before starting to learn task 3    
    if args.freeze_heads:
        print("Computing head importance for pre-trained tasks 1 & 2...")
        for task_id in PRETRAINED_TASKS:
            trainloader_pretrained, _, _ = prepare_data(
                batch_size=args.batch_size,
                classes_to_keep=TASKS[task_id]['classes'],
                remap_labels=False
            )
            importance = calculate_head_importance(
                expanded_model, trainloader_pretrained, device, task_classes=None
            )
            trainer.task_importance_scores[task_id] = importance

        #freeze heads before training on new tasks (3-5) to prevent forgetting of pre-trained tasks (1-2)
        frozen_heads = freeze_attention_heads_for_tasks(
            expanded_model, trainer.task_importance_scores, freeze_ratio=args.freeze_ratio
        )
        apply_head_freezing_mask(expanded_model, frozen_heads)
        trainer.frozen_heads_history[0] = frozen_heads
        print(f"Frozen {len(frozen_heads)} heads before training Task 3")

    # Sequentially learn tasks 3, 4, 5
    for task_id in CL_TASKS:
        # Prepare training data for this task
        trainloader, _, _ = prepare_data(
            batch_size=args.batch_size,
            classes_to_keep=TASKS[task_id]['classes'],
            remap_labels=False  # keep original CIFAR-10 indices to match the 10-class model output
        )
        print(f"\nTraining samples for Task {task_id}: {len(trainloader.dataset)}")

        # Train on this task
        trainer.train_on_task(
            task_id=task_id,
            trainloader=trainloader,
            testloaders_by_task=testloaders_by_task,
            epochs=args.epochs_per_task,
            save_model_every_n_epochs=args.save_every,
            save_dir=args.save_dir,
            freeze_heads_after=args.freeze_heads,
            freeze_ratio=args.freeze_ratio
        )

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - Catastrophic Forgetting Analysis")
    print("=" * 80)

    # Compare baseline with final
    baseline = trainer.metrics_history['after_task_2']
    last_task = CL_TASKS[-1] # the last task learned (e.g., 3, 4, or 5)
    final = trainer.metrics_history.get(f'after_task_{last_task}', {})

    if baseline and final:
        print(f"Overall Accuracy: {baseline['overall_accuracy']:.4f} → {final['overall_accuracy']:.4f}") #overall accuracy before and after continual learning

    print("\nAccuracy on each task after learning all tasks:")
    print(f"{'Task':<30} {'After Task 2':<15} {'After All Tasks':<15} {'Change':<15}")
    print("-" * 80)

    for task_id in range(1, 6):
        task_name = TASKS[task_id]['name']
        baseline_acc = baseline['task_metrics'][task_id]['accuracy']
        final_acc = final['task_metrics'][task_id]['accuracy']
        change = final_acc - baseline_acc

        print(f"{task_name:<30} {baseline_acc:<15.4f} {final_acc:<15.4f} {change:+<15.4f}")

    # Calculate average forgetting on pre-trained tasks
    forgetting_scores = []
    for task_id in PRETRAINED_TASKS:
        baseline_acc = baseline['task_metrics'][task_id]['accuracy']
        final_acc = final['task_metrics'][task_id]['accuracy']
        forgetting = baseline_acc - final_acc
        forgetting_scores.append(forgetting)

    avg_forgetting = np.mean(forgetting_scores)
    print(f"\nAverage Forgetting on Pre-trained Tasks (1-2): {avg_forgetting:.4f}")
    print(f"Overall Accuracy: {baseline['overall_accuracy']:.4f} → {final['overall_accuracy']:.4f}")

    # Save final results
    trainer.save_results(args.save_dir)

    print("\n" + "=" * 80)
    print("Task-Based Continual Learning Experiment Complete!")
    print(f"Results saved to: {os.path.join(args.save_dir, args.exp_name)}")
    print("=" * 80)


if __name__ == "__main__":
    main()