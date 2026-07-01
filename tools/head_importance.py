import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_attention_modules(model):
    """
    Find all attention modules that have head_alpha.
    For ViT-B/16, this should usually be 12 modules,
    each with 12 heads.
    """
    modules = []

    for module in model.modules():
        if hasattr(module, "head_alpha"):
            modules.append(module)

    return modules


def freeze_except_head_alpha(model):
    """
    Freeze the full model.
    Only head_alpha will receive gradients.

    This is not normal training.
    This is only for measuring head importance.
    """
    for _, param in model.named_parameters():
        param.requires_grad = False

    for module in get_attention_modules(model):
        module.head_alpha.requires_grad = True
        module.head_alpha.data.fill_(1.0)


def disable_head_alpha(model):
    """
    Disable head_alpha after diagnostic so it is not used by normal training.
    Also clear gradients, otherwise ModAdam may still see old alpha gradients.
    """
    for module in get_attention_modules(model):
        module.head_alpha.requires_grad_(False)
        module.head_alpha.grad = None
        module.head_alpha.data.fill_(1.0)


def compute_head_importance(model, dataloader, device, num_batches=100):
    """
    Compute gradient-based head importance.

    We added:
        head_output = head_output * alpha

    Then we compute:
        d(loss) / d(alpha)

    A large absolute gradient means that head is important
    for the current task.

    Returns:
        importance: Tensor with shape [num_layers, num_heads]
    """
    model.to(device)
    model.train()

    freeze_except_head_alpha(model)

    attention_modules = get_attention_modules(model)

    if len(attention_modules) == 0:
        raise RuntimeError(
            "No modules with head_alpha found. "
            "Check utils/vit_builder.py and make sure self.head_alpha was added."
        )

    print(f"Found {len(attention_modules)} attention modules with head_alpha.")

    importance = [
        torch.zeros_like(module.head_alpha.detach(), device=device)
        for module in attention_modules
    ]

    alpha_params = [module.head_alpha for module in attention_modules]

    # Optimizer is only used for zero_grad.
    # We do not need optimizer.step() for importance measurement.
    optimizer = torch.optim.SGD(alpha_params, lr=0.01)

    used_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing head importance")):
        if batch_idx >= num_batches:
            break

        images = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        # Some models return tuple/list. Use first output.
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        for i, module in enumerate(attention_modules):
            if module.head_alpha.grad is not None:
                importance[i] += module.head_alpha.grad.detach().abs()

        used_batches += 1

    if used_batches == 0:
        raise RuntimeError("No batches were processed. Check the dataloader.")

    importance = torch.stack(importance, dim=0)
    importance = importance / used_batches

    # Normalize to [0, 1] for easier comparison.
    max_value = importance.max()
    if max_value > 0:
        importance = importance / max_value

    disable_head_alpha(model)

    return importance.detach().cpu()


def save_head_importance(importance, output_path):
    """
    Save the importance matrix to a .pt file.
    """
    output_dir = os.path.dirname(output_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    torch.save(importance, output_path)

    print(f"Saved head importance to: {output_path}")
    print(f"Shape: {tuple(importance.shape)}")
    print(importance)


def select_topk_heads(importance, k=6):
    """
    Select top-k most important heads.

    Returns:
        list of (layer, head, score)
    """
    flat = importance.flatten()
    values, indices = torch.topk(flat, k)

    num_heads = importance.shape[1]
    result = []

    for value, index in zip(values, indices):
        layer = int(index // num_heads)
        head = int(index % num_heads)
        score = float(value)
        result.append((layer, head, score))

    return result


def print_topk_heads(importance, k=6):
    """
    Print top-k heads in readable form.
    """
    top_heads = select_topk_heads(importance, k=k)

    print(f"Top {k} important heads:")
    for layer, head, score in top_heads:
        print(f"Layer {layer}, Head {head}, Score {score:.4f}")