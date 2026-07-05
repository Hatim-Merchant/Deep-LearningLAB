# =============================================================================
# head_freeze.py — Michel et al. (2019) inspired head-freezing for continual
# learning, adapted to the timm ViT-B/16 used by AttentionRetentionCL.
#
# Method overview:
#   Michel et al. use the importance score I_h to identify the least important
#   attention heads and prune them at inference time. We invert this: we use
#   the same score to identify the most important heads and freeze their weight
#   slices (qkv, proj) so they are not updated on subsequent tasks. All heads
#   remain active in the forward pass; parameter count and inference speed are
#   unchanged. This is a deliberate methodological inversion for the purpose of
#   studying catastrophic forgetting in continual learning.
#
# Attribute name mapping from our small ViT (branch: Second-ViT-tintn) to the
# timm ViT-B/16 used here:
#   model.encoder.blocks            -> model.blocks
#   block.attention                 -> block.attn
#   attention.num_attention_heads   -> attn.num_heads
#   attention.attention_head_size   -> attn.head_dim
#   attention.all_head_size         -> attn.num_heads * attn.head_dim
#   attention.qkv_projection        -> attn.qkv          ([3*dim, dim])
#   attention.output_projection     -> attn.proj          ([dim, dim])
#   attention.context_layer_val     -> attn.out           ([B, H, N, head_dim])
#   model(x, output_attentions=False)[0] -> model(x)
#
# Public API (called from train_eval.py and utils/trainer.py):
#   calculate_head_importance(...)      -> dict {(layer, head): score}
#   freeze_attention_heads_for_tasks(...) -> set {(layer, head)}
#   build_head_freezing_mask(...)       -> None  (stores masks on attn modules)
#   freeze_apply_grad_mask(...)         -> None  (per step, before optimizer.step)
#   freeze_restore_weights(...)         -> None  (per step, after optimizer.step)
# =============================================================================

from collections import defaultdict
from itertools import islice
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

import random

# Type aliases for readability
HeadKey = Tuple[int, int]                  # (layer_idx, head_idx)
ImportanceDict = Dict[HeadKey, float]       # {(layer, head): score}


def calculate_head_importance(
    model: nn.Module,
    data: Dataset,
    batch_size: int,
    device: Optional[str] = None,
    normalize_scores_by_layer: bool = True,
    subset_size: float = 1.0,
    task_classes: Optional[list] = None,
    verbose: bool = True,
    disable_progress_bar: bool = False,
) -> ImportanceDict:
    """Compute attention-head importance scores according to Michel et al. Section 4.1.


    Importance formula:
        I_h = E_x | Att_h(x)^T * dL(x) / dAtt_h(x) |

    where Att_h(x) is the output of head h (context vector after softmax-weighted
    value aggregation). This equals the absolute dot product between the head
    output and its gradient, summed over tokens and averaged over samples.

    Implementation note:
        The timm Attention.forward stores Att_h(x) as `attn.out` with shape
        [B, H, N, head_dim] and calls retain_grad() on it — but only when
        GVM().training is True. This function forces that flag on for its
        duration and restores it afterwards.

    Args:
        model:                    timm VisionTransformer; qkv.weight must have
                                  requires_grad=True.
        data:                     PyTorch Dataset (not a DataLoader). Labels must
                                  be in [0, num_classes) matching the model head.
        batch_size:               Batch size for the internally created loader.
        device:                   Compute device string; defaults to the model's
                                  current device.
        normalize_scores_by_layer: L2-normalise scores within each layer so that
                                  layers with different gradient magnitudes are
                                  comparable (recommended by Michel et al.).
        subset_size:              Fraction <=1 of the dataset to use, or an
                                  absolute integer count. Using a subset speeds
                                  up importance estimation at a small cost in
                                  accuracy.
        task_classes:             Optional list mapping local label indices to
                                  global class indices. Pass None when the dataset
                                  labels already match the model head (the normal
                                  case in this codebase).
        verbose:                  Print sample / step counts to stdout.
        disable_progress_bar:     Suppress the tqdm progress bar.

    Returns:
        Dictionary mapping (layer_idx, head_idx) to a scalar importance score.
        Scores are non-negative; higher means more important.
    """
    # Why Import GlobalVarsManager here? To avoid a circular dependency at module load time
    # (head_freeze <- utils.gvm, not the other way around).
    from utils.gvm import GlobalVarsManager

    # GVM.training controls whether the timm Attention.forward stores attn.out.
    # We need it True so retain_grad() is called and we can read attn.out.grad.
    gvm = GlobalVarsManager()
    prev_gvm_training = gvm.training
    gvm.training = True

    # Switch to eval mode to disable dropout, but keep gradients flowing so
    # that loss.backward() populates attn.out.grad.
    model_was_training = model.training
    model.eval()

    device = device or str(next(model.parameters()).device)

    blocks = model.blocks
    num_layers = len(blocks)
    num_heads = blocks[0].attn.num_heads

    # Convert fractional subset_size to an absolute sample count.
    if subset_size <= 1.0:
        subset_size = int(subset_size * len(data))
    n_steps = int(np.ceil(int(subset_size) / batch_size))

    sampler = RandomSampler(data)
    # islice caps the DataLoader at n_steps batches without re-creating it.
    dataloader = islice(DataLoader(data, sampler=sampler, batch_size=batch_size), n_steps)
    iterator = tqdm(dataloader, desc="head-importance", disable=disable_progress_bar, total=n_steps)

    if verbose:
        print("***** Calculating head importance *****")
        print(f"  Num examples = {len(data)}")
        print(f"  Batch size   = {batch_size}")
        print(f"  Num steps    = {n_steps}")

    # Accumulate importance scores over all batches; shape [num_layers, num_heads].
    head_importance = torch.zeros(num_layers, num_heads, device=device)
    # Use sum reduction so that the magnitude is consistent across batch sizes.
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    if task_classes is not None:
        task_classes_tensor = torch.tensor(task_classes, device=device, dtype=torch.long)

    for batch in iterator:
        images, labels = (t.to(device) for t in batch)
        labels = labels.long()

        if task_classes is not None:
            labels = task_classes_tensor[labels]

        logits = model(images)          # timm ViT: returns logits Tensor directly
        loss = loss_fn(logits, labels)
        loss.backward()

        for layer_idx, block in enumerate(blocks):
            attn_module = block.attn

            if not hasattr(attn_module, "out"):
                raise RuntimeError(
                    f"block[{layer_idx}].attn.out not found. "
                    "The timm Attention.forward only stores this when "
                    "GVM().training is True — verify the flag is set."
                )
            ctx = attn_module.out           # [B, H, N, head_dim]: per-head context
            grad_ctx = ctx.grad             # dL / d(ctx): same shape

            if grad_ctx is None:
                raise RuntimeError(
                    f"block[{layer_idx}].attn.out.grad is None. "
                    "Ensure qkv.weight.requires_grad=True and that the forward "
                    "pass is not wrapped in torch.no_grad()."
                )
            if ctx.dim() != 4:
                raise RuntimeError(
                    f"attn.out must be 4-D [B, H, N, head_dim], got {tuple(ctx.shape)}."
                )

            # Compute |Att_h^T * dL/dAtt_h| per head, summed over batch and tokens.
            # einsum 'bhld,bhld->bhl' = element-wise product, then sum over head_dim.
            dot = torch.einsum("bhld,bhld->bhl", grad_ctx.float(), ctx.float())
            head_importance[layer_idx] += dot.abs().sum(dim=(0, 2)).detach()

        # Clear gradients after each batch to avoid accumulation across batches.
        model.zero_grad(set_to_none=True)

    if normalize_scores_by_layer:
        # L2 normalise within each layer so inter-layer scores are comparable.
        norm_by_layer = head_importance.norm(p=2, dim=1, keepdim=True)
        head_importance = head_importance / (norm_by_layer + 1e-20)

    # Convert to a plain dict for easy serialisation and lookup.
    importance_scores: ImportanceDict = {
        (layer_idx, head_idx): head_importance[layer_idx, head_idx].item()
        for layer_idx in range(num_layers)
        for head_idx in range(num_heads)
    }

    # Restore model and GVM to their original states.
    model.zero_grad(set_to_none=True)
    if model_was_training:
        model.train()
    gvm.training = prev_gvm_training

    return importance_scores


def freeze_attention_heads_for_tasks(
    model: nn.Module,
    task_importance_scores: Dict[int, ImportanceDict],
    freeze_ratio: float = 0.1,
    already_frozen: Optional[Set[HeadKey]] = None,
    selection: str = "michel",
    seed: Optional[int] = None,
) -> Set[HeadKey]:
    """Select the heads to freeze after the current task and return the new ones.

    Freezing is cumulative (monotonic): once a head is frozen it stays frozen
    for all subsequent tasks. The caller is responsible for accumulating the
    returned set into the running frozen set and passing it back as
    already_frozen on the next call.

    Selection criterion:
        For each head, sum its importance scores across all tasks seen so far.
        Pick the top n_to_freeze = int(total_heads * freeze_ratio) heads that
        are not already frozen.

    ViT-B/16 arithmetic (12 layers × 12 heads = 144 total heads):
        freeze_ratio=0.1  -> 14 new heads per task  (safe for 10-split)
        freeze_ratio=0.3  -> 43 new heads per task  (100% frozen after ~3 tasks)
        Use a small ratio for long task sequences.
    Args:
        model:                  timm VisionTransformer (used only to read the
                                number of layers and heads).
        task_importance_scores: dict  task_id -> ImportanceDict, as returned by
                                calculate_head_importance for each task.
        freeze_ratio:           Fraction of ALL heads to newly freeze per call.
        already_frozen:         Set of (layer_idx, head_idx) already frozen;
                                these are excluded from selection.
        selection:              "michel" (cumulative importance summed across
                                        all tasks), "michel_current" (importance of the
                                        just-finished task only), or "random".
        seed:                   Optional RNG seed for reproducible random selection.

    Returns:
        Set of newly selected (layer_idx, head_idx) pairs to freeze.
        Does not include already_frozen entries.
    """
    already_frozen = already_frozen or set()

    # Aggregate scores per head across all tasks seen so far.
    all_scores: Dict[HeadKey, list] = defaultdict(list)
    for _, scores in task_importance_scores.items():
        for (layer_idx, head_idx), score in scores.items():
            all_scores[(layer_idx, head_idx)].append(score)

    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads
    total_heads = num_layers * num_heads
    n_to_freeze = int(total_heads * freeze_ratio)

    if selection == "michel":
        # Build a list of (head_key, summed_score) for candidates not yet frozen.
        candidates = [
            ((layer_idx, head_idx), sum(all_scores[(layer_idx, head_idx)]))
            for layer_idx in range(num_layers)
            for head_idx in range(num_heads)
            if (layer_idx, head_idx) in all_scores
            and (layer_idx, head_idx) not in already_frozen
        ]
        # Sort descending: highest cumulative importance first.
        candidates.sort(key=lambda x: x[1], reverse=True)

        new_frozen: Set[HeadKey] = {key for key, _ in candidates[:n_to_freeze]}

    elif selection == "michel_current":
            # Task-specific variant: freeze the heads most important for the task
            # that just finished, not the cumulative top heads across all tasks.
            # Rationale: protect each task's specialist heads from being overwritten
            # by later tasks. task_importance_scores is insertion-ordered by task,
            # so its last value belongs to the current task.
            current_scores = next(reversed(task_importance_scores.values()))
            candidates = [
                ((layer_idx, head_idx), current_scores[(layer_idx, head_idx)])
                for layer_idx in range(num_layers)
                for head_idx in range(num_heads)
                if (layer_idx, head_idx) in current_scores
                and (layer_idx, head_idx) not in already_frozen
            ]
            # Sort descending: highest current-task importance first.
            candidates.sort(key=lambda x: x[1], reverse=True)
            new_frozen = {key for key, _ in candidates[:n_to_freeze]}

    elif selection == "random":
        # Random baseline: freeze a random subset of the 12x12 head grid,
        # ignoring the importance scores entirely. Every head that is not
        # already frozen is an equally likely candidate. The scores are still
        # computed upstream and logged, but they do NOT influence this choice.
        candidate_keys = [
            (layer_idx, head_idx)
            for layer_idx in range(num_layers)
            for head_idx in range(num_heads)
            if (layer_idx, head_idx) not in already_frozen
        ]
        # Clamp so we never request more heads than remain available.
        n_sample = min(n_to_freeze, len(candidate_keys))
        # Dedicated RNG: reproducible and independent of the global RNG state
        # consumed during training. Offsetting the seed by the number of tasks
        # seen so far makes each task draw a different but deterministic subset.
        rng = random.Random(
            None if seed is None else seed + len(task_importance_scores)
        )
        new_frozen = set(rng.sample(candidate_keys, n_sample))

    else:
        raise ValueError(
            f"Unknown head selection '{selection}'. Use 'michel', 'michel_current' or 'random'."
        )
    return new_frozen


def build_head_freezing_mask(model: nn.Module, frozen_heads: Set[HeadKey]) -> None:
    """Build gradient masks and weight snapshots for all frozen heads.

    Must be called once after the frozen set changes (i.e. after each call to
    freeze_attention_heads_for_tasks). The per-step helpers
    freeze_apply_grad_mask and freeze_restore_weights then enforce freezing
    during training of the next task.

    Weight layout in timm ViT-B/16:
        attn.qkv.weight:  [3 * dim, dim]  — rows 0..dim-1 are Q,
                           dim..2*dim-1 are K, 2*dim..3*dim-1 are V.
                           Each head occupies head_dim consecutive rows per
                           Q/K/V block.
        attn.proj.weight: [dim, dim]       — each head occupies head_dim
                           consecutive INPUT columns (not output rows).

    Masks are stored as private attributes on the attention module so they are
    co-located with the weights they protect and survive checkpointing if the
    model state dict is saved via torch.save.

    Args:
        model:        timm VisionTransformer.
        frozen_heads: Complete set of (layer_idx, head_idx) to freeze,
                      including previously frozen heads.
    """
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn

        if not hasattr(attn, "qkv"):
            # Skip any non-standard block that lacks a fused qkv projection.
            continue

        num_heads  = attn.num_heads
        head_size  = attn.head_dim
        dim        = num_heads * head_size          # = attn.qkv.weight.shape[1]
        device     = attn.qkv.weight.device

        # Initialise masks to 1 (trainable); frozen slices will be set to 0.
        # Broadcasting shapes: qkv_mask [3*dim, 1] broadcasts over [3*dim, dim].
        #                       out_mask [1, dim]   broadcasts over [dim, dim].
        qkv_mask = torch.ones(dim * 3, 1, device=device)   # for qkv.weight
        out_mask = torch.ones(1, dim, device=device)        # for proj.weight

        for head_idx in range(num_heads):
            if (layer_idx, head_idx) not in frozen_heads:
                continue

            # Zero the Q, K, V row slices for this head in qkv_mask.
            for qkv_block in range(3):
                row_start = qkv_block * dim + head_idx * head_size
                row_end   = row_start + head_size
                qkv_mask[row_start:row_end, :] = 0.0

            # Zero the output projection column slice for this head.
            col_start = head_idx * head_size
            col_end   = col_start + head_size
            out_mask[:, col_start:col_end] = 0.0

        # Store masks on the module so trainer.py can read them per step.
        attn._frozen_qkv_mask = qkv_mask
        attn._frozen_out_mask = out_mask

        # Bias mask (same row layout as qkv_mask, but 1-D).
        has_qkv_bias = getattr(attn.qkv, "bias", None) is not None
        if has_qkv_bias:
            attn._frozen_qkv_bias_mask = qkv_mask.squeeze(1)   # [3*dim]

        # Snapshot the current weight values for frozen slices so that
        # freeze_restore_weights can undo weight-decay and optimizer-moment
        # drift after each update step.
        attn._frozen_qkv_weight_ref = attn.qkv.weight.detach().clone()
        attn._frozen_out_weight_ref = attn.proj.weight.detach().clone()
        if has_qkv_bias:
            attn._frozen_qkv_bias_ref = attn.qkv.bias.detach().clone()


def freeze_apply_grad_mask(model: nn.Module) -> None:
    """Zero the gradients of frozen head slices before the optimizer step.

    Must be called AFTER loss.backward() and BEFORE optimizer.step().
    Multiplies each gradient tensor element-wise with the corresponding mask
    (0 for frozen slices, 1 for trainable slices), so the optimizer sees a
    zero update for frozen parameters.

    No-op for blocks where build_head_freezing_mask has not been called
    (i.e. _frozen_qkv_mask is absent), which is the case before the first
    task completes.

    Args:
        model: timm VisionTransformer with frozen-head masks attached.
    """
    for block in model.blocks:
        attn = block.attn

        if not hasattr(attn, "_frozen_qkv_mask"):
            continue    # No heads frozen in this block yet.

        if attn.qkv.weight.grad is not None:
            attn.qkv.weight.grad.mul_(
                attn._frozen_qkv_mask.expand_as(attn.qkv.weight.grad)
            )

        if (hasattr(attn, "_frozen_qkv_bias_mask")
                and getattr(attn.qkv, "bias", None) is not None
                and attn.qkv.bias.grad is not None):
            attn.qkv.bias.grad.mul_(attn._frozen_qkv_bias_mask)

        if attn.proj.weight.grad is not None:
            attn.proj.weight.grad.mul_(
                attn._frozen_out_mask.expand_as(attn.proj.weight.grad)
            )


@torch.no_grad()
def freeze_restore_weights(model: nn.Module) -> None:
    """Restore frozen weight slices to their snapshot values after the optimizer step.

    Must be called AFTER optimizer.step(). Even though freeze_apply_grad_mask
    zeroes the gradient, optimisers like Adam maintain running moment estimates
    that can drift frozen weights via weight decay. This function hard-restores
    the frozen slices from the snapshot taken by build_head_freezing_mask,
    guaranteeing that frozen weights are bit-identical to their values at freeze
    time.

    No-op for blocks where build_head_freezing_mask has not been called.

    Args:
        model: timm VisionTransformer with frozen-head masks and snapshots attached.
    """
    for block in model.blocks:
        attn = block.attn

        if not hasattr(attn, "_frozen_qkv_mask"):
            continue    # No heads frozen in this block yet.

        # torch.where(condition, x, y): keep snapshot where frozen (mask==0),
        # keep updated weight where trainable (mask==1).
        qkv_frozen = (attn._frozen_qkv_mask == 0)
        attn.qkv.weight.data = torch.where(
            qkv_frozen, attn._frozen_qkv_weight_ref, attn.qkv.weight.data
        )

        out_frozen = (attn._frozen_out_mask == 0)
        attn.proj.weight.data = torch.where(
            out_frozen, attn._frozen_out_weight_ref, attn.proj.weight.data
        )

        if (hasattr(attn, "_frozen_qkv_bias_ref")
                and getattr(attn.qkv, "bias", None) is not None):
            bias_frozen = (attn._frozen_qkv_bias_mask == 0)
            attn.qkv.bias.data = torch.where(
                bias_frozen, attn._frozen_qkv_bias_ref, attn.qkv.bias.data
            )
