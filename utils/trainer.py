from time import time as ttime
import tqdm
import numpy as np
import torch
from torch import nn, Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from utils import misc
from utils.gvm import GlobalVarsManager
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from utils.head_freeze import freeze_apply_grad_mask, freeze_restore_weights  # head-freeze method


def train_one_epoch(GVM: GlobalVarsManager, curr_epoch: int, dataloader: DataLoader, model, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer) -> str:
    """Train the model for one epoch.

    Supports two operating modes, selected via args.optimizer:
      - 'mod_adam': ARCL method (Lu et al., AAAI 2026). Runs attention-region
                    gradient masking and accumulates rollout attention maps for
                    the next task's mask generation. All ARCL logic is preserved
                    exactly; this branch is untouched.
      - 'adam'    : Head-freezing method (this work, Michel et al. inspired).
                    Plain Adam step; ARCL masking is NOT executed so that the
                    two methods can be compared fairly on the same ViT backbone.

    The two branches are mutually exclusive: args.head_freeze hooks
    (freeze_apply_grad_mask / freeze_restore_weights) run in both modes
    but have no effect unless build_head_freezing_mask was called beforehand.

    Args:
        GVM:        GlobalVarsManager carrying args, cl_mngr, and cache_dict.
        curr_epoch: Current epoch index (1-based); used to trigger end-of-epoch
                    attention-mask accumulation on the final epoch.
        dataloader: Training DataLoader for the current task.
        model:      timm ViT-B/16 with trainable qkv/proj + classifier.
        criterion:  Cross-entropy loss.
        optimizer:  Adam or ModAdam optimizer instance.

    Returns:
        Formatted scalar string (loss, accuracy, timing) for console logging.
    """
    args = GVM.args
    temperature: float = args.temperature
    use_amp: bool = args.use_amp
    assert temperature > 0.

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(lr="step_last:.3e", ce_loss="samp_avg:.4f", loss="samp_avg:.4f",
                                    data_time="step_sum:.3f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%", acc_top5="samp_avg:>6.2%")
    _btimer = ttime()

    if args.optimizer == 'mod_adam':
        data_for_all_class = dict()
        for i in GVM.cl_mngr.current_task_classes:
            data_for_all_class[i] = torch.tensor([], device=model.device)
        all_rollout_attentions = dict()
        for i in GVM.cl_mngr.current_task_classes:
            all_rollout_attentions[i] = torch.zeros((0, 12, 196), device=model.device)

    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        data_time = ttime() - _btimer

        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        mix_img = images
        mix_lbl = target

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits: Tensor = model(mix_img)

        if i_batch == 1:
            if args.seperate_head:
                assert logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
            else:
                assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        ce_loss = criterion(logits / temperature, mix_lbl)

        loss: Tensor = ce_loss

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()

        # =====================================================================
        # ARCL: attention-region gradient masking (mod_adam only).
        # Completely untouched; only gated out under --optimizer adam
        # so that the head-freeze baseline is not contaminated by this masking.
        # =====================================================================
        if args.optimizer == 'mod_adam':
            if GVM.cl_mngr.current_taskid > 0:
                attn_maps_ = GVM.cache_dict['old_avg_attn_map'].detach()

                for ii, layer in enumerate(model.blocks):
                    attn_maps = attn_maps_[ii]
                    attn_maps = torch.cat([torch.tensor([0.0], device=attn_maps.device), attn_maps])

                    attn = layer.attn.attn.detach()
                    grad = layer.attn.attn_no_softmax.grad.detach()

                    for i in range(len(images)):
                        attn[i, :, :, :] = attn[i, :, :, :] * attn_maps.view(1, 197, 1)
                        grad[i, :, :, :] = grad[i, :, :, :] * attn_maps.view(1, 197, 1)

            update_factor_dict = dict()
            i = 0
            for layer in model.blocks:
                if layer.attn.qkv.weight.grad is not None:
                    batch_size = len(images)
                    w_qkv_grad = layer.attn.qkv.weight.grad.clone().detach()
                    old_w_qkv_grad = layer.attn.qkv.weight.grad.detach()
                    w_q_grad = w_qkv_grad[0:768, :]
                    w_k_grad = w_qkv_grad[768:768 * 2, :]
                    w_v_grad = w_qkv_grad[768 * 2:768 * 3, :]
                    out_grad = layer.attn.out.grad.detach()

                    q_grad = layer.attn.attn_no_softmax.grad @ layer.attn.k
                    w_q_grad = q_grad.permute(0, 2, 1, 3).reshape(batch_size * 197, 12 * 64).T @ layer.attn.input.reshape(batch_size * 197, 12 * 64) * layer.attn.scale
                    k_grad = layer.attn.attn_no_softmax.grad.transpose(-2, -1) @ layer.attn.q_scaled
                    w_k_grad = k_grad.permute(0, 2, 1, 3).reshape(batch_size * 197, 12 * 64).T @ layer.attn.input.reshape(batch_size * 197, 12 * 64)
                    v_grad = (out_grad.transpose(-2, -1) @ layer.attn.attn.type(torch.float16)).transpose(-2, -1)
                    w_v_grad = v_grad.permute(0, 2, 1, 3).reshape(batch_size * 197, 12 * 64).T @ layer.attn.input.reshape(batch_size * 197, 12 * 64)

                    w_qkv_grad[0:768, :] = w_q_grad
                    w_qkv_grad[768:768 * 2, :] = w_k_grad
                    w_qkv_grad[768 * 2:768 * 3, :] = w_v_grad
                    update_factor_mat = w_qkv_grad / old_w_qkv_grad
                    update_factor_mat = torch.where(torch.isnan(update_factor_mat), torch.ones_like(update_factor_mat), update_factor_mat)
                    update_factor_mat = update_factor_mat.clamp(-10, 10).detach()
                    update_factor_dict[i] = update_factor_mat
                    i += 1
                else:
                    update_factor_mat = torch.ones_like(update_factor_dict[i - 1])
                    update_factor_dict[i] = update_factor_mat
                    i += 1
               # =====================================================================

        # Head-freeze method: zero the gradients of frozen head slices before
        # the optimizer step. No-op if build_head_freezing_mask was never called
        # (i.e. when --head_freeze is not set).
        if args.head_freeze: 
            freeze_apply_grad_mask(model)
        if args.optimizer == 'mod_adam':
            amp_scalar.step(optimizer, update_factor_dict=update_factor_dict)
            amp_scalar.update()
            del update_factor_dict
        else:
            amp_scalar.step(optimizer)
            amp_scalar.update()

        # Head-freeze method: restore frozen weight slices to their snapshot
        # values, undoing any drift introduced by weight decay or Adam moments.
        if args.head_freeze:
            freeze_restore_weights(model)
        if args.optimizer == 'mod_adam':
            def get_attn(model, i) -> Tensor:
                attention_map = torch.stack([model.blocks[j].attn.attn_[i].detach() for j in range(12)])
                attention_map = attention_map.mean(dim=1)
                residual_att = torch.eye(attention_map.size(1), device=attention_map.device)
                aug_att_mat = attention_map + residual_att
                aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
                rollout_attentions = torch.zeros(aug_att_mat.size(), device=aug_att_mat.device)
                rollout_attentions[0] = aug_att_mat[0]
                for n in range(1, aug_att_mat.size(0)):
                    rollout_attentions[n] = torch.matmul(aug_att_mat[n], rollout_attentions[n - 1])
                return rollout_attentions[:, 0, 1:].detach()

            for i, label in enumerate(target):
                data = get_attn(model, i)
                all_rollout_attentions[GVM.cl_mngr.current_task_classes[label]] = torch.cat((all_rollout_attentions[GVM.cl_mngr.current_task_classes[label]], data.reshape(1, 12, 196)), dim=0)
                data_for_all_class[GVM.cl_mngr.current_task_classes[label]] = torch.cat((data_for_all_class[GVM.cl_mngr.current_task_classes[label]], data.flatten()))

        acc_top1, acc_top5 = misc.calc_accuracy(logits, target, topk=(1, 2))
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), lr=optimizer.param_groups[0]['lr'], ce_loss=ce_loss.item(),
                                    loss=loss.item(), data_time=data_time, batch_time=batch_time, acc_top1=acc_top1, acc_top5=acc_top5)
        _btimer = ttime()

    if args.optimizer == 'mod_adam' and curr_epoch == GVM.args.epochs:
        for classid in data_for_all_class.keys():
            data = data_for_all_class[classid].cpu().numpy()
            data_1 = sliding_window_view(data.reshape(data.shape[0]), 196 * 12)
            image_1 = all_rollout_attentions[classid]
            for i in range(int(data.shape[0] / (196 * 12))):
                data_ = data_1[i].reshape(12, 196)
                image_ = image_1[i]
                split_point_value = [find_split_point(data_[j]) for j in range(data_.shape[0])]

                cat = []
                for j in range(data_.shape[0]):
                    t = data_[j]
                    h = image_[j]

                    temp = torch.where(h <= split_point_value[j],
                                       torch.tensor(1.0, device=model.device),
                                       torch.tensor(0.0, device=model.device))
                    cat.append(temp)

                temp = torch.stack(cat, dim=0)
                GVM.cache_dict['avg_attn_map'] += temp
                GVM.cache_dict['temp_count'] += 1

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())

    return _epoch_scalar_str


def find_split_point(data: np.ndarray, eps=0.001):
    flipped_x = np.sort(data)
    flipped_y = np.arange(len(data))
    num_bins = 10000
    regular_x = np.linspace(flipped_x.min(), flipped_x.max(), num_bins)
    regular_y = np.interp(regular_x, flipped_x, flipped_y)
    flipped_x = regular_x
    flipped_y = regular_y

    flipped_y = gaussian_filter(flipped_y, sigma=20)
    first_derivative_flipped = np.gradient(flipped_y, flipped_x)
    second_derivative_flipped = np.gradient(first_derivative_flipped, flipped_x)

    min_idx = np.argmin(second_derivative_flipped)
    min_x = min(max(flipped_x[min_idx], flipped_x[0]) - eps, flipped_x[-1])
    min_y = flipped_y[min_idx]

    return min_x
