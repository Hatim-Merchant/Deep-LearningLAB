import os
os.environ['TIMM_FUSED_ATTN'] = '0' if 'TIMM_FUSED_ATTN' not in os.environ else os.environ['TIMM_FUSED_ATTN']
from time import time as ttime
import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
import tqdm
from typing import Literal
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from utils.vit_builder import VisionTransformer
from utils.mod_adam import ModAdam
from utils.dataset_builder import define_dataset
from utils import misc
from utils.gvm import GlobalVarsManager
from utils.trainer import train_one_epoch
from utils.head_freeze import calculate_head_importance, freeze_attention_heads_for_tasks, build_head_freezing_mask

torch.set_float32_matmul_precision("high")


def get_args():
    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    # Experiment options
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'sdomainet'), help='use lowercase')
    parser.add_argument('-dr', '--data_root', type=str, default="")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 10, 20, 25, 50, 100))
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--seed', type=int, default=2025)
    # Model options
    parser.add_argument('-m', '--model', type=str, default='vit_base_patch16_224.augreg_in21k', help='vit_base_patch16_224.augreg_in21k')
    parser.add_argument('--head_dim_type', type=str, choices=('task_classes', 'pretrained', 'text_dim'), default='task_classes')
    parser.add_argument('--logit_type', type=str, choices=('head_out', 'sim_imgtext'), default='head_out')
    parser.add_argument('--logit_scale', type=float, default=4.605170249938965, help='0 | 4.605170249938965')
    parser.add_argument('--logit_scale_trainable', type=misc.str2bool, default=False)
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)
    parser.add_argument('--pretrained_ignore_patterns', type=str, nargs="*", default=[])
    parser.add_argument('--refine_head', type=misc.str2bool, default=True)
    # Data augmentation options
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug', 'prototype', 'clip'), default='autoaug')
    parser.add_argument('--interp_mode', type=str, choices=('auto', 'bilinear', 'bicubic'), default='auto')
    parser.add_argument('--prob_cutmixup', type=float, default=0)
    parser.add_argument('--cutmixup_stopepoch', type=int, default=999)
    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)
    parser.add_argument('-et', '--expand_times', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=30)
    parser.add_argument('--use_amp', type=misc.str2bool, default=True)
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--consecutive_training', type=misc.str2bool, default=True, help="")
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    parser.add_argument('--training_string', type=str, nargs='+', default=('head', 'qkv.weight'))
    # Evaluation options
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)
    parser.add_argument('--only_last_metric', type=misc.str2bool, default=False)
    parser.add_argument('--use_ncm', type=misc.str2bool, default=False)
    # Optimizer options
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_scale', type=float, default=0.01)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+', default='qkv')
    parser.add_argument('--optimizer', type=str, default='mod_adam')
    # Head-freezing options (Michel-style)
    parser.add_argument('--head_freeze', action='store_true', help='freeze the most-important heads for CL')
    parser.add_argument('--freeze_ratio', type=float, default=0.1, help='fraction of ALL heads frozen per task (cumulative)')
    parser.add_argument('--freeze_subset', type=float, default=0.2, help='fraction (<=1) or count of task samples for importance')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5, 8])
    parser.add_argument('--decay_epochs', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    # Display options
    parser.add_argument('--show_bar', action='store_true')
    parser.add_argument('--print_model', action='store_true')
    # Logging options
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory for head-freeze JSON log. '
                             'Empty string = no JSON written.')

    args = parser.parse_args()

    if args.interp_mode == 'auto':
        match args.dataset:
            case 'imagenet_r' | 'sdomainet':
                args.interp_mode = 'bilinear'
            case 'cifar100':
                args.interp_mode = 'bicubic'
    if args.optimizer not in ('mod_adam', 'sgd', 'adam'):
        raise NotImplementedError(args.optimizer)

    # Validate method selection: the head-freeze baseline must run with plain Adam, not ModAdam, otherwise ARCL masking and head-freezing run together.
    #and we do not allow sgd for comparability reasons 
    # compare ARCL (modAdam) with head-freezing (adam)
    if args.head_freeze and args.optimizer != 'adam':
        raise ValueError(
            "--head_freeze is the Michel-style baseline and must use "
            "--optimizer adam. Combining it with mod_adam runs ARCL and "
            "head-freezing simultaneously; remove one of the two flags."
        )
    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    torchvision.set_image_backend('accimage')


def set_model_mode(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, to_gpu: bool = True, training_string: tuple[str] = None) -> VisionTransformer:
    for n, p in model.named_parameters():
        if training and any([_s in n for _s in training_string]):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    model.eval()
    for n, m in model.named_modules():
        if training and any([n.endswith(_s) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]

    if to_gpu:
        model.cuda()

    if training:
        for n in GVM.cache_dict['not_pretrained_params']:
            if n not in params_requires_grad:
                raise ValueError(f"'{n}' does not require grad but it is not pretrained.")
    else:
        assert len(params_requires_grad) == 0, f"{params_requires_grad}"
        assert len(modules_training) == 0, f"{modules_training}"

    return model


def set_learning_rates(GVM: GlobalVarsManager, model: VisionTransformer, base_lr: float, lr_scale: float, lr_scale_patterns: str) -> list[dict[str: Tensor | float]]:
    param_lr_groups = [{'params': [], 'lr': base_lr},
                       {'params': [], 'lr': base_lr * lr_scale}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for n, p in model.named_parameters():
        if p.requires_grad:
            _group_idx = 1 if any(_s in n for _s in lr_scale_patterns) else 0
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)

    return param_lr_groups


def _save_chekpoint(GVM: GlobalVarsManager, taskid: int, model: VisionTransformer):
    if taskid == 0:
        base_params = OrderedDict()
    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone()
        else:
            if taskid == 0:
                base_params[n] = p.clone()
            else:
                if not torch.all((_pb := GVM.param_dict[f'base_params'][n]) == p.to(_pb.device)):
                    print(f"WARNING:: save_chekpoint(): 'base_params' is changed!")

    if taskid == 0:
        assert not 'base_params' in GVM.param_dict
        GVM.param_dict['base_params'] = base_params

    assert not f'task_params_{taskid}' in GVM.param_dict
    GVM.param_dict[f'task_params_{taskid}'] = task_params


def _log_head_freeze_json(
    log_dir: str,
    taskid: int,
    freeze_ratio: float,
    freeze_subset: float,
    new_frozen: set,
    all_frozen: set,
    total_heads: int,
    task_importance_scores: dict,
) -> None:
    """Write head-freeze metrics for one task as a JSON entry to disk.

    File name : <log_dir>/head_freeze_scores.json
    Structure : A list of per-task entries accumulated over all tasks.
    The file is overwritten atomically (tmp -> rename) after each task so
    that the last fully-written state is preserved even on a crash.

    JSON entry fields
    -----------------
    task                  : 1-based task index (for readability).
    freeze_ratio          : --freeze_ratio hyperparameter.
    freeze_subset         : --freeze_subset hyperparameter.
    total_heads           : Total number of attention heads in the model
                            (12 layers × 12 heads = 144 for ViT-B/16).
    new_frozen_count      : Number of heads newly frozen after this task.
    all_frozen_count      : Cumulative number of frozen heads up to this task.
    frozen_pct            : Percentage of all heads that are frozen.
    new_frozen_heads      : Sorted list of newly frozen head keys ("LxHy").
    top10_heads_by_score  : Top-10 heads ranked by cumulative importance score.
    all_scores_by_task    : All 144 head scores for every task seen so far,
                            keyed as {"task_1": {"L0H0": score, ...}, ...}.
                            Scores are per-task (not cumulative) and
                            L2-normalised within each layer (as computed by
                            calculate_head_importance with normalize=True).
    all_scores_cumulative : Sum of per-task scores across all tasks so far,
                            keyed as {"L0H0": cumulative_score, ...}.
                            in descending order
                            This is the ranking signal used for freezing.

    Args:
        log_dir               : Target directory. Empty string = no writing.
        taskid                : 0-based task index.
        freeze_ratio          : --freeze_ratio hyperparameter.
        freeze_subset         : --freeze_subset hyperparameter.
        new_frozen            : Set of newly frozen (layer, head) pairs.
        all_frozen            : Cumulative set of all frozen heads.
        total_heads           : Total number of heads in the model.
        task_importance_scores: {task_id: {(layer, head): score}} for all tasks.
    """
    if not log_dir:
        return

    log_path = Path(log_dir) / "head_freeze_scores.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing entries if the file already exists.
    existing: list = []
    if log_path.exists():
        with open(log_path, "r") as f:
            existing = json.load(f)

    # --- Cumulative scores across all tasks seen so far ---
    # Maps "LxHy" -> sum of per-task importance scores.
    all_scores_flat: dict = {}
    for t_scores in task_importance_scores.values():
        for (layer_idx, head_idx), score in t_scores.items():
            key = f"L{layer_idx}H{head_idx}"
            all_scores_flat[key] = all_scores_flat.get(key, 0.0) + score

    # Top-10 heads by cumulative score for this task snapshot.
    top10 = sorted(all_scores_flat.items(), key=lambda x: x[1], reverse=True)[:10]

    # --- Per-task scores for ALL 144 heads ---
    # Outer key: "task_N" (1-based). Inner key: "LxHy". Value: score (float).
    # This allows downstream analysis of how individual head scores evolve
    # across tasks without having to re-run importance estimation.
    all_scores_by_task = {
        f"task_{t + 1}": {
            f"L{layer_idx}H{head_idx}": round(score, 6)
            for (layer_idx, head_idx), score in t_scores.items()
        }
        for t, t_scores in task_importance_scores.items()
    }

    # Build the entry for this task.
    entry = {
        "task":             taskid + 1,          # 1-based for readability
        "freeze_ratio":     freeze_ratio,
        "freeze_subset":    freeze_subset,
        "total_heads":      total_heads,
        "new_frozen_count": len(new_frozen),
        "all_frozen_count": len(all_frozen),
        "frozen_pct":       round(len(all_frozen) / total_heads * 100, 2),
        # Newly frozen heads as a sorted list of "LxHy" strings.
        "new_frozen_heads": sorted(f"L{l}H{h}" for l, h in new_frozen),
        # Top-10 heads by cumulative importance score.
        "top10_heads_by_score": [
            {"head": k, "cumulative_score": round(v, 6)} for k, v in top10
        ],
        # All 144 per-task scores (not cumulative), one sub-dict per task.
        "all_scores_by_task": all_scores_by_task,
        # Cumulative score of all 144 heads up to and including this task.
        # This is the exact ranking signal used by freeze_attention_heads_for_tasks.
        "all_scores_cumulative": {
            k: round(v, 6) 
            for k, v in sorted(all_scores_flat.items(), key=lambda x: x[1], reverse=True)

        },
    }

    existing.append(entry)

    # Atomic write: write to .tmp first, then rename —
    # no half-written JSON is left on disk if the process crashes mid-write.
    tmp_path = log_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(existing, f, indent=2)
    tmp_path.replace(log_path)


def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: VisionTransformer, **kwargs) -> VisionTransformer:
    args = GVM.args
    if args.epochs == 0:
        if args.use_ncm:
            extract_class_features(GVM, model)
        return model

    print(f"*" * 90 + " Start Training " + "*" * 90)
    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: VisionTransformer = set_model_mode(GVM, model, training=True, training_string=GVM.cache_dict['training_string'])
    model = modify_head(GVM, model, training=True, task_classes=task_classes)

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, expand_times=args.expand_times)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, timeout=args.timeout if args.workers > 0 else 0,
                            drop_last=args.prob_cutmixup > 0, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_patterns)

    if taskid == 0:
        GVM.cache_dict['old_avg_attn_map'] = torch.zeros(12, 196).cuda().detach()
        GVM.cache_dict['avg_attn_map'] = torch.zeros(12, 196).cuda().detach()
        GVM.cache_dict['temp_count'] = 0

    if args.optimizer == 'mod_adam':
        optimizer = ModAdam(param_groups, update_projection_dict={}, arg_dict={}, lr=args.lr, weight_decay=args.weight_decay, foreach=True)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs

    print(":: Training:")
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(GVM, epoch, dataloader, model, criterion, optimizer)
            print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr := str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)
    GVM.cache_dict['old_avg_attn_map'] = GVM.cache_dict['avg_attn_map'].clone().detach() / GVM.cache_dict['temp_count']

    _save_chekpoint(GVM, taskid, model)

    if args.refine_head or args.use_ncm:
        extract_class_features(GVM, model)
        if args.refine_head:
            refine_head(GVM, model)

    model.remove_text_features()

    print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: VisionTransformer) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, model, training=False)
    scalar_meter = misc.ScalarMeter(acc_task_inc="samp_avg:>6.2%", acc_class_inc="samp_avg:>6.2%")

    for images, target in tqdm.tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                logits: Tensor = model(images)

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        class_inc_preds = logits.argmax(dim=1)
        _task_inc_logits = logits.clone()
        _task_inc_logits[:, :eval_taskid * GVM.cl_mngr.num_classes_per_task] = -torch.inf
        _task_inc_logits[:, (eval_taskid + 1) * GVM.cl_mngr.num_classes_per_task:] = -torch.inf
        task_inc_preds = _task_inc_logits.argmax(dim=1)

        acc_task_inc, acc_topnn_task, num_nn_task = misc.calc_acc_topnn_dynamically(task_inc_preds, target)
        acc_class_inc, acc_topnn_class, num_nn_class = misc.calc_acc_topnn_dynamically(class_inc_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_task_inc=acc_task_inc, acc_class_inc=acc_class_inc)

    assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()

    print(f"Task [{train_taskid + 1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid + 1:>{len(_tt := str(train_taskid + 1))}}/{_tt}]: eval_time={ttime() - _ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)

    return result_dict


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: VisionTransformer):
    print(f"*" * 90 + f" Start Evaluation " + "*" * 90)

    model = modify_head(GVM, model, training=False)
    average_acc_meter = misc.ScalarMeter(acc_task_inc="samp_avg:>6.2%", acc_class_inc="samp_avg:>6.2%")

    for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
        eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid)
        one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model)
        GVM.acc_mat_dict[f'AccTaskIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_task_inc']
        GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
        average_acc_meter.add_step_value(**one_result_dict)
    model.remove_text_features()

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccTaskIncList'][train_taskid] = avg_result_dict['acc_task_inc']
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    print(f"{'=' * 90} End of task [{current_taskid + 1}/{GVM.cl_mngr.num_tasks}] {'=' * 90}")

    acc_info_dict = {
        'diag_task_avg_acc': float(np.diag(GVM.acc_mat_dict['AccTaskIncMat'])[:current_taskid + 1].mean()),
        'task_inc_last_acc': float(GVM.acc_mat_dict['AccTaskIncList'][current_taskid]),
        'task_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccTaskIncMat'], current_taskid),
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),
    }
    _formatter = misc.ScalarFormatter(sep=' | ', diag_task_avg_acc=">6.2%", task_inc_last_acc=">6.2%", class_inc_last_acc=">6.2%", task_inc_last_forg=">6.2%", class_inc_last_forg=">6.2%")

    print(f":: ** Results of task [{current_taskid + 1}]: [ {_formatter(**acc_info_dict)} ] **")
    print(f":: ** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")


def find_not_pretrained_params(model: VisionTransformer, pretrained: bool = True, pretrained_cfg: dict[str, str] = None, extra_pretrained_params: list[str] = []) -> list[str]:
    assert isinstance(extra_pretrained_params, (list, tuple))
    assert pretrained_cfg is not None

    if 'open_clip' in pretrained_cfg.get('hf_hub_filename', ''):
        _filename = timm.models._hub.HF_OPEN_CLIP_WEIGHTS_NAME
    else:
        _filename = timm.models._hub.HF_WEIGHTS_NAME
    pre_state_dict: OrderedDict[str, Tensor] = timm.models.load_state_dict_from_hf(pretrained_cfg['hf_hub_id'], _filename)

    if 'visual.class_embedding' in pre_state_dict.keys():
        pre_state_dict = timm.models.vision_transformer._convert_openai_clip(pre_state_dict, model)

    not_pretrained_params = []
    for n, p in model.named_parameters():
        if n not in pre_state_dict.keys() or not pretrained:
            not_pretrained_params.append(n)
        else:
            if p.shape != pre_state_dict[n].shape:
                not_pretrained_params.append(n)

    for n in deepcopy(not_pretrained_params):
        for _p in extra_pretrained_params:
            if _p in n:
                not_pretrained_params.remove(n)

    return not_pretrained_params


def get_param_id_dict(model: VisionTransformer, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any([_s in n for _s in patterns]):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    return param_id_dict


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type

    match args.logit_type:
        case 'sim_imgtext':
            assert head_dim_type in ('pretrained', 'text_dim')
        case 'head_out':
            assert head_dim_type in ('task_classes')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case 'text_dim':
            head_dim_arg_dict['num_classes'] = 512
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes

    if args.logit_type == 'head_out':
        if model.head.out_features != len(_target_classes):
            _mh = deepcopy(model.head)
            _mdevice = _mh.weight.device
            _mdtype = _mh.weight.dtype
            model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
            model.head.requires_grad_(_mh.weight.requires_grad)

            if training:
                assert model.head.weight.requires_grad
            else:
                assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
                _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.weight.data.shape == _hw.shape
                model.head.weight.data = _hw

                if _mh.bias is not None:
                    _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                    assert model.head.bias.data.shape == _hb.shape
                    model.head.bias.data = _hb
    else:
        raise ValueError(args.logit_type)

    return model


def extract_class_features(GVM: GlobalVarsManager, model: VisionTransformer) -> None:
    model = set_model_mode(GVM, model, training=False)

    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    feats = torch.empty([len(dataset), 768], dtype=torch.float32)
    label = torch.empty([len(dataset)], dtype=torch.long)

    smp_idx = 0
    for img, lbl in dataloader:
        with torch.no_grad():
            img: Tensor
            lbl: Tensor
            _feat = model.encode_image(img.cuda(non_blocking=True), pre_logits=True).cpu()
            for _f, _l in zip(_feat, lbl):
                feats[smp_idx] = _f
                label[smp_idx] = _l
                smp_idx += 1
    assert smp_idx == len(dataset)

    if GVM.args.refine_head:
        _mean_list = []
        _cov_list = []
        _class_list = []
        for _l in label.unique():
            _cls_feats = feats[label == _l]
            _mean_list.append(torch.mean(_cls_feats, dim=0, keepdim=False))
            _cov_list.append(torch.cov(torch.tensor(_cls_feats, dtype=torch.float64).T) + torch.eye(_cls_feats.shape[-1]) * 1e-4)
            _class_list.append(_l)
        _mean_list = torch.stack(_mean_list)
        _cov_list = torch.stack(_cov_list)
        _class_list = torch.stack(_class_list)

        _key = 'class_features'
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] = {'mean': _mean_list, 'cov': _cov_list, 'class': _class_list}
        else:
            GVM.cache_dict[_key]['mean'] = torch.cat([GVM.cache_dict[_key]['mean'], _mean_list])
            GVM.cache_dict[_key]['cov'] = torch.cat([GVM.cache_dict[_key]['cov'], _cov_list])
            GVM.cache_dict[_key]['class'] = torch.cat([GVM.cache_dict[_key]['class'], _class_list])
            assert len(GVM.cache_dict[_key]['mean']) == len(GVM.cache_dict[_key]['cov']) == len(GVM.cache_dict[_key]['class']) == len(GVM.cl_mngr.sofar_task_classes)

    if GVM.args.use_ncm:
        _proto_list = []
        _class_list = []
        for _l in label.unique():
            _proto_list.append(torch.mean(feats[label == _l], dim=0, keepdim=True))  # [1, 768]
            _class_list.append(_l)
        _proto_list = torch.cat(_proto_list)
        _class_list = torch.stack(_class_list)

        _key = 'prototypes'
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] = {'proto': _proto_list, 'class': _class_list}
        else:
            GVM.cache_dict[_key]['proto'] = torch.cat([GVM.cache_dict[_key]['proto'], _proto_list])
            GVM.cache_dict[_key]['class'] = torch.cat([GVM.cache_dict[_key]['class'], _class_list])
        assert len(GVM.cache_dict[_key]['proto']) == len(GVM.cache_dict[_key]['class']) == len(GVM.cl_mngr.sofar_task_classes)

    return None


def refine_head(GVM: GlobalVarsManager, model: VisionTransformer):
    feats_mean: Tensor = GVM.cache_dict['class_features']['mean']
    feats_cov: Tensor = GVM.cache_dict['class_features']['cov']
    feats_class: Tensor = GVM.cache_dict['class_features']['class']
    assert len(feats_class.unique()) == len(GVM.cl_mngr.sofar_task_classes)

    stat_dataset = TensorDataset(feats_mean, feats_cov, feats_class)

    model = modify_head(GVM, model, training=False)
    mhead = model.head

    mhead.train()
    mhead.cuda()
    mhead.requires_grad_()

    optimizer = create_optimizer_v2(mhead, opt='sgd', lr=0.001, weight_decay=1e-4, momentum=0.9)
    scheduler, num_epochs = create_scheduler_v2(optimizer, 'multistep', num_epochs=20, decay_milestones=[999,], decay_rate=0.1)
    criterion = nn.CrossEntropyLoss().cuda()
    from torch.distributions.multivariate_normal import MultivariateNormal

    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", acc_top1="samp_avg:>6.2%")
    print(":: Refine head:")
    for epoch in range(1, num_epochs + 1):
        scheduler.step(epoch)

        smp_inp = []
        smp_tgt = []
        assert len(stat_dataset) == len(GVM.cl_mngr.sofar_task_classes)
        _ns = 128
        for _cmean, _ccov, _cclass in stat_dataset:
            m = MultivariateNormal(_cmean.float(), _ccov.float())
            _smp = m.sample(sample_shape=(_ns,))
            smp_inp.append(_smp)
            smp_tgt.append(torch.as_tensor([_cclass,] * _ns, dtype=torch.long))
        smp_inp = torch.cat(smp_inp)
        smp_tgt = torch.cat(smp_tgt)

        train_data = TensorDataset(smp_inp, smp_tgt)
        assert len(train_data) == len(stat_dataset) * _ns
        dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

        for inp, tgt in dataloader:
            out: Tensor = mhead(inp.cuda(non_blocking=True))
            if model.logit_type == 'head_out':
                logits = out
            elif model.logit_type == 'sim_imgtext':
                logits = model.forward_logits(out)
            loss: Tensor = criterion(logits / GVM.args.temperature, tgt.cuda(non_blocking=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_top1, = misc.calc_accuracy(logits.cpu(), tgt.cpu(), topk=(1, ))
            scalar_meter.add_step_value(len(inp), loss=loss.item(), acc_top1=acc_top1)
        if (epoch % 10 == 0 or epoch == num_epochs):
            print(f"epoch [{epoch}/{num_epochs}]: {scalar_meter.format_outout(scalar_meter.update_epoch_average_value())}")


if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)

    GVM = GlobalVarsManager()
    GVM.init_from_args(args)

    GVM.cache_dict['exp_start_time'] = ttime()

    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#' * 90} Task: [{taskid + 1}/{GVM.cl_mngr.num_tasks}] {'#' * 90}")

        if not args.consecutive_training or taskid == 0:
            _other_args_dict = misc.get_specific_args_dict(args, 'logit_')
            _head_dim_arg_dict = get_head_dim_arg_dict(GVM, args)
            if args.logit_type == 'sim_imgtext':
                raise NotImplementedError()
            model: VisionTransformer = timm.create_model(args.model, pretrained=True, pretrained_strict=False, **_head_dim_arg_dict, other_args_dict=_other_args_dict)
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)

        _not_pretrained_params = find_not_pretrained_params(model, pretrained_cfg=model.pretrained_cfg)
        GVM.cache_dict['not_pretrained_params'] = _not_pretrained_params

        GVM.update_label_maps(taskid, current_task_classes)
        GVM.cache_dict['training_string'] = args.training_string
        misc.check_param_training(GVM.cache_dict['not_pretrained_params'], GVM.cache_dict['training_string'])
        GVM.training = True
        model = train_one_task(GVM, taskid, current_task_classes, model)
        GVM.training = False

        if args.only_last_metric:
            if taskid + 1 < GVM.cl_mngr.num_tasks:
                continue

        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)
