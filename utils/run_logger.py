"""Lightweight JSON run logger for training / evaluation metrics.

This module mirrors the scalars that are printed to the terminal into
machine-readable JSON files, so that experiments
can be compared later without re-parsing console output.

Files written (all inside ``log_dir``):
    config.json   -- the full argparse configuration of the run
                     (all hyperparameters, including the head-freeze settings
                     such as ``freeze_ratio`` / ``freeze_subset``).
    metrics.json  -- per-epoch training scalars, per-task evaluation results,
                     per-task end-of-task summaries, and the final accuracy
                     matrices for the whole run.

``head_freeze_scores.json`` is written separately by
``train_eval._log_head_freeze_json`` and is intentionally NOT duplicated here.

Gating:
    If ``log_dir`` is an empty string the logger becomes a no-op. This keeps
    the previous "empty --log_dir means no JSON is written" behaviour, and it
    is independent of ``--head_freeze`` so that both training methods produce
    comparable metric files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_python(value: Any) -> Any:
    """Convert numpy / torch objects into JSON-serialisable Python types.

    argparse values are already native Python, but the metric dictionaries
    produced by ``ScalarMeter`` and ``acc_mat_dict`` can contain numpy scalars,
    numpy arrays, or torch tensors (e.g. the accuracy matrices). ``json.dump``
    cannot serialise those directly.

    Args:
        value: Any value that may be a numpy scalar/array, a torch tensor, or
               an already-native Python type.

    Returns:
        A JSON-serialisable representation of ``value``.
    """
    # torch tensor -> nested Python list (or scalar for 0-dim tensors).
    # Checked via duck typing so torch does not have to be imported here.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    # numpy array -> nested list.
    if isinstance(value, np.ndarray):
        return value.tolist()
    # numpy scalar (e.g. np.float32) -> native Python scalar.
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON to ``path`` atomically.

    The data is written to a temporary sibling file and then renamed onto the
    target path. A reader therefore never observes a half-written file, and a
    crash mid-write leaves the previous good version intact.

    Args:
        path:    Destination JSON file path.
        payload: JSON-serialisable object to write.
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


class RunLogger:
    """Accumulates run metrics in memory and flushes them to JSON on disk.

    A single nested dictionary is kept in memory and ``metrics.json`` is
    rewritten after every update. For the scale of these experiments (tens of
    tasks, a few epochs each) the rewrite cost is negligible, and it keeps the
    on-disk file consistent after every step. This matches the atomic
    rewrite-per-task strategy already used for ``head_freeze_scores.json``.
    """

    def __init__(self, log_dir: str) -> None:
        """Initialise the logger.

        Args:
            log_dir: Output directory for the JSON files. An empty string
                     disables all disk writes (no-op logger).
        """
        # Empty log_dir disables all disk writes (no-op logger).
        self.enabled: bool = bool(log_dir)
        self.log_dir: Path | None = Path(log_dir) if self.enabled else None

        # In-memory metric store; mirrors the structure of metrics.json.
        self.metrics: dict[str, Any] = {
            "training": [],    # one entry per (task, epoch)
            "evaluation": [],  # one entry per (train_task, eval_task)
            "task_end": [],    # one entry per finished task
            "final": {},       # accuracy matrices / lists at the end of the run
        }

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # -- configuration --------------------------------------------------------

    def log_config(self, config: dict[str, Any]) -> None:
        """Write the run configuration (hyperparameters) to config.json.

        Args:
            config: Configuration dictionary, typically ``vars(args)``.
        """
        if not self.enabled:
            return
        serialisable = {k: _to_python(v) for k, v in config.items()}
        _atomic_write_json(self.log_dir / "config.json", serialisable)

    # -- per-step metric recording -------------------------------------------

    def log_training_epoch(self, taskid: int, epoch: int, scalars: dict[str, float]) -> None:
        """Record one epoch of training scalars (loss, accuracy, timing, lr).

        Args:
            taskid:  0-based task index (stored 1-based for readability).
            epoch:   1-based epoch index.
            scalars: Epoch-averaged scalar dict from ``ScalarMeter``.
        """
        if not self.enabled:
            return
        entry: dict[str, Any] = {"task": taskid + 1, "epoch": epoch}
        entry.update({k: _to_python(v) for k, v in scalars.items()})
        self.metrics["training"].append(entry)
        self._flush()

    def log_evaluation(self, train_taskid: int, eval_taskid: int, result: dict[str, float]) -> None:
        """Record the evaluation of one past task after training ``train_taskid``.

        Args:
            train_taskid: 0-based index of the task just trained.
            eval_taskid:  0-based index of the task being evaluated.
            result:       Evaluation result dict (accuracies, sample count).
        """
        if not self.enabled:
            return
        entry: dict[str, Any] = {"train_task": train_taskid + 1, "eval_task": eval_taskid + 1}
        entry.update({k: _to_python(v) for k, v in result.items()})
        self.metrics["evaluation"].append(entry)
        self._flush()

    def log_task_end(self, taskid: int, acc_info: dict[str, float]) -> None:
        """Record the end-of-task summary (avg / last accuracy, forgetting).

        Args:
            taskid:   0-based task index (stored 1-based for readability).
            acc_info: End-of-task metric dict.
        """
        if not self.enabled:
            return
        entry: dict[str, Any] = {"task": taskid + 1}
        entry.update({k: _to_python(v) for k, v in acc_info.items()})
        self.metrics["task_end"].append(entry)
        self._flush()

    def log_final(self, final: dict[str, Any]) -> None:
        """Record final run-level results (e.g. accuracy matrices / lists).

        Args:
            final: Mapping of result name to numpy array / list / scalar.
        """
        if not self.enabled:
            return
        self.metrics["final"] = {k: _to_python(v) for k, v in final.items()}
        self._flush()

    # -- internal -------------------------------------------------------------

    def _flush(self) -> None:
        """Persist the current in-memory metrics to metrics.json atomically."""
        _atomic_write_json(self.log_dir / "metrics.json", self.metrics)
