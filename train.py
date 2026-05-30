import torch
import pytorch_lightning as pl
import argparse

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.dataset import CIFAR10DataModule
from src.models.basic import ViT
from src.experiment_utils import create_experiment_dir, save_json
from src.metrics import MetricsCallback


BASE_DIR = Path(__file__).parent

LOG_EVERY_N_STEPS = 50

PATCH_SIZE = 4
SIZE = PATCH_SIZE * PATCH_SIZE * 3  # CIFAR10 RGB patches
HIDDEN_SIZE = 48
NUM_PATCHES = int(32 * 32 / PATCH_SIZE ** 2)

NUM_HEADS = 8
NUM_ENCODERS = 6

DROPOUT = 0.1
EMB_DROPOUT = 0.1

WEIGHT_DECAY = 1e-6

torch.set_float32_matmul_precision('medium')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default="vit_cifar10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    paths = create_experiment_dir(BASE_DIR / "experiments")

    RUN_DIR = paths["run_dir"]
    CHECKPOINT_DIR = paths["checkpoint_dir"]
    LIGHTNING_DIR = paths["lightning_dir"]
    CONFIG_PATH = paths["config_path"]
    MODEL_PATH = paths["model_path"]
    LAST_CHECKPOINT_PATH = paths["last_checkpoint_path"]
    METRICS_PATH = paths["metrics_path"]

    data = CIFAR10DataModule(
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        patch_size=PATCH_SIZE
    )

    config = {
        "exp_name": args.exp_name,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patch_size": PATCH_SIZE,
        "size": SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_patches": NUM_PATCHES,
        "num_heads": NUM_HEADS,
        "num_encoders": NUM_ENCODERS,
        "dropout": DROPOUT,
        "emb_dropout": EMB_DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "num_classes": data.classes,
        "selected_classes_to_train": data.selected_classes_to_train,
        "use_remapping": data.use_remapping,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    }

    save_json(config, CONFIG_PATH)

    model = ViT(
        size=SIZE,
        hidden_size=HIDDEN_SIZE,
        num_patches=NUM_PATCHES,
        num_classes=data.classes,
        num_heads=NUM_HEADS,
        num_encoders=NUM_ENCODERS,
        emb_dropout=EMB_DROPOUT,
        dropout=DROPOUT,
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        epochs=args.epochs
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_last=True,
        mode="min",
        save_top_k=3,
        verbose=True
    )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        MetricsCallback(METRICS_PATH),
    ]

    if not args.no_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=16
            )
        )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16" if torch.cuda.is_available() else 32,
        default_root_dir=LIGHTNING_DIR,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        max_epochs=args.epochs,
        callbacks=callbacks
    )

    ckpt_path = None
    if args.resume:
        ckpt_path = LAST_CHECKPOINT_PATH

    trainer.fit(model, data, ckpt_path=ckpt_path)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Final model saved to: {MODEL_PATH}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
