import json
from pathlib import Path
import pytorch_lightning as pl


class MetricsCallback(pl.Callback):
    def __init__(self, metrics_path):
        super().__init__()
        self.metrics_path = Path(metrics_path)

        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "val_accuracies": [],
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        logged_metrics = trainer.callback_metrics

        train_loss = logged_metrics.get("train_loss")
        val_loss = logged_metrics.get("val_loss")
        val_accuracy = logged_metrics.get("val_accuracy")

        if train_loss is not None:
            self.metrics["train_losses"].append(float(train_loss.detach().cpu()))

        if val_loss is not None:
            self.metrics["val_losses"].append(float(val_loss.detach().cpu()))

        if val_accuracy is not None:
            self.metrics["val_accuracies"].append(float(val_accuracy.detach().cpu()))

        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)