import json
from pathlib import Path
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_metrics(exp_name):
    exp_dir = EXPERIMENTS_DIR / exp_name

    metrics_path = exp_dir / "metrics.json"
    config_path = exp_dir / "config.json"

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder does not exist: {exp_dir}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json was not found: {metrics_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"config.json was not found: {config_path}")

    metrics = load_json(metrics_path)
    config = load_json(config_path)

    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    val_accuracies = metrics["val_accuracies"]

    epochs_config = config["epochs"]

    num_train_loss = len(train_losses)
    num_val_loss = len(val_losses)
    num_val_acc = len(val_accuracies)

    epochs_loss = list(range(1, min(num_train_loss, num_val_loss) + 1))
    epochs_acc = list(range(1, num_val_acc + 1))

    train_losses_plot = train_losses[:len(epochs_loss)]
    val_losses_plot = val_losses[:len(epochs_loss)]
    val_accuracies_plot = val_accuracies[:len(epochs_acc)]

    #exp_title = config.get("exp_name", exp_name)
    batch_size = config.get("batch_size")
    lr = config.get("lr")
    accelerator = config.get("accelerator")
    #num_classes = config.get("num_classes")
    selected_classes = config.get("selected_classes_to_train")

    subtitle = (
        f"epochs={epochs_config}, "
        #f"train_loss_values={num_train_loss}, "
        #f"val_loss_values={num_val_loss}, "
        #f"val_accuracy_values={num_val_acc}, "
        f"batch_size={batch_size}, lr={lr}, "
        f"classes={selected_classes}, accelerator={accelerator}"
    )
    # Combined Plot: left = Accuracy, right = Loss
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Accuracy
    axes[0].plot(epochs_acc, val_accuracies_plot, label="Validation Accuracy", color="green")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("ViT Training: Validation Accuracy over Epochs")
    axes[0].legend()
    axes[0].grid(True)

    # Right: Loss
    axes[1].plot(epochs_loss, train_losses_plot, label="Train Loss")
    axes[1].plot(epochs_loss, val_losses_plot, label="Validation Loss", color="red")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("ViT Training: Loss over Epochs")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"ViT Training Overview\n{subtitle}")
    plt.tight_layout()
    plt.savefig(exp_dir / "training_overview.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    exp_name = input("Which folder in experiments do you want to plot? ").strip()
    plot_metrics(exp_name)