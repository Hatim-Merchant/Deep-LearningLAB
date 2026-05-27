from pathlib import Path
from datetime import datetime
import json


def create_experiment_dir(base_dir: Path = Path("experiments")):
    base_dir.mkdir(parents=True, exist_ok=True)

    folder_name = input("Name of experiment folder: ").strip()

    if folder_name == "":
        folder_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

    run_dir = base_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lightning_dir = run_dir / "lightning_logs"
    lightning_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "lightning_dir": lightning_dir,
        "config_path": run_dir / "config.json",
        "metrics_path": run_dir / "metrics.json",
        "model_path": run_dir / "model_final.pt",
        "last_checkpoint_path": checkpoint_dir / "last.ckpt",
    }


def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)