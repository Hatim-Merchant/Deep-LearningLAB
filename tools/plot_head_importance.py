import os
import glob
import torch
import matplotlib.pyplot as plt


def plot_one_heatmap(pt_path, output_dir):
    importance = torch.load(pt_path, map_location="cpu")

    if importance.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {importance.shape} from {pt_path}")

    filename = os.path.basename(pt_path)
    task_name = filename.replace("head_importance_", "").replace(".pt", "")

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(importance.numpy(), aspect="auto")
    plt.colorbar(label="Normalized gradient importance")

    plt.title(f"Head importance heatmap - {task_name}")
    plt.xlabel("Head index")
    plt.ylabel("Layer index")

    num_layers, num_heads = importance.shape
    plt.xticks(range(num_heads))
    plt.yticks(range(num_layers))

    # Add values inside cells
    for layer in range(num_layers):
        for head in range(num_heads):
            value = importance[layer, head].item()
            plt.text(head, layer, f"{value:.2f}", ha="center", va="center", fontsize=6)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{task_name}_heatmap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    input_pattern = "outputs/head_importance_task_*.pt"
    output_dir = "outputs/head_importance_heatmaps"

    pt_files = sorted(glob.glob(input_pattern))

    if len(pt_files) == 0:
        raise FileNotFoundError(f"No files found matching: {input_pattern}")

    print(f"Found {len(pt_files)} head importance files.")

    for pt_path in pt_files:
        plot_one_heatmap(pt_path, output_dir)


if __name__ == "__main__":
    main()