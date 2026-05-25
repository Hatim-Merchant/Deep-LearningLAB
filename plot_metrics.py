import os
import json
import matplotlib.pyplot as plt

def plot_vit_metrics(input_dir):
    """
    Plots and saves ViT training metrics from a given directory.
    
    Args:
        input_dir (str): The directory containing 'metrics.json'
    """
    # Define the path to the metrics file
    file_path = os.path.join(input_dir, 'metrics.json')
    
    # Check if the file exists before proceeding
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return

    # Load metrics from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract individual metric lists
    accuracies = data['accuracies']
    test_losses = data['test_losses']
    train_losses = data['train_losses']

    # Define the epoch range dynamically based on data length
    epochs = range(1, len(accuracies) + 1)

    # Initialize a side-by-side subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Accuracy Curve
    ax1.plot(epochs, accuracies, label='Test Accuracy', color='#2ca02c', linewidth=2)
    ax1.set_title(f'ViT Training ({input_dir}): Test Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    # Panel 2: Loss Curves (Train vs. Test)
    ax2.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    ax2.plot(epochs, test_losses, label='Test Loss', color='#d62728', linewidth=2)
    ax2.set_title(f'ViT Training ({input_dir}): Loss over Epochs', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    # Adjust layout to prevent truncation or overlap of labels
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = "plots_performance_vit"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the dynamic filename (e.g., 'vit_metrics_ViT_100_Epochs.png')
    output_filename = f"vit_metrics_{input_dir}.png"
    output_path = os.path.join(output_dir, output_filename)

    # Save the plot and close it to free memory
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot successfully generated and saved to '{output_path}'.")

# --- Example Usage ---
if __name__ == "__main__":
    # Plot for the 100 epochs run
    plot_vit_metrics('ViT_100_Epochs')
    
    # Plot for the 60 epochs run
    plot_vit_metrics('ViT_60_Epochs')