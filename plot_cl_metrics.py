"""
Plotting utilities for Continual Learning experiments.

Generates visualizations showing:
1. Catastrophic forgetting: Performance degradation on old classes
2. Learning progress: Accuracy improvement on new classes
3. Overall performance trajectory
4. Per-class accuracy evolution
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# CIFAR-10 class names
CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Color scheme
COLOR_OLD_CLASSES = '#e74c3c'  # Red for old classes (showing forgetting)
COLOR_NEW_CLASSES = '#2ecc71'  # Green for new classes (showing learning)
COLOR_OVERALL = '#3498db'      # Blue for overall performance
COLOR_FORGETTING = '#e67e22'   # Orange for forgetting curve


def load_cl_metrics(metrics_path):
    """Load continual learning metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


def plot_forgetting_curves(metrics, output_path=None, title=None):
    """
    Create a comprehensive visualization of the continual learning experiment.
    
    Shows:
    - Top left: Accuracy on old vs new classes over time
    - Top right: Forgetting curve (baseline - current old class accuracy)
    - Bottom left: Overall accuracy
    - Bottom right: Per-class accuracies heatmap
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    epochs = list(range(1, len(metrics['old_class_accuracy']) + 1))
    baseline_old = metrics['baseline']['old_class_accuracy']
    
    # --- Plot 1: Old vs New Class Accuracy ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot([0] + epochs, [baseline_old] + metrics['old_class_accuracy'], 
             'o-', color=COLOR_OLD_CLASSES, linewidth=2, markersize=6,
             label='Old Classes (0-4: plane, car, bird, cat, deer)')
    ax1.plot([0] + epochs, [0.0] + metrics['new_class_accuracy'], 
             's-', color=COLOR_NEW_CLASSES, linewidth=2, markersize=6,
             label='New Classes (5-9: dog, frog, horse, ship, truck)')
    ax1.axhline(y=baseline_old, color=COLOR_OLD_CLASSES, linestyle='--', 
                alpha=0.5, label=f'Old Class Baseline ({baseline_old:.3f})')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy on Old vs New Classes', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(epochs))
    
    # --- Plot 2: Forgetting Curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    forgetting = [baseline_old - acc for acc in metrics['old_class_accuracy']]
    ax2.fill_between(epochs, forgetting, alpha=0.3, color=COLOR_FORGETTING)
    ax2.plot(epochs, forgetting, 'o-', color=COLOR_FORGETTING, 
             linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Forgetting (Baseline - Current)', fontsize=12)
    ax2.set_title('Catastrophic Forgetting Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    
    # Add annotation for final forgetting
    final_forgetting = forgetting[-1]
    ax2.annotate(f'Final Forgetting: {final_forgetting:.3f}',
                xy=(len(epochs), final_forgetting),
                xytext=(len(epochs) * 0.6, max(forgetting) * 0.8),
                arrowprops=dict(arrowstyle='->', color=COLOR_FORGETTING),
                fontsize=11, fontweight='bold', color=COLOR_FORGETTING)
    
    # --- Plot 3: Overall Accuracy ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, metrics['overall_accuracy'], 
             'o-', color=COLOR_OVERALL, linewidth=2, markersize=6)
    
    # Add target baseline (if model had no forgetting)
    theoretical_max = (baseline_old * 5 + 1.0 * 5) / 10  # Assumes perfect new class learning
    ax3.axhline(y=theoretical_max, color='gray', linestyle='--', 
                alpha=0.5, label=f'Theoretical Max ({theoretical_max:.3f})')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Overall Accuracy (All 10 Classes)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, len(epochs))
    
    # --- Plot 4: Per-Class Accuracy Heatmap ---
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Prepare data for heatmap
    per_class = metrics['per_class_accuracies']
    num_classes = len(per_class)
    num_epochs = len(epochs)
    
    # Create accuracy matrix (classes x epochs)
    accuracy_matrix = np.zeros((num_classes, num_epochs))
    for cls_idx in sorted([int(k) for k in per_class.keys()]):
        cls_str = str(cls_idx)
        if cls_str in per_class:
            accuracy_matrix[cls_idx, :] = per_class[cls_str]
    
    # Plot heatmap
    im = ax4.imshow(accuracy_matrix, aspect='auto', cmap='RdYlGn', 
                    vmin=0, vmax=1, interpolation='nearest')
    
    # Color old vs new classes differently on y-axis
    old_class_indices = [0, 1, 2, 3, 4]
    new_class_indices = [5, 6, 7, 8, 9]
    
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([f'{i}: {CIFAR10_CLASSES[i]}' for i in range(10)], fontsize=9)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_title('Per-Class Accuracy Evolution', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, label='Accuracy')
    cbar.set_label('Accuracy', fontsize=10)
    
    # Add annotations for old vs new
    ax4.axhline(y=4.5, color='white', linewidth=2)
    ax4.text(num_epochs * 1.15, 2, 'OLD\nCLASSES', fontsize=9, 
            ha='center', va='center', color=COLOR_OLD_CLASSES, fontweight='bold')
    ax4.text(num_epochs * 1.15, 7, 'NEW\nCLASSES', fontsize=9, 
            ha='center', va='center', color=COLOR_NEW_CLASSES, fontweight='bold')
    
    # Main title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary text box
    final_old = metrics['old_class_accuracy'][-1]
    final_new = metrics['new_class_accuracy'][-1]
    final_overall = metrics['overall_accuracy'][-1]
    
    summary_text = (
        f"Final Results:\n"
        f"  Old Classes: {final_old:.3f} (↓{baseline_old - final_old:.3f})\n"
        f"  New Classes: {final_new:.3f}\n"
        f"  Overall: {final_overall:.3f}\n"
        f"  Forgetting: {baseline_old - final_old:.3f}"
    )
    
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig


def plot_comparison(metrics_list, labels, output_path=None):
    """
    Compare multiple continual learning experiments.
    
    Args:
        metrics_list: List of metrics dictionaries
        labels: List of experiment labels
        output_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels)):
        epochs = list(range(1, len(metrics['old_class_accuracy']) + 1))
        baseline_old = metrics['baseline']['old_class_accuracy']
        
        # Forgetting on old classes
        axes[0, 0].plot([0] + epochs, [baseline_old] + metrics['old_class_accuracy'],
                       'o-', color=colors[idx], linewidth=2, markersize=5, label=label)
        
        # Learning on new classes
        axes[0, 1].plot([0] + epochs, [0.0] + metrics['new_class_accuracy'],
                       's-', color=colors[idx], linewidth=2, markersize=5, label=label)
        
        # Forgetting curve
        forgetting = [baseline_old - acc for acc in metrics['old_class_accuracy']]
        axes[1, 0].plot(epochs, forgetting, 'o-', color=colors[idx], 
                       linewidth=2, markersize=5, label=label)
        
        # Overall accuracy
        axes[1, 1].plot(epochs, metrics['overall_accuracy'], 'o-',
                       color=colors[idx], linewidth=2, markersize=5, label=label)
    
    # Configure subplots
    axes[0, 0].set_title('Old Class Accuracy (Forgetting)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('New Class Accuracy (Learning)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Forgetting Magnitude', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Forgetting (Baseline - Current)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Overall Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    fig.suptitle('Continual Learning Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    return fig


def generate_summary_table(metrics_list, labels):
    """Generate a text summary table of results."""
    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Old Acc':<12} {'New Acc':<12} {'Overall':<12} {'Forgetting':<12}")
    print("-" * 80)
    
    for metrics, label in zip(metrics_list, labels):
        baseline_old = metrics['baseline']['old_class_accuracy']
        final_old = metrics['old_class_accuracy'][-1]
        final_new = metrics['new_class_accuracy'][-1]
        final_overall = metrics['overall_accuracy'][-1]
        forgetting = baseline_old - final_old
        
        print(f"{label:<25} {final_old:<12.4f} {final_new:<12.4f} "
              f"{final_overall:<12.4f} {forgetting:<12.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Plot Continual Learning metrics'
    )
    parser.add_argument('--metrics', type=str, required=True, nargs='+',
                        help='Path(s) to cl_metrics.json file(s)')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Labels for each experiment (for comparison)')
    parser.add_argument('--output-dir', type=str, default='CL/plots',
                        help='Directory to save plots')
    parser.add_argument('--compare', action='store_true',
                        help='Create comparison plot for multiple experiments')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all metrics
    metrics_list = []
    for path in args.metrics:
        if os.path.exists(path):
            metrics_list.append(load_cl_metrics(path))
        else:
            print(f"Warning: Metrics file not found: {path}")
    
    if len(metrics_list) == 0:
        print("No valid metrics files found!")
        return
    
    # Generate labels if not provided
    if args.labels is None:
        args.labels = [f"Experiment {i+1}" for i in range(len(metrics_list))]
    elif len(args.labels) != len(metrics_list):
        print("Warning: Number of labels doesn't match number of metrics files")
        args.labels = args.labels[:len(metrics_list)]
    
    if args.compare and len(metrics_list) > 1:
        # Generate comparison plot
        compare_path = os.path.join(args.output_dir, 'cl_comparison.png')
        plot_comparison(metrics_list, args.labels, output_path=compare_path)
        generate_summary_table(metrics_list, args.labels)
    
    # Generate individual plots for each experiment
    for metrics, label in zip(metrics_list, args.labels):
        safe_label = label.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(args.output_dir, f'cl_forgetting_{safe_label}.png')
        plot_forgetting_curves(metrics, output_path=output_path, title=f"Catastrophic Forgetting - {label}")
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
