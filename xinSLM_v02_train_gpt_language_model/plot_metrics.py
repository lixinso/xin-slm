#!/usr/bin/env python3
"""
Script to visualize training metrics from the GPT language model training.
Reads metrics from training_metrics.json and creates training curves.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def load_metrics(metrics_file):
    """Load training metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_training_curves(metrics, save_path=None):
    """Create training curve plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss over Steps
    if metrics['train_losses'] and metrics['steps']:
        ax1.plot(metrics['steps'], metrics['train_losses'], 'b-', alpha=0.7, label='Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss vs Steps')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Validation Loss over Epochs
    if metrics['val_losses'] and metrics['epochs']:
        ax2.plot(metrics['epochs'], metrics['val_losses'], 'r-', marker='o', label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss vs Epochs')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Plot 3: Loss over Time
    if metrics['train_losses'] and metrics['timestamps']:
        ax3.plot(metrics['timestamps'], metrics['train_losses'], 'g-', alpha=0.7, label='Training Loss')
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss vs Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # Plot 4: Loss Distribution
    if metrics['train_losses']:
        ax4.hist(metrics['train_losses'], bins=20, alpha=0.7, color='blue', label='Training Loss')
        if metrics['val_losses']:
            ax4.hist(metrics['val_losses'], bins=10, alpha=0.7, color='red', label='Validation Loss')
        ax4.set_xlabel('Loss Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Loss Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("Plot saved to training_curves.png")
    
    plt.show()

def print_metrics_summary(metrics):
    """Print a summary of the training metrics."""
    print("\n=== Training Metrics Summary ===")
    
    if metrics['train_losses']:
        print(f"Training Loss:")
        print(f"  Initial: {metrics['train_losses'][0]:.4f}")
        print(f"  Final: {metrics['train_losses'][-1]:.4f}")
        print(f"  Min: {min(metrics['train_losses']):.4f}")
        print(f"  Max: {max(metrics['train_losses']):.4f}")
        print(f"  Mean: {np.mean(metrics['train_losses']):.4f}")
        print(f"  Std: {np.std(metrics['train_losses']):.4f}")
    
    if metrics['val_losses']:
        print(f"\nValidation Loss:")
        print(f"  Initial: {metrics['val_losses'][0]:.4f}")
        print(f"  Final: {metrics['val_losses'][-1]:.4f}")
        print(f"  Min: {min(metrics['val_losses']):.4f}")
        print(f"  Max: {max(metrics['val_losses']):.4f}")
        print(f"  Mean: {np.mean(metrics['val_losses']):.4f}")
        print(f"  Std: {np.std(metrics['val_losses']):.4f}")
    
    if metrics['timestamps']:
        print(f"\nTraining Time: {metrics['timestamps'][-1]:.2f} seconds")
    
    print(f"Total Training Steps: {len(metrics['train_losses'])}")
    print(f"Total Epochs: {len(metrics['val_losses'])}")

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from GPT language model')
    parser.add_argument('--metrics-file', '-m', default='training_metrics.json',
                       help='Path to metrics JSON file (default: training_metrics.json)')
    parser.add_argument('--output', '-o', default='training_curves.png',
                       help='Output plot filename (default: training_curves.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Only print summary, do not create plots')
    
    args = parser.parse_args()
    
    if not Path(args.metrics_file).exists():
        print(f"Error: Metrics file '{args.metrics_file}' not found!")
        print("Run the training script first to generate metrics.")
        return
    
    try:
        metrics = load_metrics(args.metrics_file)
        print_metrics_summary(metrics)
        
        if not args.no_plot:
            plot_training_curves(metrics, args.output)
    
    except Exception as e:
        print(f"Error processing metrics: {e}")

if __name__ == "__main__":
    main()