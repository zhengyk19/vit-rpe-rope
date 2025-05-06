#!/usr/bin/env python
"""
Comparison script for shared vs non-shared heads in PolynomialRPE.
This script runs training jobs comparing both parameter sharing modes
for different polynomial degrees and logs the results.
"""

import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import numpy as np

def get_args():
    """Parse command-line arguments for the comparison configuration."""
    parser = argparse.ArgumentParser(description='Compare Shared vs Non-shared Heads in PolynomialRPE')
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    # Polynomial degrees to test
    parser.add_argument('--degrees', type=str, default='1,3,5', 
                       help='Comma-separated list of polynomial degrees to test')
    # Training configuration
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs for each run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    # Output directories
    parser.add_argument('--results_dir', type=str, default='results_shared_comparison',
                       help='Directory to save comparison results')
    return parser.parse_args()

def run_training(degree, shared_heads, args):
    """Run a training job with specified polynomial degree and sharing mode."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"{args.results_dir}/logs"
    ckpt_dir = f"{args.results_dir}/checkpoints"
    
    sharing_mode = "shared" if shared_heads else "nonshared"
    
    # Create the command for training with specific configuration
    cmd = [
        "python", "train.py",
        "--dataset", args.dataset,
        "--pos_encoding", "polynomial",
        "--poly_degree", str(degree),
        "--poly_shared_heads" if shared_heads else "--no-poly_shared_heads",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--log_dir", log_dir,
        "--ckpt_dir", ckpt_dir
    ]
    
    # Run the training process
    print(f"\n{'='*80}")
    print(f"Starting training with polynomial degree = {degree}, sharing mode = {sharing_mode}")
    print(f"{'='*80}\n")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    # Stream and collect the output
    output = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")  # Stream to console
        output.append(line)
        if not line:
            break
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Error: Training process exited with code {return_code}")
        for line in process.stderr:
            print(line, end="")
    
    # Find the CSV log file
    log_files = [f for f in os.listdir(log_dir) if f.endswith(f'{timestamp}.csv')]
    if log_files:
        log_path = os.path.join(log_dir, log_files[0])
        return log_path, degree, shared_heads
    else:
        # Find the most recent log file with 'polynomial' in the name
        log_files = [f for f in os.listdir(log_dir) if 'polynomial' in f]
        if log_files:
            # Sort by modification time
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            log_path = os.path.join(log_dir, log_files[0])
            return log_path, degree, shared_heads
    
    return None, degree, shared_heads

def main():
    """Run comparison between shared and non-shared heads for different polynomial degrees."""
    args = get_args()
    
    # Parse the polynomial degrees to test
    degrees = [int(d.strip()) for d in args.degrees.split(',')]
    
    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.results_dir}/plots", exist_ok=True)
    
    # Prepare for storing results
    results = []
    
    # Run training for each configuration
    for degree in degrees:
        for shared_heads in [True, False]:
            log_path, degree, shared_heads = run_training(degree, shared_heads, args)
            results.append((degree, shared_heads, log_path))
    
    # Analyze and visualize results
    analyze_results(results, degrees, args)

def analyze_results(results, degrees, args):
    """Analyze and visualize the comparison results."""
    # Set up figure with shared x-axis but multiple y-axis subplots
    fig, axes = plt.subplots(len(degrees), 1, figsize=(10, 5*len(degrees)), sharex=True)
    if len(degrees) == 1:
        axes = [axes]  # Make sure axes is a list for easier indexing
    
    # Dictionary to store results by degree and sharing mode
    results_dict = {}
    
    # Process each result
    for degree, shared_heads, log_path in results:
        if log_path and os.path.exists(log_path):
            df = pd.read_csv(log_path)
            
            # Store the results
            key = (degree, shared_heads)
            results_dict[key] = {
                'final_acc': df['test_acc'].iloc[-1],
                'best_acc': df['best_acc'].max(),
                'df': df
            }
            
            # Plot on the appropriate subplot
            degree_idx = degrees.index(degree)
            sharing_label = "Shared" if shared_heads else "Non-shared"
            style = '-o' if shared_heads else '--s'
            axes[degree_idx].plot(df['epoch'], df['test_acc'], style, label=f'{sharing_label} Heads')
            
            # Format the subplot
            axes[degree_idx].set_title(f'Polynomial Degree = {degree}')
            axes[degree_idx].set_ylabel('Test Accuracy (%)')
            axes[degree_idx].grid(True, linestyle='--', alpha=0.7)
            axes[degree_idx].legend()
    
    # Set common x-label
    axes[-1].set_xlabel('Epoch')
    
    # Add overall title
    fig.suptitle(f'Shared vs. Non-shared Heads Comparison\nDataset: {args.dataset}', 
                fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"{args.results_dir}/plots/shared_heads_comparison_{args.dataset}_{timestamp}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    
    # Create a bar chart comparing best accuracies
    plt.figure(figsize=(10, 6))
    
    # Prepare data for bar chart
    degree_labels = [f'Degree {d}' for d in degrees]
    shared_accs = []
    nonshared_accs = []
    
    for degree in degrees:
        shared_key = (degree, True)
        nonshared_key = (degree, False)
        
        shared_acc = results_dict.get(shared_key, {}).get('best_acc', 0)
        nonshared_acc = results_dict.get(nonshared_key, {}).get('best_acc', 0)
        
        shared_accs.append(shared_acc)
        nonshared_accs.append(nonshared_acc)
    
    # Set width of bars
    bar_width = 0.35
    x = np.arange(len(degrees))
    
    # Create bars
    plt.bar(x - bar_width/2, shared_accs, bar_width, label='Shared Heads')
    plt.bar(x + bar_width/2, nonshared_accs, bar_width, label='Non-shared Heads')
    
    # Add labels and formatting
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Best Test Accuracy (%)')
    plt.title(f'Best Accuracy Comparison: Shared vs. Non-shared Heads\nDataset: {args.dataset}')
    plt.xticks(x, degree_labels)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(shared_accs):
        plt.text(i - bar_width/2, v + 0.5, f'{v:.2f}', 
                 ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(nonshared_accs):
        plt.text(i + bar_width/2, v + 0.5, f'{v:.2f}', 
                 ha='center', va='bottom', fontsize=9)
    
    # Save the bar chart
    bar_path = f"{args.results_dir}/plots/shared_heads_bar_chart_{args.dataset}_{timestamp}.png"
    plt.savefig(bar_path)
    print(f"Saved bar chart to {bar_path}")
    
    # Print comparison table
    print("\nComparison Results:")
    print("=" * 70)
    print(f"{'Degree':^8} | {'Sharing Mode':^15} | {'Final Accuracy':^14} | {'Best Accuracy':^13}")
    print("-" * 70)
    
    for degree in degrees:
        for shared_heads in [True, False]:
            key = (degree, shared_heads)
            if key in results_dict:
                mode = "Shared" if shared_heads else "Non-shared"
                final_acc = results_dict[key]['final_acc']
                best_acc = results_dict[key]['best_acc']
                print(f"{degree:^8} | {mode:^15} | {final_acc:^14.2f} | {best_acc:^13.2f}")
    
    # Save results to CSV
    results_rows = []
    for degree in degrees:
        for shared_heads in [True, False]:
            key = (degree, shared_heads)
            if key in results_dict:
                mode = "Shared" if shared_heads else "Non-shared"
                final_acc = results_dict[key]['final_acc']
                best_acc = results_dict[key]['best_acc']
                results_rows.append({
                    'Degree': degree,
                    'Sharing Mode': mode,
                    'Final Accuracy': final_acc,
                    'Best Accuracy': best_acc
                })
    
    results_df = pd.DataFrame(results_rows)
    results_path = f"{args.results_dir}/shared_heads_results_{args.dataset}_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved detailed results to {results_path}")
    
    # Find the best configuration
    best_config = max(results_dict.items(), key=lambda x: x[1]['best_acc'])
    best_degree, best_shared = best_config[0]
    best_accuracy = best_config[1]['best_acc']
    
    print("\nBest Configuration:")
    print(f"  Polynomial Degree: {best_degree}")
    print(f"  Sharing Mode: {'Shared' if best_shared else 'Non-shared'}")
    print(f"  Best Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main() 