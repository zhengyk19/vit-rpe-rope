#!/usr/bin/env python
"""
Parameter search script for polynomial degree in PolynomialRPE.
This script runs multiple training jobs with different polynomial degrees
and logs the results for comparison.
"""

import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def get_args():
    """Parse command-line arguments for parameter search configuration."""
    parser = argparse.ArgumentParser(description='Polynomial Degree Parameter Search')
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    # Search range for polynomial degree
    parser.add_argument('--min_degree', type=int, default=1, help='Minimum polynomial degree to test')
    parser.add_argument('--max_degree', type=int, default=5, help='Maximum polynomial degree to test')
    # Training configuration
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs for each run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    # Parameter sharing mode
    parser.add_argument('--shared_heads', action='store_true', default=True,
                        help='Whether to share coefficients across attention heads')
    # Output directories
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save search results')
    return parser.parse_args()

def run_training(degree, args):
    """Run a training job with the specified polynomial degree."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"{args.results_dir}/logs"
    ckpt_dir = f"{args.results_dir}/checkpoints"
    
    # Create the command for training with specific polynomial degree
    cmd = [
        "python", "train.py",
        "--dataset", args.dataset,
        "--pos_encoding", "polynomial",
        "--poly_degree", str(degree),
        "--poly_shared_heads" if args.shared_heads else "--no-poly_shared_heads",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--log_dir", log_dir,
        "--ckpt_dir", ckpt_dir
    ]
    
    # Run the training process
    print(f"\n{'='*80}")
    print(f"Starting training with polynomial degree = {degree}")
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
        return log_path, degree
    else:
        # Find the most recent log file with 'polynomial' in the name
        log_files = [f for f in os.listdir(log_dir) if 'polynomial' in f]
        if log_files:
            # Sort by modification time
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            log_path = os.path.join(log_dir, log_files[0])
            return log_path, degree
    
    return None, degree

def main():
    """Run parameter search for polynomial degree."""
    args = get_args()
    
    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.results_dir}/plots", exist_ok=True)
    
    # Prepare for storing results
    degrees = range(args.min_degree, args.max_degree + 1)
    results = []
    
    # Run training for each degree value
    for degree in degrees:
        log_path, degree = run_training(degree, args)
        results.append((degree, log_path))
    
    # Analyze and visualize results
    analyze_results(results, args)

def analyze_results(results, args):
    """Analyze and visualize the results of the parameter search."""
    plt.figure(figsize=(10, 6))
    
    # Load and plot results for each degree
    final_accs = []
    best_accs = []
    
    for degree, log_path in results:
        if log_path and os.path.exists(log_path):
            df = pd.read_csv(log_path)
            
            # Plot the test accuracy curve
            plt.plot(df['epoch'], df['test_acc'], marker='o', label=f'Degree {degree}')
            
            # Store the final and best accuracies
            final_accs.append((degree, df['test_acc'].iloc[-1]))
            best_accs.append((degree, df['best_acc'].max()))
        else:
            print(f"Warning: No log file found for degree {degree}")
    
    # Format and save the plot
    plt.title(f'Test Accuracy vs. Epochs for Different Polynomial Degrees\nDataset: {args.dataset}, Shared Heads: {args.shared_heads}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"{args.results_dir}/plots/poly_degree_search_{args.dataset}_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Saved visualization to {plot_path}")
    
    # Print the final results
    print("\nFinal Results:")
    print("=" * 40)
    print("Polynomial Degree | Final Accuracy | Best Accuracy")
    print("-" * 40)
    
    for (deg1, final_acc), (deg2, best_acc) in zip(final_accs, best_accs):
        assert deg1 == deg2
        print(f"{deg1:^16} | {final_acc:^14.2f} | {best_acc:^13.2f}")
    
    # Determine the best degree
    best_degree = max(best_accs, key=lambda x: x[1])[0]
    best_accuracy = max(best_accs, key=lambda x: x[1])[1]
    
    print("\nBest polynomial degree:", best_degree)
    print(f"Best accuracy: {best_accuracy:.2f}%")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Degree': [d for d, _ in best_accs],
        'Final Accuracy': [a for _, a in final_accs],
        'Best Accuracy': [a for _, a in best_accs]
    })
    
    results_path = f"{args.results_dir}/poly_degree_results_{args.dataset}_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved detailed results to {results_path}")

if __name__ == "__main__":
    main() 