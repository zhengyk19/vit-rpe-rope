#!/usr/bin/env python
"""
Visualize RoPE frequencies using 2D Fourier reconstruction.

This script generates visualizations of Rotary Position Encoding (RoPE) frequency
representations by:
1. Creating sample 2D positional inputs
2. Applying RoPE frequency encoding (Axial and Mixed methods)
3. Using FFT and inverse FFT to evaluate representation capabilities
4. Visualizing the results similar to the paper figure
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.positional_encoding import RoPEAxial, RoPEMixed
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os
from datetime import datetime

def get_args():
    """Parse command-line arguments for visualization configuration."""
    parser = argparse.ArgumentParser(description='RoPE Frequency Visualization')
    parser.add_argument('--grid_size', type=int, default=8, 
                        help='Size of grid for visualization (default: 8)')
    parser.add_argument('--dim', type=int, default=64, 
                        help='Dimension for RoPE encoding (default: 64)')
    parser.add_argument('--theta_axial', type=float, default=100.0,
                       help='Theta parameter for RoPE-Axial (default: 100.0)')
    parser.add_argument('--theta_mixed', type=float, default=10.0,
                       help='Theta parameter for RoPE-Mixed (default: 10.0)')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads for RoPE-Mixed (default: 4)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--test_positions', type=str, nargs='+', 
                        default=['single', 'double'],
                        help='Test position patterns to visualize (options: single, double, corner)')
    return parser.parse_args()

def create_input_positions(pattern, grid_size):
    """
    Create input position tensor with a specific pattern.
    
    Args:
        pattern (str): Pattern name ('single', 'double', or 'corner')
        grid_size (int): Size of the grid
        
    Returns:
        torch.Tensor: Position tensor with 1s at specified positions, 0s elsewhere
    """
    positions = torch.zeros(grid_size, grid_size)
    
    if pattern == 'single':
        # Single point in the middle-left area
        positions[grid_size//2, grid_size//4] = 1.0
    elif pattern == 'double':
        # Two points
        positions[grid_size//4, grid_size//4] = 1.0
        positions[grid_size//4, 3*grid_size//4] = 1.0
    elif pattern == 'corner':
        # Point in the corner
        positions[0, 0] = 1.0
    elif pattern == 'diagonal':
        # Points on the diagonal
        for i in range(grid_size):
            positions[i, i] = 1.0
    
    return positions

def apply_rope_and_fft(pos_grid, rope_encoder, head_idx=0):
    """
    Apply RoPE encoding and FFT to evaluate representation.
    
    Args:
        pos_grid (torch.Tensor): Input position grid
        rope_encoder (nn.Module): RoPE encoder (Axial or Mixed)
        head_idx (int): Head index to use for Mixed encoding
        
    Returns:
        tuple: (frequency representation after FFT, reconstructed positions after iFFT)
    """
    grid_size = pos_grid.shape[0]
    seq_len = grid_size * grid_size
    
    # Get device safely
    try:
        device = next(rope_encoder.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    # Flatten the grid to sequence format
    seq_positions = pos_grid.flatten().unsqueeze(0)  # [1, seq_len]
    
    # Get frequency components
    if isinstance(rope_encoder, RoPEAxial):
        cos, sin = rope_encoder.get_freqs_cis(seq_len, device)
        # RoPE-Axial returns [seq_len, dim/2] tensors
        freqs = cos.detach().numpy()  # Just use cosine component for visualization
    else:
        cos, sin = rope_encoder.get_freqs_cis(seq_len, device)
        # RoPE-Mixed returns [num_heads, seq_len, dim/2] tensors
        freqs = cos[head_idx].detach().numpy()  # Use a specific head
    
    # Reshape frequencies to 2D grid for visualization
    freq_grid = np.mean(freqs, axis=-1).reshape(grid_size, grid_size)
    
    # Apply 2D FFT
    fft_result = np.fft.fft2(freq_grid)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Apply inverse FFT to see representation quality
    ifft_result = np.fft.ifft2(fft_result)
    reconstructed = np.abs(ifft_result)
    
    # Normalize for visualization
    magnitude = np.abs(fft_shifted)
    magnitude_log = np.log1p(magnitude)
    
    return magnitude_log, reconstructed

def visualize_rope_frequencies(args):
    """
    Create visualizations of RoPE frequency representations.
    
    Args:
        args: Command-line arguments containing visualization parameters
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up RoPE encoders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rope_axial = RoPEAxial(dim=args.dim, theta=args.theta_axial).to(device)
    rope_mixed = RoPEMixed(dim=args.dim, num_heads=args.num_heads, theta=args.theta_mixed).to(device)
    
    # Process each test position pattern
    for pattern in args.test_positions:
        # Create input positions
        input_positions = create_input_positions(pattern, args.grid_size)
        
        # Apply RoPE and FFT
        axial_freqs, axial_recon = apply_rope_and_fft(input_positions, rope_axial)
        
        # For Mixed, visualize head 0 (could do multiple heads)
        mixed_freqs, mixed_recon = apply_rope_and_fft(input_positions, rope_mixed, head_idx=0)
        
        # Create a figure for visualization
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # Custom colormap similar to the figure
        colors = [(0.5, 0, 0.5), (0, 0, 0.5), (0, 1, 0), (1, 1, 0), (1, 0.5, 0)]  # Purple -> Blue -> Green -> Yellow -> Orange
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Plot input positions
        axes[0].imshow(input_positions.numpy(), cmap='viridis')
        axes[0].set_title('Input Positions')
        axes[0].axis('off')
        
        # Add FFT & iFFT arrow
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'FFT\n&\niFFT', ha='center', va='center', fontsize=14)
        axes[1].set_title('')
        
        # Plot Axial frequencies
        axes[2].imshow(axial_freqs, cmap=cmap)
        axes[2].set_title('Axial Frequencies')
        axes[2].axis('off')
        
        # Plot Mixed frequencies
        axes[3].imshow(mixed_freqs, cmap=cmap)
        axes[3].set_title('Mixed Frequencies')
        axes[3].axis('off')
        
        # Add visualization of reconstruction if desired
        axes[4].imshow(mixed_recon, cmap='viridis')
        axes[4].set_title('Reconstruction')
        axes[4].axis('off')
        
        # Fine-tune layout
        plt.tight_layout()
        
        # Save the figure
        output_path = f"{args.output_dir}/rope_freq_{pattern}_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization for '{pattern}' pattern to {output_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    # Create a second figure with multiple patterns side by side
    if len(args.test_positions) > 1:
        visualize_multiple_patterns(args, rope_axial, rope_mixed, timestamp)

def visualize_multiple_patterns(args, rope_axial, rope_mixed, timestamp):
    """Create a combined visualization with multiple input patterns side by side."""
    # Create a figure for the side-by-side comparison
    patterns = args.test_positions
    fig, axes = plt.subplots(len(patterns), 5, figsize=(20, 4*len(patterns)))
    
    # Custom colormap
    colors = [(0.5, 0, 0.5), (0, 0, 0.5), (0, 1, 0), (1, 1, 0), (1, 0.5, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    for i, pattern in enumerate(patterns):
        # Create input positions
        input_positions = create_input_positions(pattern, args.grid_size)
        
        # Apply RoPE and FFT
        axial_freqs, _ = apply_rope_and_fft(input_positions, rope_axial)
        mixed_freqs, mixed_recon = apply_rope_and_fft(input_positions, rope_mixed, head_idx=0)
        
        # Access row of subplots
        if len(patterns) == 1:
            row_axes = axes
        else:
            row_axes = axes[i]
        
        # Plot input positions
        row_axes[0].imshow(input_positions.numpy(), cmap='viridis')
        row_axes[0].set_title('Input Positions' if i == 0 else '')
        row_axes[0].axis('off')
        
        # Add FFT & iFFT arrow
        row_axes[1].axis('off')
        if i == 0:
            row_axes[1].text(0.5, 0.5, 'FFT\n&\niFFT', ha='center', va='center', fontsize=14)
        
        # Plot Axial frequencies
        row_axes[2].imshow(axial_freqs, cmap=cmap)
        row_axes[2].set_title('Axial Frequencies' if i == 0 else '')
        row_axes[2].axis('off')
        
        # Plot Mixed frequencies
        row_axes[3].imshow(mixed_freqs, cmap=cmap)
        row_axes[3].set_title('Mixed Frequencies' if i == 0 else '')
        row_axes[3].axis('off')
        
        # Plot reconstruction if desired
        row_axes[4].imshow(mixed_recon, cmap='viridis')
        row_axes[4].set_title('Reconstruction' if i == 0 else '')
        row_axes[4].axis('off')
    
    # Fine-tune layout
    plt.tight_layout()
    
    # Add a title for the entire figure
    fig.suptitle("2D Fourier Reconstruction with RoPE Frequencies", fontsize=16, y=1.02)
    
    # Save the figure
    output_path = f"{args.output_dir}/rope_freq_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    args = get_args()
    visualize_rope_frequencies(args) 