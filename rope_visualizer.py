#!/usr/bin/env python
"""
RoPE Frequency Visualizer - A tool to visualize and compare different RoPE variants.

This script provides a user-friendly interface to:
1. Visualize Rotary Position Encoding (RoPE) frequency patterns
2. Compare Axial and Mixed variants
3. Examine different input patterns and their frequency representations
4. Optionally load trained models to visualize their learned frequencies
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.positional_encoding import RoPEAxial, RoPEMixed
from models.vit import VisionTransformer
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os
from datetime import datetime
import seaborn as sns

def get_args():
    """Parse command-line arguments for the visualization tool."""
    parser = argparse.ArgumentParser(description='RoPE Frequency Visualization Tool')
    
    # Basic visualization settings
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
    
    # Test position patterns
    parser.add_argument('--patterns', type=str, nargs='+', 
                        default=['single', 'double', 'corner', 'diagonal'],
                        help='Test position patterns (options: single, double, corner, diagonal, custom)')
    parser.add_argument('--custom_pattern', type=str, default=None,
                       help='Custom pattern as comma-separated list of coordinates "row1,col1,row2,col2,..."')
    
    # Advanced options
    parser.add_argument('--head_indices', type=int, nargs='+', default=[0],
                       help='Head indices to visualize for RoPE-Mixed (default: [0])')
    parser.add_argument('--compare_thetas', action='store_true',
                       help='Compare different theta values for sensitivity analysis')
    parser.add_argument('--theta_values', type=float, nargs='+', 
                        default=[10.0, 100.0, 1000.0],
                        help='Theta values to compare (default: [10.0, 100.0, 1000.0])')
    
    # Model loading
    parser.add_argument('--load_model', action='store_true',
                       help='Load a trained model to visualize its learned frequencies')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_config', type=str, default='rope-mixed',
                       choices=['rope-axial', 'rope-mixed'],
                       help='Positional encoding method of the model')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 300)')
    parser.add_argument('--cmap', type=str, default='custom',
                       choices=['custom', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
                       help='Colormap for frequency visualization')
    
    return parser.parse_args()

def create_colormap(name):
    """Create a colormap for visualization."""
    if name == 'custom':
        # Purple -> Blue -> Green -> Yellow -> Orange
        colors = [(0.5, 0, 0.5), (0, 0, 0.5), (0, 1, 0), (1, 1, 0), (1, 0.5, 0)]
        return LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    else:
        return plt.get_cmap(name)

def create_input_positions(pattern, grid_size, custom_coords=None):
    """
    Create input position tensor with a specific pattern.
    
    Args:
        pattern (str): Pattern name ('single', 'double', 'corner', 'diagonal', or 'custom')
        grid_size (int): Size of the grid
        custom_coords (list): List of row,col coordinates for custom pattern
        
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
    elif pattern == 'custom' and custom_coords:
        # Parse custom coordinates
        coords = list(map(int, custom_coords))
        for i in range(0, len(coords), 2):
            if i+1 < len(coords):
                row, col = coords[i], coords[i+1]
                if 0 <= row < grid_size and 0 <= col < grid_size:
                    positions[row, col] = 1.0
    
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
        freqs = cos.detach().cpu().numpy()  # Just use cosine component for visualization
    else:
        cos, sin = rope_encoder.get_freqs_cis(seq_len, device)
        # RoPE-Mixed returns [num_heads, seq_len, dim/2] tensors
        freqs = cos[head_idx].detach().cpu().numpy()  # Use a specific head
    
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

def load_trained_model(model_path, model_config, num_heads=4, device='cpu'):
    """
    Load a trained model to extract its RoPE encoder.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_config (str): Positional encoding method
        num_heads (int): Number of attention heads
        device (str): Device to load the model on
        
    Returns:
        nn.Module: RoPE encoder from the trained model
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create a temporary model with the same config to load the state dict
        model = VisionTransformer(
            pos_encoding=model_config,
            num_heads=num_heads
        ).to(device)
        
        # Load the state dictionary
        model.load_state_dict(checkpoint)
        
        # Extract the RoPE encoder
        encoder = model.pos_embed
        return encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def visualize_rope_frequencies(args):
    """
    Create visualizations of RoPE frequency representations.
    
    Args:
        args: Command-line arguments containing visualization parameters
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get colormap
    cmap = create_colormap(args.cmap)
    
    # Set up RoPE encoders
    if args.load_model and args.model_path:
        # Load encoders from trained model
        encoder = load_trained_model(args.model_path, args.model_config, args.num_heads, device)
        if encoder is None:
            print("Falling back to default encoders...")
            rope_axial = RoPEAxial(dim=args.dim, theta=args.theta_axial).to(device)
            rope_mixed = RoPEMixed(dim=args.dim, num_heads=args.num_heads, theta=args.theta_mixed).to(device)
        else:
            print(f"Successfully loaded RoPE encoder from model: {args.model_path}")
            if args.model_config == 'rope-axial':
                rope_axial = encoder
                rope_mixed = RoPEMixed(dim=args.dim, num_heads=args.num_heads, theta=args.theta_mixed).to(device)
            else:
                rope_mixed = encoder
                rope_axial = RoPEAxial(dim=args.dim, theta=args.theta_axial).to(device)
    else:
        # Use default encoders
        rope_axial = RoPEAxial(dim=args.dim, theta=args.theta_axial).to(device)
        rope_mixed = RoPEMixed(dim=args.dim, num_heads=args.num_heads, theta=args.theta_mixed).to(device)
    
    # Process each test position pattern
    for pattern in args.patterns:
        # Handle custom pattern
        if pattern == 'custom' and args.custom_pattern:
            custom_coords = args.custom_pattern.split(',')
            input_positions = create_input_positions(pattern, args.grid_size, custom_coords)
        else:
            input_positions = create_input_positions(pattern, args.grid_size)
        
        # Apply RoPE and FFT
        axial_freqs, axial_recon = apply_rope_and_fft(input_positions, rope_axial)
        
        # Create visualization for each head if using mixed
        for head_idx in args.head_indices:
            if head_idx >= args.num_heads:
                continue
                
            # For Mixed, visualize the specified head
            mixed_freqs, mixed_recon = apply_rope_and_fft(input_positions, rope_mixed, head_idx=head_idx)
            
            # Create a figure for visualization
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Plot input positions
            axes[0].imshow(input_positions.numpy(), cmap='viridis')
            axes[0].set_title('Input Positions')
            axes[0].axis('off')
            
            # Add FFT & iFFT arrow
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'FFT\n&\niFFT', ha='center', va='center', fontsize=14)
            axes[1].set_title('')
            
            # Plot Axial frequencies
            im_axial = axes[2].imshow(axial_freqs, cmap=cmap)
            axes[2].set_title(f'Axial (θ={args.theta_axial})')
            axes[2].axis('off')
            
            # Plot Mixed frequencies
            im_mixed = axes[3].imshow(mixed_freqs, cmap=cmap)
            head_title = f'Mixed Head {head_idx}' if len(args.head_indices) > 1 else 'Mixed'
            axes[3].set_title(f'{head_title} (θ={args.theta_mixed})')
            axes[3].axis('off')
            
            # Add visualization of reconstruction
            axes[4].imshow(mixed_recon, cmap='viridis')
            axes[4].set_title('Reconstruction')
            axes[4].axis('off')
            
            # Add colorbar
            fig.colorbar(im_mixed, ax=axes[3], fraction=0.046, pad=0.04)
            fig.colorbar(im_axial, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Fine-tune layout
            plt.tight_layout()
            
            # Add figure title
            model_info = f"From model: {os.path.basename(args.model_path)}" if args.load_model else "Default parameters"
            fig.suptitle(f"RoPE Frequency Visualization - Pattern: {pattern.capitalize()}\n{model_info}", 
                        fontsize=14, y=1.05)
            
            # Save the figure
            head_suffix = f"_head{head_idx}" if len(args.head_indices) > 1 else ""
            output_path = f"{args.output_dir}/rope_freq_{pattern}{head_suffix}_{timestamp}.png"
            plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved visualization for '{pattern}' pattern (head {head_idx}) to {output_path}")
            
            # Close the figure to free memory
            plt.close(fig)
    
    # Create theta comparison if requested
    if args.compare_thetas:
        visualize_theta_comparison(args, cmap, timestamp)

def visualize_theta_comparison(args, cmap, timestamp):
    """Create visualizations comparing different theta values for RoPE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the first pattern for comparison
    pattern = args.patterns[0]
    if pattern == 'custom' and args.custom_pattern:
        custom_coords = args.custom_pattern.split(',')
        input_positions = create_input_positions(pattern, args.grid_size, custom_coords)
    else:
        input_positions = create_input_positions(pattern, args.grid_size)
    
    # Create a figure with a row for each theta value
    fig, axes = plt.subplots(len(args.theta_values), 3, figsize=(12, 4*len(args.theta_values)))
    
    for i, theta in enumerate(args.theta_values):
        # Create encoders with this theta
        rope_axial = RoPEAxial(dim=args.dim, theta=theta).to(device)
        rope_mixed = RoPEMixed(dim=args.dim, num_heads=args.num_heads, theta=theta).to(device)
        
        # Get frequencies
        axial_freqs, _ = apply_rope_and_fft(input_positions, rope_axial)
        mixed_freqs, _ = apply_rope_and_fft(input_positions, rope_mixed, head_idx=args.head_indices[0])
        
        # Get row of axes
        if len(args.theta_values) == 1:
            row_axes = axes
        else:
            row_axes = axes[i]
        
        # Plot input positions (only for first row)
        if i == 0:
            row_axes[0].imshow(input_positions.numpy(), cmap='viridis')
            row_axes[0].set_title('Input Positions')
        else:
            row_axes[0].text(0.5, 0.5, f'θ = {theta}', ha='center', va='center', fontsize=14)
            row_axes[0].set_title('')
        row_axes[0].axis('off')
        
        # Plot Axial frequencies
        im_axial = row_axes[1].imshow(axial_freqs, cmap=cmap)
        row_axes[1].set_title('Axial' if i == 0 else '')
        row_axes[1].axis('off')
        
        # Plot Mixed frequencies
        im_mixed = row_axes[2].imshow(mixed_freqs, cmap=cmap)
        row_axes[2].set_title('Mixed' if i == 0 else '')
        row_axes[2].axis('off')
        
        # Add colorbar if last row
        if i == len(args.theta_values) - 1:
            fig.colorbar(im_axial, ax=row_axes[1], fraction=0.046, pad=0.04)
            fig.colorbar(im_mixed, ax=row_axes[2], fraction=0.046, pad=0.04)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add title
    fig.suptitle(f"Impact of Theta (θ) on RoPE Frequency Patterns\nPattern: {pattern.capitalize()}", 
                fontsize=16, y=1.02)
    
    # Save the figure
    output_path = f"{args.output_dir}/rope_theta_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved theta comparison to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    args = get_args()
    visualize_rope_frequencies(args) 