"""
RoPE Visualizer - Advanced Tool for Visualizing Rotary Position Encoding

This module provides comprehensive visualization tools for analyzing and comparing 
frequency patterns of different RoPE (Rotary Position Encoding) variants, including
Axial and Mixed implementations. It supports various input patterns, theta parameter
analysis, and trained model visualization.

The visualizer offers insights into how different RoPE implementations encode spatial
information in transformer-based vision models, helping to understand their representational
capabilities and differences.
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
    """Parse command-line arguments for visualization settings."""
    parser = argparse.ArgumentParser(description='RoPE Frequency Visualization Tool')
    
    # Configuration for visualization
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
    
    # Input position pattern configurations
    parser.add_argument('--patterns', type=str, nargs='+', 
                        default=['single', 'double', 'corner', 'diagonal'],
                        help='Test position patterns (options: single, double, corner, diagonal, custom)')
    parser.add_argument('--custom_pattern', type=str, default=None,
                       help='Custom pattern as comma-separated list of coordinates "row1,col1,row2,col2,..."')
    
    # Advanced visualization options
    parser.add_argument('--head_indices', type=int, nargs='+', default=[0],
                       help='Head indices to visualize for RoPE-Mixed (default: [0])')
    parser.add_argument('--compare_thetas', action='store_true',
                       help='Compare different theta values for sensitivity analysis')
    parser.add_argument('--theta_values', type=float, nargs='+', 
                        default=[10.0, 100.0, 1000.0],
                        help='Theta values to compare (default: [10.0, 100.0, 1000.0])')
    
    # Model loading parameters
    parser.add_argument('--load_model', action='store_true',
                       help='Load a trained model to visualize its learned frequencies')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_config', type=str, default='rope-mixed',
                       choices=['rope-axial', 'rope-mixed'],
                       help='Positional encoding method of the model')
    
    # Comparative model analysis
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare two models (Axial and Mixed) side by side')
    parser.add_argument('--axial_model_path', type=str, default=None,
                       help='Path to the RoPE-Axial model checkpoint')
    parser.add_argument('--mixed_model_path', type=str, default=None,
                       help='Path to the RoPE-Mixed model checkpoint')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 300)')
    parser.add_argument('--cmap', type=str, default='custom',
                       choices=['custom', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
                       help='Colormap for frequency visualization')
    
    return parser.parse_args()

def create_colormap(name):
    """Create a custom colormap for visualization."""
    if name == 'custom':
        # Purple -> Blue -> Green -> Yellow -> Orange
        colors = [(0.5, 0, 0.5), (0, 0, 0.5), (0, 1, 0), (1, 1, 0), (1, 0.5, 0)]
        return LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    else:
        return plt.get_cmap(name)

def create_input_positions(pattern, grid_size, custom_coords=None):
    """
    Generate input position tensor with a specific pattern for visualization.
    
    Args:
        pattern (str): Pattern name ('single', 'double', 'corner', 'diagonal', 'custom')
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
        # Two points for comparative analysis
        positions[grid_size//4, grid_size//4] = 1.0
        positions[grid_size//4, 3*grid_size//4] = 1.0
    elif pattern == 'corner':
        # Point in the corner to test boundary representations
        positions[0, 0] = 1.0
    elif pattern == 'diagonal':
        # Points on the diagonal to test spatial correlations
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
    Apply RoPE encoding and Fast Fourier Transform to analyze frequency representation.
    
    This function takes an input position grid, applies RoPE encoding (either Axial or Mixed),
    and performs 2D FFT to visualize the frequency domain representation. It also performs
    the inverse FFT to evaluate reconstruction quality.
    
    Args:
        pos_grid (torch.Tensor): Input position grid
        rope_encoder: RoPE encoder instance (Axial or Mixed)
        head_idx (int): Head index to use for Mixed encoding (ignored for Axial)
        
    Returns:
        tuple: (frequency representation after FFT, reconstructed positions after inverse FFT)
    """
    print("Using encoder:", rope_encoder.__class__.__name__)
    grid_size = pos_grid.shape[0]
    seq_len = grid_size * grid_size
    
    # Get device safely
    try:
        device = next(rope_encoder.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    # Flatten the grid to sequence format
    seq = pos_grid.flatten().to(device)  # [seq_len]
    
    # Get frequency components based on encoder type
    if isinstance(rope_encoder, RoPEAxial):
        cos, sin = rope_encoder.get_freqs_cis(seq_len, device)
        # Mask the frequencies with input positions
        cos_masked = seq[:, None] * cos  # [seq_len, dim/2]
        sin_masked = seq[:, None] * sin  # [seq_len, dim/2]
    else:
        cos, sin = rope_encoder.get_freqs_cis(seq_len, device)
        # Get specific head and mask with input positions
        cos_head = cos[head_idx]  # [seq_len, dim/2]
        sin_head = sin[head_idx]  # [seq_len, dim/2]
        cos_masked = seq[:, None] * cos_head
        sin_masked = seq[:, None] * sin_head
    
    # Create complex values using both cosine and sine components
    complex_vals = cos_masked + 1j * sin_masked
    
    # Convert to numpy and mean across frequency dimension
    complex_grid = np.mean(complex_vals.detach().cpu().numpy(), axis=-1).reshape(grid_size, grid_size)
    
    # Apply 2D FFT for frequency domain analysis
    fft_result = np.fft.fft2(complex_grid)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Apply inverse FFT to evaluate representation quality
    ifft_result = np.fft.ifft2(fft_result)
    reconstructed = np.abs(ifft_result)
    
    # Normalize for visualization
    magnitude = np.abs(fft_shifted)
    magnitude_log = np.log1p(magnitude)
    
    return magnitude_log, reconstructed

def load_trained_model(model_path, model_config, num_heads=4, dim=64, theta_axial=100.0, theta_mixed=10.0, device='cpu'):
    """
    Load a trained model to extract its RoPE encoder.
    
    Args:
        model_path: Path to the model checkpoint
        model_config: Positional encoding method ('rope-axial' or 'rope-mixed')
        num_heads: Number of attention heads
        dim: Embedding dimension
        theta_axial: Theta parameter for RoPE-Axial
        theta_mixed: Theta parameter for RoPE-Mixed
        device: Device to load the model on
        
    Returns:
        RoPE encoder from the trained model, or None if loading fails
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        # Load the model checkpoint to inspect its structure
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract dimension and number of heads from the checkpoint
        model_dim = None
        model_heads = None
        
        # Examine the checkpoint structure to determine the dimensions
        for key, value in checkpoint.items():
            if key == 'patch_embed.weight':
                # Patch embedding weight has shape [embed_dim, in_chans, patch_size, patch_size]
                model_dim = value.shape[0]
            elif key.endswith('.qkv.weight'):
                # QKV weight has shape [3*embed_dim, embed_dim]
                model_dim = value.shape[1]
            elif key.endswith('.qkv.bias'):
                # QKV bias has shape [3*embed_dim]
                model_dim = value.shape[0] // 3
            elif 'pos_embed.inv_freq' in key:
                # inv_freq is [dim//2]
                model_dim = value.shape[0] * 2
            
            # For RoPE-Mixed, try to determine number of heads
            if model_config == 'rope-mixed' and 'pos_embed.freqs' in key:
                if len(value.shape) >= 1:
                    model_heads = value.shape[0]
            
            # Break once we have both dimensions
            if model_dim is not None and (model_config == 'rope-axial' or model_heads is not None):
                break
        
        # If we couldn't determine dimensions, try to get them from error message
        if model_dim is None:
            # Create a temporary model with default dimensions
            temp_model = VisionTransformer(
                pos_encoding=model_config,
                num_heads=num_heads
            ).to(device)
            
            try:
                # Try loading and catch the error to extract dimensions
                temp_model.load_state_dict(checkpoint)
            except RuntimeError as e:
                # Parse error message to extract dimensions
                error_msg = str(e)
                import re
                
                # Look for dimension mismatches in the error
                dim_matches = re.findall(r'size mismatch for .*: copying a param with shape torch.Size\(\[(.*)\]\)', error_msg)
                if dim_matches:
                    if model_config == 'rope-axial' and 'inv_freq' in error_msg:
                        # inv_freq is [dim//2]
                        model_dim = int(dim_matches[0]) * 2
                    elif model_config == 'rope-mixed' and 'freqs' in error_msg:
                        # freqs is [num_heads, seq_len, dim//2]
                        parts = dim_matches[0].split(', ')
                        if len(parts) >= 3:
                            model_heads = int(parts[0])
                            model_dim = int(parts[2]) * 2
        
        # If we still couldn't determine dimensions, use arguments
        if model_dim is None:
            print("Warning: Could not determine model dimensions from checkpoint")
            model_dim = dim  # Use provided dimension
        
        if model_config == 'rope-mixed' and model_heads is None:
            print("Warning: Could not determine number of heads from checkpoint")
            model_heads = num_heads  # Use provided value
        
        print(f"Creating model with dim={model_dim}" + 
              (f", heads={model_heads}" if model_config == 'rope-mixed' else ""))
        
        # Create a model with matching dimensions
        if model_config == 'rope-axial':
            return RoPEAxial(dim=model_dim, theta=theta_axial).to(device)
        else:
            return RoPEMixed(dim=model_dim, num_heads=model_heads, theta=theta_mixed).to(device)
    
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
    
    # Get model basename for output filename
    if args.load_model and args.model_path:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    else:
        model_name = "default"
    
    # Set up RoPE encoders
    if args.load_model and args.model_path:
        # Load encoders from trained model
        encoder = load_trained_model(
            model_path=args.model_path, 
            model_config=args.model_config, 
            num_heads=args.num_heads, 
            dim=args.dim, 
            theta_axial=args.theta_axial, 
            theta_mixed=args.theta_mixed, 
            device=device
        )
        if encoder is None:
            print("Model loading failed. Exiting without visualization.")
            return
        else:
            print(f"Successfully loaded RoPE encoder from model: {args.model_path}")
            if args.model_config == 'rope-axial':
                rope_axial = encoder
                rope_mixed = None
                use_both_encoders = False
            else:
                rope_mixed = encoder
                rope_axial = None
                use_both_encoders = False
    else:
        # Use default encoders for visualization and comparison
        print("Using default encoders")
        rope_axial = RoPEAxial(dim=args.dim, theta=args.theta_axial).to(device)
        rope_mixed = RoPEMixed(dim=args.dim, num_heads=args.num_heads, theta=args.theta_mixed).to(device)
        use_both_encoders = True
    
    # Process each test position pattern
    for pattern in args.patterns:
        # Handle custom pattern
        if pattern == 'custom' and args.custom_pattern:
            custom_coords = args.custom_pattern.split(',')
            input_positions = create_input_positions(pattern, args.grid_size, custom_coords)
        else:
            input_positions = create_input_positions(pattern, args.grid_size)
        
        # Apply RoPE and FFT based on the loaded model type
        if use_both_encoders:
            # Using both encoders for comparison
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
                model_info = f"From model: {model_name}"
                fig.suptitle(f"RoPE Frequency Visualization - Pattern: {pattern.capitalize()}\n{model_info}", 
                            fontsize=14, y=1.05)
                
                # Save the figure
                head_suffix = f"_head{head_idx}" if len(args.head_indices) > 1 else ""
                output_path = f"{args.output_dir}/rope_freq_{model_name}_{pattern}{head_suffix}_{timestamp}.png"
                plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
                print(f"Saved visualization for '{pattern}' pattern (head {head_idx}) to {output_path}")
                
                # Close the figure to free memory
                plt.close(fig)
        else:
            # Using only the loaded model's encoder
            if args.model_config == 'rope-axial':
                model_freqs, model_recon = apply_rope_and_fft(input_positions, rope_axial)
                encoder_name = 'Axial'
                theta = args.theta_axial
                head_idx = None
            else:
                # For RoPE-Mixed models, create visualization for each requested head
                for head_idx in args.head_indices:
                    if head_idx >= args.num_heads:
                        continue
                        
                    model_freqs, model_recon = apply_rope_and_fft(input_positions, rope_mixed, head_idx=head_idx)
                    encoder_name = f'Mixed Head {head_idx}' if len(args.head_indices) > 1 else 'Mixed'
                    theta = args.theta_mixed
                    
                    # Create a figure for visualization
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    
                    # Plot input positions
                    axes[0].imshow(input_positions.numpy(), cmap='viridis')
                    axes[0].set_title('Input Positions')
                    axes[0].axis('off')
                    
                    # Add FFT & iFFT arrow
                    axes[1].axis('off')
                    axes[1].text(0.5, 0.5, 'FFT\n&\niFFT', ha='center', va='center', fontsize=14)
                    axes[1].set_title('')
                    
                    # Plot model frequencies
                    im_model = axes[2].imshow(model_freqs, cmap=cmap)
                    axes[2].set_title(f'{encoder_name} (θ={theta})')
                    axes[2].axis('off')
                    
                    # Add visualization of reconstruction
                    axes[3].imshow(model_recon, cmap='viridis')
                    axes[3].set_title('Reconstruction')
                    axes[3].axis('off')
                    
                    # Add colorbar
                    fig.colorbar(im_model, ax=axes[2], fraction=0.046, pad=0.04)
                    
                    # Fine-tune layout
                    plt.tight_layout()
                    
                    # Add figure title
                    model_info = f"From model: {model_name}"
                    fig.suptitle(f"RoPE Frequency Visualization - Pattern: {pattern.capitalize()}\n{model_info}", 
                                fontsize=14, y=1.05)
                    
                    # Save the figure
                    head_suffix = f"_head{head_idx}" if len(args.head_indices) > 1 else ""
                    output_path = f"{args.output_dir}/rope_freq_{model_name}_{pattern}{head_suffix}_{timestamp}.png"
                    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
                    print(f"Saved visualization for '{pattern}' pattern (head {head_idx}) to {output_path}")
                    
                    # Close the figure to free memory
                    plt.close(fig)
                
                # Skip the rest of the loop for Mixed models since we've handled all heads
                continue
                
            # For Axial models
            # Create a figure for visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Plot input positions
            axes[0].imshow(input_positions.numpy(), cmap='viridis')
            axes[0].set_title('Input Positions')
            axes[0].axis('off')
            
            # Add FFT & iFFT arrow
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'FFT\n&\niFFT', ha='center', va='center', fontsize=14)
            axes[1].set_title('')
            
            # Plot model frequencies
            im_model = axes[2].imshow(model_freqs, cmap=cmap)
            axes[2].set_title(f'{encoder_name} (θ={theta})')
            axes[2].axis('off')
            
            # Add visualization of reconstruction
            axes[3].imshow(model_recon, cmap='viridis')
            axes[3].set_title('Reconstruction')
            axes[3].axis('off')
            
            # Add colorbar
            fig.colorbar(im_model, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Fine-tune layout
            plt.tight_layout()
            
            # Add figure title
            model_info = f"From model: {model_name}"
            fig.suptitle(f"RoPE Frequency Visualization - Pattern: {pattern.capitalize()}\n{model_info}", 
                        fontsize=14, y=1.05)
            
            # Save the figure
            output_path = f"{args.output_dir}/rope_freq_{model_name}_{pattern}_{timestamp}.png"
            plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved visualization for '{pattern}' pattern to {output_path}")
            
            # Close the figure to free memory
            plt.close(fig)
    
    # Create theta comparison if requested
    if args.compare_thetas:
        visualize_theta_comparison(args, cmap, timestamp, model_name)

def visualize_theta_comparison(args, cmap, timestamp, model_name="default"):
    """
    Compare different theta values for RoPE frequency patterns.
    
    Args:
        args: Command-line arguments
        cmap: Colormap for visualization
        timestamp: Timestamp for output filename
        model_name: Name of the model for output filename
    """
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
    output_path = f"{args.output_dir}/rope_theta_comparison_{model_name}_{timestamp}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved theta comparison to {output_path}")
    plt.close(fig)

def visualize_model_comparison(args):
    """
    Compare RoPE-Axial and RoPE-Mixed models side by side.
    
    Args:
        args: Command-line arguments with visualization parameters
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get colormap
    cmap = create_colormap(args.cmap)
    
    # Load RoPE-Axial model
    rope_axial = load_trained_model(
        model_path=args.axial_model_path, 
        model_config='rope-axial', 
        num_heads=args.num_heads, 
        dim=args.dim, 
        theta_axial=args.theta_axial, 
        device=device
    )
    
    if rope_axial is None:
        print("Failed to load RoPE-Axial model. Exiting without visualization.")
        return
    
    # Load RoPE-Mixed model
    rope_mixed = load_trained_model(
        model_path=args.mixed_model_path, 
        model_config='rope-mixed', 
        num_heads=args.num_heads, 
        dim=args.dim, 
        theta_mixed=args.theta_mixed, 
        device=device
    )
    
    if rope_mixed is None:
        print("Failed to load RoPE-Mixed model. Exiting without visualization.")
        return
    
    print(f"Successfully loaded both models for comparison")
    axial_name = os.path.splitext(os.path.basename(args.axial_model_path))[0]
    mixed_name = os.path.splitext(os.path.basename(args.mixed_model_path))[0]
    
    # Process each test position pattern
    for pattern in args.patterns:
        # Handle custom pattern
        if pattern == 'custom' and args.custom_pattern:
            custom_coords = args.custom_pattern.split(',')
            input_positions = create_input_positions(pattern, args.grid_size, custom_coords)
        else:
            input_positions = create_input_positions(pattern, args.grid_size)
        
        # Get Axial frequencies and reconstruction
        axial_freqs, axial_recon = apply_rope_and_fft(input_positions, rope_axial)
        
        # For each requested head in the Mixed model
        for head_idx in args.head_indices:
            # Get Mixed frequencies and reconstruction
            mixed_freqs, mixed_recon = apply_rope_and_fft(input_positions, rope_mixed, head_idx=head_idx)
            
            # Create a figure with 6 subplots for side-by-side comparison
            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            
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
            axes[2].set_title(f'Axial')
            axes[2].axis('off')
            
            # Plot Mixed frequencies
            im_mixed = axes[3].imshow(mixed_freqs, cmap=cmap)
            head_title = f'Mixed Head {head_idx}' if len(args.head_indices) > 1 else 'Mixed'
            axes[3].set_title(f'{head_title}')
            axes[3].axis('off')
            
            # Add visualization of Axial reconstruction
            axes[4].imshow(axial_recon, cmap='viridis')
            axes[4].set_title('Axial Reconstruction')
            axes[4].axis('off')
            
            # Add visualization of Mixed reconstruction
            axes[5].imshow(mixed_recon, cmap='viridis')
            axes[5].set_title('Mixed Reconstruction')
            axes[5].axis('off')
            
            # Add colorbars
            fig.colorbar(im_axial, ax=axes[2], fraction=0.046, pad=0.04)
            fig.colorbar(im_mixed, ax=axes[3], fraction=0.046, pad=0.04)
            
            # Fine-tune layout
            plt.tight_layout()
            
            # Add figure title
            fig.suptitle(f"RoPE Model Comparison - Pattern: {pattern.capitalize()}\nAxial: {axial_name}, Mixed: {mixed_name}", 
                        fontsize=14, y=1.05)
            
            # Save the figure
            head_suffix = f"_head{head_idx}" if len(args.head_indices) > 1 else ""
            output_path = f"{args.output_dir}/rope_comparison_{pattern}{head_suffix}_{timestamp}.png"
            plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved model comparison for '{pattern}' pattern (head {head_idx}) to {output_path}")
            
            # Close the figure to free memory
            plt.close(fig)

if __name__ == "__main__":
    args = get_args()
    
    if args.compare_models and args.axial_model_path and args.mixed_model_path:
        visualize_model_comparison(args)
    else:
        visualize_rope_frequencies(args) 