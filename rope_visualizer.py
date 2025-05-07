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
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare Axial and Mixed models side by side')
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
    parser.add_argument('--axial_model', type=str, default=None,
                       help='Path to RoPE-Axial model checkpoint for comparison')
    parser.add_argument('--mixed_model', type=str, default=None,
                       help='Path to RoPE-Mixed model checkpoint for comparison')
    
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
    
    # Get frequency components
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
    
    # Apply 2D FFT
    fft_result = np.fft.fft2(complex_grid)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Apply inverse FFT to see representation quality
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
        model_path (str): Path to the model checkpoint
        model_config (str): Positional encoding method
        num_heads (int): Number of attention heads
        dim (int): Embedding dimension
        theta_axial (float): Theta parameter for RoPE-Axial
        theta_mixed (float): Theta parameter for RoPE-Mixed
        device (str): Device to load the model on
        
    Returns:
        nn.Module: RoPE encoder from the trained model, or None if loading fails
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

def compare_rope_models(axial_model, mixed_model, input_positions, args, head_idx=0, cmap=None, timestamp=None, model_name="comparison"):
    """
    Compare two different RoPE models (Axial and Mixed) side by side.
    
    Args:
        axial_model: The RoPEAxial model to visualize
        mixed_model: The RoPEMixed model to visualize
        input_positions (torch.Tensor): Input position grid for visualization
        args: Command-line arguments containing visualization parameters
        head_idx (int): Head index to use for Mixed model
        cmap: Colormap to use for visualization
        timestamp (str): Timestamp for output filename
        model_name (str): Base name for output file
        
    Returns:
        str: Path to the saved visualization file
    """
    # Apply RoPE and FFT for both models
    axial_freqs, axial_recon = apply_rope_and_fft(input_positions, axial_model)
    mixed_freqs, mixed_recon = apply_rope_and_fft(input_positions, mixed_model, head_idx=head_idx)
    
    # Create a figure for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Input positions (shared for both models)
    axes[0, 0].imshow(input_positions.numpy(), cmap='viridis')
    axes[0, 0].set_title('Input Positions')
    axes[0, 0].axis('off')
    
    # Axial model results
    im_axial_freq = axes[0, 1].imshow(axial_freqs, cmap=cmap)
    axes[0, 1].set_title(f'Axial Frequencies')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(axial_recon, cmap='viridis')
    axes[0, 2].set_title('Axial Reconstruction')
    axes[0, 2].axis('off')
    
    # Mixed model results
    axes[1, 0].imshow(input_positions.numpy(), cmap='viridis')
    axes[1, 0].set_title('Input Positions')
    axes[1, 0].axis('off')
    
    im_mixed_freq = axes[1, 1].imshow(mixed_freqs, cmap=cmap)
    head_title = f'Mixed Head {head_idx}' if hasattr(mixed_model, 'num_heads') and mixed_model.num_heads > 1 else 'Mixed'
    axes[1, 1].set_title(f'{head_title} Frequencies')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mixed_recon, cmap='viridis')
    axes[1, 2].set_title('Mixed Reconstruction')
    axes[1, 2].axis('off')
    
    # Add colorbars
    fig.colorbar(im_axial_freq, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im_mixed_freq, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Fine-tune layout
    plt.tight_layout()
    
    # Add figure title
    pattern_name = "Custom" if not hasattr(input_positions, 'pattern') else input_positions.pattern.capitalize()
    fig.suptitle(f"RoPE Model Comparison - Pattern: {pattern_name}", fontsize=16, y=1.02)
    
    # Save the figure
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    head_suffix = f"_head{head_idx}" if hasattr(mixed_model, 'num_heads') and mixed_model.num_heads > 1 else ""
    output_path = f"{args.output_dir}/rope_comparison_{model_name}_{pattern_name}{head_suffix}_{timestamp}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved model comparison to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path

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
    
    # Handle different model loading scenarios
    model_name = "default"
    
    # Check if explicit axial and mixed models are provided
    if args.axial_model and args.mixed_model:
        print(f"Loading Axial model from: {args.axial_model}")
        print(f"Loading Mixed model from: {args.mixed_model}")
        
        rope_axial = load_trained_model(
            model_path=args.axial_model,
            model_config='rope-axial',
            num_heads=args.num_heads,
            dim=args.dim,
            theta_axial=args.theta_axial,
            theta_mixed=args.theta_mixed,
            device=device
        )
        
        rope_mixed = load_trained_model(
            model_path=args.mixed_model,
            model_config='rope-mixed',
            num_heads=args.num_heads,
            dim=args.dim,
            theta_axial=args.theta_axial,
            theta_mixed=args.theta_mixed,
            device=device
        )
        
        if rope_axial is None or rope_mixed is None:
            print("One or both models failed to load. Exiting without visualization.")
            return
        
        model_name = "comparison"
        use_both_encoders = True
    
    # Handle single model loading
    elif args.load_model and args.model_path:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        
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

    # Add new argument to indicate if we should use the comparison function
    if hasattr(args, 'compare_models') and args.compare_models and use_both_encoders:
        # Use the new comparison function for each pattern
        for pattern in args.patterns:
            # Handle custom pattern
            if pattern == 'custom' and args.custom_pattern:
                custom_coords = args.custom_pattern.split(',')
                input_positions = create_input_positions(pattern, args.grid_size, custom_coords)
            else:
                input_positions = create_input_positions(pattern, args.grid_size)
            
            # Store the pattern name for reference
            input_positions.pattern = pattern
            
            # For Mixed, compare for each specified head
            for head_idx in args.head_indices:
                if hasattr(rope_mixed, 'num_heads') and head_idx >= rope_mixed.num_heads:
                    continue
                
                compare_rope_models(
                    axial_model=rope_axial,
                    mixed_model=rope_mixed,
                    input_positions=input_positions,
                    args=args,
                    head_idx=head_idx,
                    cmap=cmap,
                    timestamp=timestamp,
                    model_name=model_name
                )
        
        # If comparison mode is active and both models are available, return after comparison
        if rope_axial is not None and rope_mixed is not None:
            return

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
    output_path = f"{args.output_dir}/rope_theta_comparison_{model_name}_{timestamp}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved theta comparison to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    args = get_args()
    visualize_rope_frequencies(args) 