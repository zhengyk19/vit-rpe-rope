#!/usr/bin/env python
"""
Positional Encoding Similarity Visualizer

This script visualizes the similarity matrices for different positional encoding methods:
- Absolute positional encoding
- Relative positional encoding
- Polynomial relative positional encoding

It can extract positional embeddings from trained models or visualize them with default parameters.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import seaborn as sns
from models.positional_encoding import (
    AbsolutePositionalEncoding,
    RelativePositionalEncoding,
    PolynomialRPE
)
from models.vit import VisionTransformer

def get_args():
    """Parse command-line arguments for the visualization tool."""
    parser = argparse.ArgumentParser(description='Positional Encoding Similarity Visualizer')
    
    # Basic visualization settings
    parser.add_argument('--grid_size', type=int, default=14, 
                        help='Size of grid for visualization (default: 14)')
    parser.add_argument('--dim', type=int, default=192, 
                        help='Embedding dimension (default: 192)')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of attention heads (default: 6)')
    
    # Positional encoding methods
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['absolute', 'relative', 'polynomial'],
                        help='Positional encoding methods to visualize')
    
    # Polynomial RPE specific arguments
    parser.add_argument('--poly_degree', type=int, default=3,
                        help='Degree for polynomial RPE (default: 3)')
    parser.add_argument('--poly_shared_heads', type=bool, default=True,
                        help='Whether to share polynomial coefficients across heads (default: True)')
    
    # Model loading
    parser.add_argument('--load_model', action='store_true',
                        help='Load a trained model to visualize its learned positional encodings')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_config', type=str, default='absolute',
                        choices=['absolute', 'relative', 'polynomial'],
                        help='Positional encoding method of the model')
    
    # Comparison mode
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare multiple models side by side')
    parser.add_argument('--model_paths', type=str, nargs='+', default=None,
                        help='List of model paths to compare')
    parser.add_argument('--model_configs', type=str, nargs='+', default=None,
                        help='List of model configs corresponding to model_paths')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='Custom names for models in comparison (optional)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--cmap', type=str, default='viridis',
                        choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'],
                        help='Colormap for visualization')
    
    return parser.parse_args()

def load_trained_model(model_path, model_config, num_heads=6, dim=192, device='cpu'):
    """
    Load a trained model to extract its positional encoding.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_config (str): Positional encoding method
        num_heads (int): Number of attention heads
        dim (int): Embedding dimension
        device (str): Device to load the model on
        
    Returns:
        nn.Module: Positional encoding module from the trained model, or None if loading fails
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract dimension from the checkpoint
        model_dim = None
        
        # Look for embedding dimension in checkpoint
        for key, value in checkpoint.items():
            if key == 'patch_embed.weight':
                model_dim = value.shape[0]
            elif key.endswith('.qkv.weight'):
                model_dim = value.shape[1]  # embed_dim
            elif key.endswith('.qkv.bias'):
                model_dim = value.shape[0] // 3
            
            if model_dim is not None:
                break
        
        if model_dim is None:
            print("Warning: Could not determine model dimensions from checkpoint")
            model_dim = dim
        
        print(f"Creating model with dim={model_dim}, num_heads={num_heads}")
        
        # Create a temporary model with the same config to load the state dict
        model = VisionTransformer(
            pos_encoding=model_config,
            embed_dim=model_dim,
            num_heads=num_heads
        ).to(device)
        
        # Load the state dictionary
        model.load_state_dict(checkpoint)
        
        # Extract the positional encoding
        encoder = model.pos_embed
        return encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_absolute_similarity(encoder, grid_size):
    """
    Calculate similarity matrix for absolute positional encoding.
    
    Args:
        encoder (AbsolutePositionalEncoding): The absolute positional encoding module
        grid_size (int): Size of the grid
        
    Returns:
        np.ndarray: Similarity matrix of shape [grid_size*grid_size, grid_size*grid_size]
    """
    seq_len = grid_size * grid_size
    
    # Get the dimension from the pos_embed parameter instead of accessing 'dim' attribute
    embed_dim = encoder.pos_embed.shape[-1]
    
    # Generate a sequence of positions - with zeros as placeholders
    positions = torch.zeros(1, seq_len + 1, embed_dim)  # +1 for class token
    
    # Apply positional encoding - this adds position embeddings to positions 1: (class token doesn't get position)
    positions = encoder(positions)
    
    # Remove class token and reshape
    pos_embeddings = positions[0, 1:, :]  # [seq_len, dim]
    
    # Calculate cosine similarity
    norms = torch.norm(pos_embeddings, dim=1, keepdim=True)
    normalized = pos_embeddings / norms
    similarity = torch.matmul(normalized, normalized.transpose(0, 1))
    
    return similarity.detach().cpu().numpy()

def get_relative_similarity(encoder, grid_size):
    """
    Calculate similarity matrix for relative positional encoding.
    
    Args:
        encoder (RelativePositionalEncoding): The relative positional encoding module
        grid_size (int): Size of the grid
        
    Returns:
        np.ndarray: Similarity matrix of shape [grid_size*grid_size, grid_size*grid_size]
    """
    # The relative PE directly provides the similarity/bias matrix
    similarity = encoder.get_bias()
    
    # Average across heads if multiple heads
    if len(similarity.shape) == 3:  # [num_heads, seq_len, seq_len]
        similarity = similarity.mean(dim=0)
    
    # Remove class token if present
    seq_len = grid_size * grid_size
    if similarity.shape[0] > seq_len:
        similarity = similarity[1:seq_len+1, 1:seq_len+1]
    
    # Make the result more like a cosine similarity (-1 to 1 range)
    # Normalize to range [-1, 1] for consistent visualization
    similarity = similarity.float()  # Ensure floating point
    if similarity.abs().max() > 0:
        similarity = similarity / similarity.abs().max()
    
    return similarity.detach().cpu().numpy()

def get_polynomial_similarity(encoder, grid_size):
    """
    Calculate similarity matrix for polynomial relative positional encoding.
    
    Args:
        encoder (PolynomialRPE): The polynomial relative positional encoding module
        grid_size (int): Size of the grid
        
    Returns:
        np.ndarray: Similarity matrix of shape [grid_size*grid_size, grid_size*grid_size]
    """
    # The polynomial RPE directly provides the similarity/bias matrix
    similarity = encoder.get_bias()
    
    # Average across heads if multiple heads
    if len(similarity.shape) == 3:  # [num_heads, seq_len, seq_len]
        similarity = similarity.mean(dim=0)
    
    # Remove class token if present
    seq_len = grid_size * grid_size
    if similarity.shape[0] > seq_len:
        # Extract only the patch positions (exclude class token)
        similarity = similarity[1:seq_len+1, 1:seq_len+1]
    
    # Make the result more like a cosine similarity (-1 to 1 range)
    # Normalize to range [-1, 1] for consistent visualization
    similarity = similarity.float()  # Ensure floating point
    if similarity.abs().max() > 0:
        similarity = similarity / similarity.abs().max()
    
    return similarity.detach().cpu().numpy()

def visualize_similarity_matrix(similarity, method, grid_size, model_name, args):
    """
    Visualize a similarity matrix for a positional encoding method.
    
    Args:
        similarity (np.ndarray): Similarity matrix
        method (str): Positional encoding method name
        grid_size (int): Size of the grid
        model_name (str): Name of the model for output filename
        args: Command-line arguments
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap
    ax = sns.heatmap(similarity, cmap=args.cmap, vmin=-1, vmax=1, 
                     xticklabels=range(1, grid_size+1), 
                     yticklabels=range(1, grid_size+1))
    
    # Customize ticks
    ax.set_xticks(np.arange(0, grid_size**2, grid_size))
    ax.set_yticks(np.arange(0, grid_size**2, grid_size))
    ax.set_xticklabels(range(1, grid_size+1))
    ax.set_yticklabels(range(1, grid_size+1))
    
    # Add labels
    plt.xlabel("Input patch column")
    plt.ylabel("Input patch row")
    
    # Add title
    method_name = method.capitalize()
    if args.load_model:
        plt.title(f"{method_name} Position Embedding\nFrom model: {model_name}")
    else:
        plt.title(f"{method_name} Position Embedding\nDefault parameters")
    
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Cosine similarity")
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{args.output_dir}/pe_similarity_{model_name}_{method}_{timestamp}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {method} similarity visualization to {output_path}")

def visualize_positional_encodings(args):
    """
    Create visualizations of positional encoding similarity matrices.
    
    Args:
        args: Command-line arguments containing visualization parameters
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get model basename for output filename
    if args.load_model and args.model_path:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    else:
        model_name = "default"
    
    # Set up positional encodings
    encoders = {}
    
    # Load model if requested
    if args.load_model and args.model_path:
        encoder = load_trained_model(
            model_path=args.model_path,
            model_config=args.model_config,
            num_heads=args.num_heads,
            dim=args.dim,
            device=device
        )
        
        if encoder is None:
            print("Model loading failed. Exiting without visualization.")
            return
        
        print(f"Successfully loaded positional encoding from model: {args.model_path}")
        encoders[args.model_config] = encoder
    else:
        # Initialize default encoders for requested methods
        print("Using default encoders")
        seq_len = args.grid_size * args.grid_size
        
        for method in args.methods:
            if method == 'absolute':
                encoders[method] = AbsolutePositionalEncoding(args.dim).to(device)
            elif method == 'relative':
                encoders[method] = RelativePositionalEncoding(seq_len, args.num_heads).to(device)
            elif method == 'polynomial':
                encoders[method] = PolynomialRPE(seq_len, degree=args.poly_degree,
                                               num_heads=args.num_heads, 
                                               shared_across_heads=args.poly_shared_heads).to(device)
    
    # Generate and visualize similarity matrices
    for method, encoder in encoders.items():
        # Calculate similarity matrix
        if method == 'absolute' or isinstance(encoder, AbsolutePositionalEncoding):
            similarity = get_absolute_similarity(encoder, args.grid_size)
        elif method == 'relative' or isinstance(encoder, RelativePositionalEncoding):
            similarity = get_relative_similarity(encoder, args.grid_size)
        elif method == 'polynomial' or isinstance(encoder, PolynomialRPE):
            similarity = get_polynomial_similarity(encoder, args.grid_size)
        else:
            print(f"Unsupported method: {method}")
            continue
        
        # Check that the similarity matrix has the expected shape
        expected_size = args.grid_size * args.grid_size
        if similarity.shape[0] != expected_size or similarity.shape[1] != expected_size:
            print(f"Warning: Similarity matrix has shape {similarity.shape}, expected ({expected_size}, {expected_size})")
            print(f"Adjusting for visualization...")
            
            # Try to extract the correct size matrix from any larger matrix
            # (e.g., if it includes class token or padding)
            if similarity.shape[0] > expected_size and similarity.shape[1] > expected_size:
                # Try to extract the main NxN grid, skipping any class token or padding
                similarity = similarity[:expected_size, :expected_size]
            else:
                # If the matrix is too small, pad it
                print(f"Cannot resize similarity matrix from {similarity.shape} to ({expected_size}, {expected_size})")
                print(f"Skipping visualization for {method}.")
                continue
        
        # Reshape to 2D grid format (like in the image)
        try:
            similarity_grid = similarity.reshape(args.grid_size, args.grid_size, 
                                               args.grid_size, args.grid_size)
        except ValueError as e:
            print(f"Error reshaping similarity matrix: {e}")
            print(f"Similarity matrix shape: {similarity.shape}, expected size: {expected_size}x{expected_size}")
            print(f"Skipping visualization for {method}.")
            continue
        
        # Create a figure with a grid of heatmaps, one for each row position
        fig, axes = plt.subplots(args.grid_size, args.grid_size, 
                               figsize=(2*args.grid_size, 2*args.grid_size),
                               sharex=True, sharey=True)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Create a heatmap for each position
        for pos in range(args.grid_size * args.grid_size):
            i, j = pos // args.grid_size, pos % args.grid_size
            pos_similarity = similarity_grid[i, j].reshape(args.grid_size, args.grid_size)
            
            ax = axes[pos]
            im = ax.imshow(pos_similarity, cmap='viridis', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add thin borders
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.5)
            
            # Add row/column labels on the edges
            if j == 0:  # leftmost column
                ax.set_ylabel(f"{i+1}", fontsize=8)
            if i == args.grid_size - 1:  # bottom row
                ax.set_xlabel(f"{j+1}", fontsize=8)
        
        # Add a single colorbar for the entire figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Cosine similarity")
        
        # Add title
        method_name = method.capitalize()
        if args.load_model:
            title = f"{method_name} Position Embeddings - From model: {model_name}"
        else:
            title = f"{method_name} Position Embeddings - Default parameters"
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)
        
        # Add axis labels
        fig.text(0.5, 0.01, "Input patch column", ha='center', fontsize=12)
        fig.text(0.01, 0.5, "Input patch row", va='center', rotation='vertical', fontsize=12)
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{args.output_dir}/pe_similarity_grid_{model_name}_{method}_{timestamp}.png"
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {method} similarity visualization to {output_path}")
        
        # Create a simplified visualization similar to the image in the reference
        # This is a more compact version that shows the full similarity matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Reshape similarity matrix to have all positions in a single matrix
        # We'll rearrange so each block represents all patch positions
        block_matrix = np.zeros((args.grid_size**2, args.grid_size**2))
        
        for i in range(args.grid_size):
            for j in range(args.grid_size):
                for k in range(args.grid_size):
                    for l in range(args.grid_size):
                        # Position (i,j)'s similarity to position (k,l)
                        row_idx = i * args.grid_size + j
                        col_idx = k * args.grid_size + l
                        block_matrix[row_idx, col_idx] = similarity_grid[i, j, k, l]
        
        # Create heatmap
        im = ax.imshow(block_matrix, cmap='viridis', vmin=-1, vmax=1)
        
        # Add grid lines to separate rows and columns
        ax.set_xticks(np.arange(-0.5, args.grid_size**2, args.grid_size), minor=True)
        ax.set_yticks(np.arange(-0.5, args.grid_size**2, args.grid_size), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        
        # Add row and column labels
        ax.set_xticks(np.arange(args.grid_size//2, args.grid_size**2, args.grid_size))
        ax.set_yticks(np.arange(args.grid_size//2, args.grid_size**2, args.grid_size))
        ax.set_xticklabels(range(1, args.grid_size+1))
        ax.set_yticklabels(range(1, args.grid_size+1))
        
        # Add labels
        plt.xlabel("Input patch column")
        plt.ylabel("Input patch row")
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("Cosine similarity")
        
        # Add title with model info
        model_info = f"From model: {model_name}" if args.load_model else "Default parameters"
        plt.title(f"{method_name} Position Embeddings\n{model_info}", fontsize=14)
        
        # Save figure
        output_path = f"{args.output_dir}/pe_similarity_compact_{model_name}_{method}_{timestamp}.png"
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved compact {method} similarity visualization to {output_path}")

def visualize_model_comparison(args):
    """
    Create a side-by-side comparison of different positional encoding models.
    
    Args:
        args: Command-line arguments containing visualization parameters
    """
    if not args.model_paths or not args.model_configs:
        print("Error: Model paths and configs must be provided for comparison")
        return
    
    if len(args.model_paths) != len(args.model_configs):
        print("Error: Number of model paths and configs must match")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load all models and get their similarities
    similarities = []
    model_labels = []
    
    for i, (model_path, model_config) in enumerate(zip(args.model_paths, args.model_configs)):
        # Get model name
        if args.model_names and i < len(args.model_names):
            model_name = args.model_names[i]
        else:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        model_labels.append(model_name)
        
        # Load the model
        encoder = load_trained_model(
            model_path=model_path,
            model_config=model_config,
            num_heads=args.num_heads,
            dim=args.dim,
            device=device
        )
        
        if encoder is None:
            print(f"Failed to load model: {model_path}")
            continue
        
        # Calculate similarity matrix
        if model_config == 'absolute':
            similarity = get_absolute_similarity(encoder, args.grid_size)
        elif model_config == 'relative':
            similarity = get_relative_similarity(encoder, args.grid_size)
        elif model_config == 'polynomial':
            similarity = get_polynomial_similarity(encoder, args.grid_size)
        else:
            print(f"Unsupported method: {model_config}")
            continue
        
        # Check matrix size
        expected_size = args.grid_size * args.grid_size
        if similarity.shape[0] != expected_size or similarity.shape[1] != expected_size:
            print(f"Warning: {model_name} similarity matrix has shape {similarity.shape}, expected ({expected_size}, {expected_size})")
            
            # Try to extract the correct size matrix
            if similarity.shape[0] > expected_size and similarity.shape[1] > expected_size:
                similarity = similarity[:expected_size, :expected_size]
            else:
                print(f"Cannot resize similarity matrix for {model_name}")
                continue
        
        try:
            # Reshape for visualization
            similarity_grid = similarity.reshape(args.grid_size, args.grid_size, 
                                               args.grid_size, args.grid_size)
            similarities.append(similarity_grid)
        except ValueError as e:
            print(f"Error reshaping similarity matrix for {model_name}: {e}")
            continue
    
    if not similarities:
        print("No valid models to compare")
        return
    
    # Create comparison figure
    num_models = len(similarities)
    
    # Create a compact comparison of all models
    fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
    
    if num_models == 1:
        axes = [axes]  # Make into a list for consistent indexing
    
    # Create block matrices for each model
    for i, (similarity_grid, model_name) in enumerate(zip(similarities, model_labels)):
        # Create block matrix
        block_matrix = np.zeros((args.grid_size**2, args.grid_size**2))
        
        for a in range(args.grid_size):
            for b in range(args.grid_size):
                for c in range(args.grid_size):
                    for d in range(args.grid_size):
                        row_idx = a * args.grid_size + b
                        col_idx = c * args.grid_size + d
                        block_matrix[row_idx, col_idx] = similarity_grid[a, b, c, d]
        
        # Create heatmap
        im = axes[i].imshow(block_matrix, cmap=args.cmap, vmin=-1, vmax=1)
        
        # Add grid lines to separate rows and columns
        axes[i].set_xticks(np.arange(-0.5, args.grid_size**2, args.grid_size), minor=True)
        axes[i].set_yticks(np.arange(-0.5, args.grid_size**2, args.grid_size), minor=True)
        axes[i].grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        
        # Add row and column labels
        axes[i].set_xticks(np.arange(args.grid_size//2, args.grid_size**2, args.grid_size))
        axes[i].set_yticks(np.arange(args.grid_size//2, args.grid_size**2, args.grid_size))
        axes[i].set_xticklabels(range(1, args.grid_size+1))
        axes[i].set_yticklabels(range(1, args.grid_size+1))
        
        # Add title
        axes[i].set_title(model_name)
        
        # Add labels for first model only
        if i == 0:
            axes[i].set_ylabel("Input patch row")
    
    # Add common xlabel
    fig.text(0.5, 0.01, "Input patch column", ha='center')
    
    # Add a single colorbar for all plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cosine similarity")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.suptitle("Positional Encoding Comparison", fontsize=16)
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{args.output_dir}/pe_model_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison to {output_path}")

def main():
    """Main function to run the visualization script."""
    args = get_args()
    
    if args.compare_models:
        visualize_model_comparison(args)
    else:
        visualize_positional_encodings(args)

if __name__ == "__main__":
    main() 