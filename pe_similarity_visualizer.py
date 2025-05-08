#!/usr/bin/env python
"""
Positional Encoding Similarity Visualizer

This tool analyzes and visualizes similarity matrices for different positional encoding methods
used in Vision Transformers:
- Absolute positional encoding
- Relative positional encoding
- Polynomial relative positional encoding
- RoPE-Axial positional encoding
- RoPE-Mixed positional encoding

The visualizer provides insights into how different positional encoding strategies represent
spatial relationships between input patches, which is critical for vision transformers to 
understand image structure.

Features:
- Extract positional embeddings from trained models
- Compare multiple encoding approaches side-by-side
- Create detailed similarity visualizations across grid positions
- Support for various model architectures and datasets
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import seaborn as sns
from models.rope_utils import apply_rotary_emb, reshape_for_broadcast
from models.vit import VisionTransformer

def get_args():
    """
    Parse and configure command-line arguments for the visualization tool.
    
    Defines various configuration options including grid size, embedding dimensions,
    visualization methods, model loading parameters, and output settings.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Positional Encoding Similarity Visualizer')
    
    # Visualization configuration
    parser.add_argument('--grid_size', type=int, default=14, 
                        help='Size of grid for visualization (default: 14)')
    parser.add_argument('--dim', type=int, default=192, 
                        help='Embedding dimension (default: 192)')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of attention heads (default: 6)')
    
    # Encoding method selection
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['absolute', 'relative', 'polynomial', 'rope-axial', 'rope-mixed'],
                        help='Positional encoding methods to visualize')
    
    # Polynomial RPE parameters
    parser.add_argument('--poly_degree', type=int, default=3,
                        help='Degree for polynomial RPE (default: 3)')
    parser.add_argument('--poly_shared_heads', type=bool, default=True,
                        help='Whether to share polynomial coefficients across heads (default: True)')
    
    # RoPE configuration
    parser.add_argument('--rope_theta', type=float, default=100.0,
                        help='Theta parameter for RoPE variants (default: 100.0)')
    parser.add_argument('--rope_head_idx', type=int, default=0,
                        help='Head index to visualize for RoPE-Mixed (default: 0)')
    
    # Model loading parameters
    parser.add_argument('--load_model', action='store_true',
                        help='Load a trained model to visualize its learned positional encodings')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_config', type=str, default='absolute',
                        choices=['absolute', 'relative', 'polynomial', 'rope-axial', 'rope-mixed'],
                        help='Positional encoding method of the model')
    
    # Model comparison settings
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare multiple models side by side')
    parser.add_argument('--model_paths', type=str, nargs='+', default=None,
                        help='List of model paths to compare')
    parser.add_argument('--model_configs', type=str, nargs='+', default=None,
                        help='List of model configs corresponding to model_paths')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='Custom names for models in comparison (optional)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--cmap', type=str, default='viridis',
                        choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'],
                        help='Colormap for visualization')
    
    return parser.parse_args()

def load_trained_model(model_path, model_config, num_heads=6, dim=192, device='cpu', rope_theta=None):
    """
    Load a trained ViT model and extract its positional encoding module.
    
    This function loads a checkpoint from the specified path, determines the model's 
    architecture parameters, and extracts the positional encoding component for analysis.
    It handles different encoding types and automatically adapts to the model dimensions.
    
    Args:
        model_path (str): Path to the model checkpoint file
        model_config (str): Type of positional encoding used ('absolute', 'relative', 
                           'polynomial', 'rope-axial', or 'rope-mixed')
        num_heads (int): Number of attention heads in the model
        dim (int): Embedding dimension, used as fallback if not detected from checkpoint
        device (str): Device to load the model on ('cpu' or 'cuda')
        rope_theta (float, optional): Theta parameter for RoPE variants. If None, 
                                     uses appropriate defaults based on encoding type
        
    Returns:
        nn.Module: Positional encoding module from the trained model, or None if loading fails
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # Set default rope_theta based on model type if not provided
    if rope_theta is None:
        if model_config == 'rope-axial':
            rope_theta = 100.0
            print(f"Using default rope_theta={rope_theta} for rope-axial")
        elif model_config == 'rope-mixed':
            rope_theta = 10.0
            print(f"Using default rope_theta={rope_theta} for rope-mixed")
        else:
            rope_theta = 100.0  # Default value for other models
    
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
            num_heads=num_heads,
            rope_theta=rope_theta
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
    
    # If the shape doesn't match expected size, we might need to adjust
    if pos_embeddings.shape[0] != seq_len:
        actual_grid_size = int(np.sqrt(pos_embeddings.shape[0]))
        if actual_grid_size**2 == pos_embeddings.shape[0]:
            print(f"Adjusting grid size from {grid_size} to {actual_grid_size} based on model shape")
            grid_size = actual_grid_size
            seq_len = grid_size * grid_size
            # Use only the needed part of the embedding
            pos_embeddings = pos_embeddings[:seq_len, :]
    
    # Calculate cosine similarity
    norms = torch.norm(pos_embeddings, dim=1, keepdim=True)
    normalized = pos_embeddings / norms
    similarity = torch.matmul(normalized, normalized.transpose(0, 1))
    
    return similarity.detach().cpu().numpy(), grid_size

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
    
    # Detect actual grid size from similarity
    if similarity.shape[0] > 1:  # Has at least some elements
        actual_seq_len = similarity.shape[0] - 1  # -1 for class token
        actual_grid_size = int(np.sqrt(actual_seq_len))
        if actual_grid_size**2 == actual_seq_len:
            print(f"Detected grid size {actual_grid_size} from model shape")
            grid_size = actual_grid_size
    
    # Remove class token if present
    seq_len = grid_size * grid_size
    if similarity.shape[0] > seq_len:
        similarity = similarity[1:seq_len+1, 1:seq_len+1]
    
    # Make the result more like a cosine similarity (-1 to 1 range)
    # Normalize to range [-1, 1] for consistent visualization
    similarity = similarity.float()  # Ensure floating point
    if similarity.abs().max() > 0:
        similarity = similarity / similarity.abs().max()
    
    return similarity.detach().cpu().numpy(), grid_size

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
    
    # Detect actual grid size from similarity
    if similarity.shape[0] > 1:  # Has at least some elements
        actual_seq_len = similarity.shape[0] - 1  # -1 for class token
        actual_grid_size = int(np.sqrt(actual_seq_len))
        if actual_grid_size**2 == actual_seq_len:
            print(f"Detected grid size {actual_grid_size} from model shape")
            grid_size = actual_grid_size
    
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
    
    return similarity.detach().cpu().numpy(), grid_size

def get_rope_axial_similarity(encoder, grid_size, dim=None):
    """
    Calculate similarity matrix for RoPE-Axial positional encoding.
    
    Args:
        encoder (RoPEAxial): The RoPE-Axial positional encoding module
        grid_size (int): Size of the grid
        dim (int, optional): Embedding dimension
        
    Returns:
        np.ndarray: Similarity matrix of shape [grid_size*grid_size, grid_size*grid_size]
    """
    # For CIFAR-10 models, force grid_size to 8
    if grid_size > 8:
        print(f"Adjusting grid size from {grid_size} to 8 for CIFAR-10 model")
        grid_size = 8
    
    seq_len = grid_size * grid_size
    
    # Get device safely - check if encoder has parameters first
    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        # If no parameters, use default device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the encoder's dimension if not provided
    if dim is None:
        dim = encoder.dim
    
    # Generate random query vectors for all positions
    # We'll use the same random vectors for all positions to isolate position effects
    q_base = torch.ones(1, 1, 1, dim, device=device)
    q_base = q_base.expand(1, 1, seq_len, -1)  # [1, 1, seq_len, dim]
    k_base = q_base.clone()  # Use same base for keys
    
    # Get rotary embeddings for this grid size
    cos, sin = encoder.get_freqs_cis(seq_len, device)
    
    # Reshape for broadcasting
    cos = reshape_for_broadcast(cos, q_base)
    sin = reshape_for_broadcast(sin, q_base)
    
    # Apply rotary embeddings to get position-specific queries and keys
    q_rot, k_rot = apply_rotary_emb(q_base, k_base, cos, sin)
    
    # Calculate attention scores (dot product similarity)
    # Remove batch and head dimensions
    q_rot = q_rot.squeeze(0).squeeze(0)  # [seq_len, dim]
    k_rot = k_rot.squeeze(0).squeeze(0)  # [seq_len, dim]
    
    # Calculate cosine similarity
    q_norm = torch.norm(q_rot, dim=1, keepdim=True)
    k_norm = torch.norm(k_rot, dim=1, keepdim=True)
    
    q_normalized = q_rot / q_norm
    k_normalized = k_rot / k_norm
    
    similarity = torch.matmul(q_normalized, k_normalized.transpose(0, 1))
    
    return similarity.detach().cpu().numpy(), grid_size

def get_rope_mixed_similarity(encoder, grid_size, head_idx=0, dim=None):
    """
    Calculate similarity matrix for RoPE-Mixed positional encoding.
    
    Args:
        encoder (RoPEMixed): The RoPE-Mixed positional encoding module
        grid_size (int): Size of the grid
        head_idx (int): Head index to visualize
        dim (int, optional): Embedding dimension
        
    Returns:
        np.ndarray: Similarity matrix of shape [grid_size*grid_size, grid_size*grid_size]
    """
    # For CIFAR-10 models, force grid_size to 8
    if grid_size > 8:
        print(f"Adjusting grid size from {grid_size} to 8 for CIFAR-10 model")
        grid_size = 8
    
    seq_len = grid_size * grid_size
    
    # Get device safely - check if encoder has parameters first
    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        # If no parameters, use default device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the encoder's dimension if not provided
    if dim is None:
        dim = encoder.dim
    
    # Generate random query vectors for all positions
    # We'll use the same random vectors for all positions to isolate position effects
    q_base = torch.ones(1, encoder.num_heads, 1, dim, device=device)
    q_base = q_base.expand(1, encoder.num_heads, seq_len, -1)  # [1, num_heads, seq_len, dim]
    k_base = q_base.clone()  # Use same base for keys
    
    # Get rotary embeddings for this grid size
    cos, sin = encoder.get_freqs_cis(seq_len, device)
    
    # Reshape for broadcasting
    cos = reshape_for_broadcast(cos, q_base)
    sin = reshape_for_broadcast(sin, q_base)
    
    # Apply rotary embeddings to get position-specific queries and keys
    q_rot, k_rot = apply_rotary_emb(q_base, k_base, cos, sin)
    
    # Extract the requested head
    q_rot = q_rot[0, head_idx]  # [seq_len, dim]
    k_rot = k_rot[0, head_idx]  # [seq_len, dim]
    
    # Calculate cosine similarity
    q_norm = torch.norm(q_rot, dim=1, keepdim=True)
    k_norm = torch.norm(k_rot, dim=1, keepdim=True)
    
    q_normalized = q_rot / q_norm
    k_normalized = k_rot / k_norm
    
    similarity = torch.matmul(q_normalized, k_normalized.transpose(0, 1))
    
    return similarity.detach().cpu().numpy(), grid_size

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
    
    # Normalize data to use full colormap range
    vmin = -1
    vmax = 1
    
    # Create a heatmap
    ax = sns.heatmap(similarity, cmap=args.cmap, vmin=vmin, vmax=vmax, 
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
    Generate visualizations of positional encoding similarity matrices from a trained model.
    
    This function loads a model's positional encoding module, computes similarity matrices
    based on the encoding type, and creates multiple visualization formats:
    1. A detailed grid showing position-specific similarity patterns
    2. A compact representation of the full similarity matrix
    
    The function automatically adapts to different model architectures and grid sizes,
    adjusting parameters for specific datasets like CIFAR-10.
    
    Args:
        args: Command-line arguments containing visualization parameters,
              including model path, encoding type, and output settings
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get model basename for output filename
    if args.model_path:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    else:
        print("Error: No model path specified. Exiting without visualization.")
        return
    
    # Set grid size to 8 for CIFAR-10 models
    if 'cifar10' in model_name.lower():
        print(f"Setting grid size to 8 for CIFAR-10 model: {model_name}")
        grid_size = 8
    else:
        grid_size = args.grid_size
    
    # Set appropriate rope_theta based on model type
    rope_theta = None  # Use default in load_trained_model
    if args.model_config == 'rope-axial':
        rope_theta = 100.0
        print(f"Using rope_theta={rope_theta} for {model_name} (rope-axial)")
    elif args.model_config == 'rope-mixed':
        rope_theta = 10.0
        print(f"Using rope_theta={rope_theta} for {model_name} (rope-mixed)")
    elif args.rope_theta is not None:
        rope_theta = args.rope_theta
        print(f"Using user-specified rope_theta={rope_theta} for {model_name}")
    
    # Load the model
    encoder = load_trained_model(
        model_path=args.model_path,
        model_config=args.model_config,
        num_heads=args.num_heads,
        dim=args.dim,
        device=device,
        rope_theta=rope_theta
    )
    
    if encoder is None:
        print("Model loading failed. Exiting without visualization.")
        return
    
    print(f"Successfully loaded positional encoding from model: {args.model_path}")
    
    # Calculate similarity matrix
    method = args.model_config
    if method == 'absolute':
        similarity, grid_size = get_absolute_similarity(encoder, grid_size)
    elif method == 'relative':
        similarity, grid_size = get_relative_similarity(encoder, grid_size)
    elif method == 'polynomial':
        similarity, grid_size = get_polynomial_similarity(encoder, grid_size)
    elif method == 'rope-axial':
        similarity, grid_size = get_rope_axial_similarity(encoder, grid_size)
    elif method == 'rope-mixed':
        similarity, grid_size = get_rope_mixed_similarity(encoder, grid_size, head_idx=args.rope_head_idx)
    else:
        print(f"Unsupported method: {method}")
        return
    
    # Check that the similarity matrix has the expected shape
    expected_size = grid_size * grid_size
    if similarity.shape[0] != expected_size or similarity.shape[1] != expected_size:
        print(f"Warning: Similarity matrix has shape {similarity.shape}, expected ({expected_size}, {expected_size})")
        print(f"Skipping visualization for {method}.")
        return
    
    # Create a figure with a grid of heatmaps, one for each row position
    try:
        # Reshape for visualization
        similarity_grid = similarity.reshape(grid_size, grid_size, grid_size, grid_size)
    except ValueError as e:
        print(f"Error reshaping similarity matrix: {e}")
        print(f"Similarity matrix shape: {similarity.shape}, expected size: {grid_size}x{grid_size}")
        print(f"Skipping visualization for {method}.")
        return
    
    # Create a figure with a grid of heatmaps, one for each row position
    fig, axes = plt.subplots(grid_size, grid_size, 
                           figsize=(2*grid_size, 2*grid_size),
                           sharex=True, sharey=True)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Find global min and max for consistent coloring across all subplots
    vmin = -1
    vmax = 1
    
    # Create a heatmap for each position
    for pos in range(grid_size * grid_size):
        i, j = pos // grid_size, pos % grid_size
        pos_similarity = similarity_grid[i, j].reshape(grid_size, grid_size)
        
        ax = axes[pos]
        im = ax.imshow(pos_similarity, cmap='viridis', vmin=vmin, vmax=vmax)
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
        if i == grid_size - 1:  # bottom row
            ax.set_xlabel(f"{j+1}", fontsize=8)
    
    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cosine similarity")
    
    # Add title
    method_name = method.capitalize()
    title = f"{method_name} Position Embeddings - From model: {model_name} ({grid_size}x{grid_size})"
    
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
    block_matrix = np.zeros((grid_size**2, grid_size**2))
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                for l in range(grid_size):
                    # Position (i,j)'s similarity to position (k,l)
                    row_idx = i * grid_size + j
                    col_idx = k * grid_size + l
                    block_matrix[row_idx, col_idx] = similarity_grid[i, j, k, l]
    
    # Create heatmap
    im = ax.imshow(block_matrix, cmap='viridis', vmin=-1, vmax=1)
    
    # Add grid lines to separate rows and columns
    ax.set_xticks(np.arange(-0.5, grid_size**2, grid_size), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size**2, grid_size), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Add row and column labels
    ax.set_xticks(np.arange(grid_size//2, grid_size**2, grid_size))
    ax.set_yticks(np.arange(grid_size//2, grid_size**2, grid_size))
    ax.set_xticklabels(range(1, grid_size+1))
    ax.set_yticklabels(range(1, grid_size+1))
    
    # Add labels
    plt.xlabel("Input patch column")
    plt.ylabel("Input patch row")
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Cosine similarity")
    
    # Add title with model info
    model_info = f"From model: {model_name}"
    plt.title(f"{method_name} Position Embeddings\n{model_info} ({grid_size}x{grid_size})", fontsize=14)
    
    # Save figure
    output_path = f"{args.output_dir}/pe_similarity_compact_{model_name}_{method}_{timestamp}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved compact {method} similarity visualization to {output_path}")

def visualize_model_comparison(args):
    """
    Create a side-by-side comparison of different positional encoding approaches.
    
    This function loads multiple models with different positional encoding methods,
    computes their similarity matrices, and arranges them for direct comparison.
    The visualization helps identify distinctive patterns and characteristics of
    each encoding approach, highlighting their strengths and weaknesses in capturing
    spatial relationships.
    
    The function handles varying grid sizes and automatically adapts visualization
    parameters to enable fair comparison across different model architectures.
    
    Args:
        args: Command-line arguments containing comparison parameters, including
              model paths, encoding types, and visualization settings
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
    grid_sizes = []
    
    for i, (model_path, model_config) in enumerate(zip(args.model_paths, args.model_configs)):
        # Get model name
        if args.model_names and i < len(args.model_names):
            model_name = args.model_names[i]
        else:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        model_labels.append(model_name)
        
        # Set grid size to 8 for CIFAR-10 models
        if 'cifar10' in model_path.lower():
            print(f"Setting grid size to 8 for CIFAR-10 model: {model_path}")
            grid_size = 8
        else:
            grid_size = args.grid_size
        
        # Set appropriate rope_theta based on model type
        rope_theta = None  # Use default in load_trained_model
        if model_config == 'rope-axial':
            rope_theta = 100.0
            print(f"Using rope_theta={rope_theta} for {model_name} (rope-axial)")
        elif model_config == 'rope-mixed':
            rope_theta = 10.0
            print(f"Using rope_theta={rope_theta} for {model_name} (rope-mixed)")
        elif args.rope_theta is not None:
            rope_theta = args.rope_theta
            print(f"Using user-specified rope_theta={rope_theta} for {model_name}")
        
        # Load the model
        encoder = load_trained_model(
            model_path=model_path,
            model_config=model_config,
            num_heads=args.num_heads,
            dim=args.dim,
            device=device,
            rope_theta=rope_theta
        )
        
        if encoder is None:
            print(f"Failed to load model: {model_path}")
            continue
        
        # Calculate similarity matrix with updated functions that return grid_size
        if model_config == 'absolute':
            similarity, grid_size = get_absolute_similarity(encoder, grid_size)
        elif model_config == 'relative':
            similarity, grid_size = get_relative_similarity(encoder, grid_size)
        elif model_config == 'polynomial':
            similarity, grid_size = get_polynomial_similarity(encoder, grid_size)
        elif model_config == 'rope-axial':
            similarity, grid_size = get_rope_axial_similarity(encoder, grid_size)
        elif model_config == 'rope-mixed':
            similarity, grid_size = get_rope_mixed_similarity(encoder, grid_size, head_idx=args.rope_head_idx)
        else:
            print(f"Unsupported method: {model_config}")
            continue
        
        # Check matrix size
        expected_size = grid_size * grid_size
        if similarity.shape[0] != expected_size or similarity.shape[1] != expected_size:
            print(f"Warning: Matrix shape {similarity.shape} doesn't match expected size ({expected_size}, {expected_size})")
            continue
        
        grid_sizes.append(grid_size)
        
        try:
            # Reshape for visualization
            similarity_grid = similarity.reshape(grid_size, grid_size, grid_size, grid_size)
            similarities.append((similarity_grid, grid_size))
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
    for i, ((similarity_grid, grid_size), model_name) in enumerate(zip(similarities, model_labels)):
        # Create block matrix
        block_matrix = np.zeros((grid_size**2, grid_size**2))
        
        for a in range(grid_size):
            for b in range(grid_size):
                for c in range(grid_size):
                    for d in range(grid_size):
                        row_idx = a * grid_size + b
                        col_idx = c * grid_size + d
                        block_matrix[row_idx, col_idx] = similarity_grid[a, b, c, d]
        
        # Ensure the matrix uses the full colormap range
        vmin = -1
        vmax = 1
        
        # Create heatmap
        im = axes[i].imshow(block_matrix, cmap=args.cmap, vmin=vmin, vmax=vmax)
        
        # Add grid lines to separate rows and columns
        axes[i].set_xticks(np.arange(-0.5, grid_size**2, grid_size), minor=True)
        axes[i].set_yticks(np.arange(-0.5, grid_size**2, grid_size), minor=True)
        axes[i].grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        
        # Add row and column labels
        axes[i].set_xticks(np.arange(grid_size//2, grid_size**2, grid_size))
        axes[i].set_yticks(np.arange(grid_size//2, grid_size**2, grid_size))
        axes[i].set_xticklabels(range(1, grid_size+1))
        axes[i].set_yticklabels(range(1, grid_size+1))
        
        # Add title
        axes[i].set_title(f"{model_name} ({grid_size}x{grid_size})")
        
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
    """
    Main entry point for the positional encoding similarity visualization tool.
    
    This function:
    1. Parses command-line arguments
    2. Handles automatic detection of available model checkpoints if needed
    3. Routes execution to the appropriate visualization mode:
       - Single model visualization
       - Multi-model comparison
    4. Provides helpful error messages if required parameters are missing
    """
    args = get_args()
    
    # If methods are specified but not using load_model, convert to load_model mode
    if not args.load_model and args.methods:
        # Check if we have checkpoints for these methods
        available_checkpoints = {}
        for method in args.methods:
            checkpoint_path = f"checkpoints/cifar10_{method}_best.pth"
            if os.path.exists(checkpoint_path):
                available_checkpoints[method] = checkpoint_path
            else:
                print(f"Warning: No checkpoint found for method '{method}' at {checkpoint_path}")
        
        if available_checkpoints:
            print(f"Found {len(available_checkpoints)} checkpoints. Using them instead of creating new models.")
            # Convert to compare_models mode
            args.compare_models = True
            args.model_paths = list(available_checkpoints.values())
            args.model_configs = list(available_checkpoints.keys())
            args.model_names = [config.capitalize() for config in available_checkpoints.keys()]
        else:
            print("Error: No checkpoints found for the specified methods. Exiting.")
            return
    
    if args.compare_models:
        visualize_model_comparison(args)
    elif args.load_model and args.model_path:
        visualize_positional_encodings(args)
    else:
        print("Error: Please specify either --load_model with --model_path or --compare_models with appropriate parameters.")
        print("\nExample usage:")
        print("  • Visualize a single model:")
        print("    python pe_similarity_visualizer.py --load_model --model_path checkpoints/cifar10_rope-axial_best.pth --model_config rope-axial")
        print("\n  • Auto-detect and compare available models:")
        print("    python pe_similarity_visualizer.py --methods rope-axial rope-mixed")
        print("\n  • Compare specific models:")
        print("    python pe_similarity_visualizer.py --compare_models --model_paths checkpoints/cifar10_absolute_best.pth checkpoints/cifar10_rope-axial_best.pth --model_configs absolute rope-axial")
        return

if __name__ == "__main__":
    main() 

# python pe_similarity_visualizer.py --load_model --model_path checkpoints/cifar10_absolute_best.pth --model_config absolute
# python pe_similarity_visualizer.py --load_model --model_path checkpoints/cifar10_relative_best.pth --model_config relative
# python pe_similarity_visualizer.py --load_model --model_path checkpoints/cifar10_polynomial_best.pth --model_config polynomial
# python pe_similarity_visualizer.py --load_model --model_path checkpoints/cifar10_rope-axial_best.pth --model_config rope-axial
# python pe_similarity_visualizer.py --load_model --model_path checkpoints/cifar10_rope-mixed_best.pth --model_config rope-mixed