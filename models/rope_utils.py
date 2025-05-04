import torch

def apply_rotary_emb(q, k, cos, sin):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Uses only real arithmetic (no complex tensors) with the classic "rotate-half" trick.
    
    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        cos: Cosine component of shape compatible with q/k
        sin: Sine component of shape compatible with q/k
        
    Returns:
        tuple: Rotated (q, k) tensors
    """
    B, H, N, D = q.shape
    D2 = D // 2
    
    # Split into halves
    q1, q2 = q[..., :D2], q[..., D2:]
    k1, k2 = k[..., :D2], k[..., D2:]
    
    # Apply rotation using real arithmetic:
    # (q1 + i q2) * (cos + i sin) = (q1*cos - q2*sin) + i(q1*sin + q2*cos)
    q_rot = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)
    
    k_rot = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)
    
    return q_rot, k_rot

def reshape_for_broadcast(x, target_tensor):
    """
    Reshape frequency tensor for broadcasting with attention tensor.
    
    Args:
        x: Frequency tensor (cos or sin)
        target_tensor: Tensor to broadcast against
        
    Returns:
        Reshaped tensor suitable for broadcasting
    """
    # Expected shapes:
    # x: [seq_len, dim/2] or [num_heads, seq_len, dim/2]
    # target: [batch, heads, seq_len, dim]
    
    ndim = target_tensor.ndim
    
    # Check if x already has head dimension
    if x.ndim == 3 and target_tensor.ndim == 4:
        # x shape is [num_heads, seq_len, dim/2]
        # Need to reshape to [1, num_heads, seq_len, dim/2]
        return x.unsqueeze(0)
    elif x.ndim == 2 and target_tensor.ndim == 4:
        # x shape is [seq_len, dim/2]
        # Need to reshape to [1, 1, seq_len, dim/2]
        return x.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected tensor shapes: {x.shape} vs {target_tensor.shape}") 