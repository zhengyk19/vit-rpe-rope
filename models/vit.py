"""
This module implements the Vision Transformer (ViT) architecture with support for different positional encoding methods.
The implementation includes the core components: Attention mechanism, Transformer blocks, and the main VisionTransformer class.
"""

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import DropPath
from .positional_encoding import AbsolutePositionalEncoding, RelativePositionalEncoding, PolynomialRPE, RoPEAxial, RoPEMixed

def reshape_for_broadcast(freqs_cis, x):
    """
    Reshape frequency tensor for broadcasting with query/key tensors
    
    Args:
        freqs_cis (torch.Tensor): Complex frequency tensor
        x (torch.Tensor): Target tensor to broadcast against
        
    Returns:
        torch.Tensor: Reshaped tensor for broadcasting
    """
    ndim = x.ndim
    if freqs_cis.shape == (x.shape[1]-1, x.shape[-1] // 2):
        # For RoPE-Axial: [seq_len, dim/2] -> [1, seq_len, 1, dim/2]
        # The -1 accounts for the class token which is not rotated
        return freqs_cis.unsqueeze(0).unsqueeze(2)
    elif freqs_cis.shape[0] == x.shape[0]:  # For RoPE-Mixed with num_heads as first dim
        # [num_heads, seq_len, dim/2] -> [num_heads, seq_len, 1, dim/2]
        return freqs_cis.unsqueeze(2)
    else:
        # General case - create broadcast shape based on target tensor dimensions
        # For complex numbers, we need to ensure the last dimension is correct
        return freqs_cis.unsqueeze(0).unsqueeze(2)

def apply_rotary_emb(q, k, freqs_cis):
    """
    Apply rotary embeddings to queries and keys using complex multiplication
    
    Args:
        q (torch.Tensor): Query tensor [B, num_heads, seq_len, head_dim]
        k (torch.Tensor): Key tensor [B, num_heads, seq_len, head_dim]
        freqs_cis (torch.Tensor): Complex rotation tensor
        
    Returns:
        tuple: (rotated_q, rotated_k) with the same shape as inputs
    """
    # Skip the class token (first position)
    q_real = q[:, :, 1:].float()
    k_real = k[:, :, 1:].float()
    
    # Get real shapes for debugging
    B, H, L, D = q_real.shape
    
    # View last dimension as complex numbers (pairs of real/imaginary)
    # Make sure D is even for complex view
    if D % 2 != 0:
        raise ValueError(f"Head dimension ({D}) must be even for complex number representation")
    
    # Reshape for complex multiplication
    q_comp = torch.view_as_complex(q_real.reshape(*q_real.shape[:-1], -1, 2))
    k_comp = torch.view_as_complex(k_real.reshape(*k_real.shape[:-1], -1, 2))
    
    # Make sure freqs_cis is on the same device
    if freqs_cis.device != q.device:
        freqs_cis = freqs_cis.to(q.device)
    
    # Reshape freqs_cis for proper broadcasting
    if freqs_cis.dim() == 2:  # [seq_len, dim/2]
        # For RoPE-Axial
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim/2]
    elif freqs_cis.dim() == 3:  # [num_heads, seq_len, dim/2]
        # For RoPE-Mixed
        freqs_cis = freqs_cis.unsqueeze(2)  # [num_heads, seq_len, 1, dim/2]
    
    # Apply rotation via complex multiplication
    # z' = z * e^(i*θ) rotates complex number z by angle θ
    q_out = torch.view_as_real(q_comp * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_comp * freqs_cis).flatten(3)
    
    # Convert back to original dtype
    q_out = q_out.type_as(q)
    k_out = k_out.type_as(k)
    
    # Combine with class token which remains unchanged
    q_with_cls = torch.cat([q[:, :, :1], q_out], dim=2)
    k_with_cls = torch.cat([k[:, :, :1], k_out], dim=2)
    
    return q_with_cls, k_with_cls

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism for Vision Transformer.
    
    This class implements the scaled dot-product attention mechanism with multiple attention heads.
    It includes linear projections for queries, keys, and values, followed by attention computation
    and a final projection layer.
    
    Different positional encoding methods can be applied:
    - For absolute positional encoding, encoding is added to embeddings before attention.
    - For relative positional encoding methods, bias is added to attention logits.
    - For rotary positional encoding (RoPE-Axial and RoPE-Mixed), queries and keys are rotated 
      before computing dot products.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores

        # Linear projection for queries, keys, and values
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # For tracking which position encoding to use
        self.pos_encoding = None

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        
        # Reshape and permute for multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Apply RoPE if provided - rotates q and k before computing attention
        if (isinstance(self.pos_encoding, RoPEAxial) or isinstance(self.pos_encoding, RoPEMixed)) and freqs_cis is not None:
            # Apply rotation to queries and keys
            q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
            
            # Compute attention scores with rotated q and k
            attn = (q_rot @ k_rot.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            
        else:
            # Standard attention calculation
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            
            # Apply relative position bias if provided
            if isinstance(self.pos_encoding, RelativePositionalEncoding) or isinstance(self.pos_encoding, PolynomialRPE):
                rel_pos_bias = self.pos_encoding.get_bias()  # [num_heads, N, N]
                attn = attn + rel_pos_bias
        
        # Softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values and reshape back
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def set_pos_encoding(self, pos_encoding):
        """Set the positional encoding method for this attention module"""
        self.pos_encoding = pos_encoding

class Block(nn.Module):
    """
    Transformer block consisting of multi-head attention and feed-forward network.
    
    Each block contains:
    1. Layer normalization
    2. Multi-head attention
    3. Drop path for regularization
    4. Feed-forward network
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs_cis=None):
        # First sub-block: attention
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis=freqs_cis))
        # Second sub-block: feed-forward network
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
    def set_pos_encoding(self, pos_encoding):
        """Set the positional encoding method for this block's attention module"""
        self.attn.set_pos_encoding(pos_encoding)

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model implementation.
    
    This class implements the complete Vision Transformer architecture, including:
    1. Patch embedding
    2. Positional encoding
    3. Transformer blocks
    4. Classification head
    
    The model supports different positional encoding methods:
    - Absolute positional encoding: Added directly to patch embeddings
    - Relative positional encoding: Added to attention logits
    - Polynomial relative positional encoding: Added to attention logits
    - RoPE-Axial: Rotary position encoding with separate x and y frequency bands
    - RoPE-Mixed: Rotary position encoding with learnable mixed frequencies
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.,
                 pos_encoding='absolute', rope_theta=100.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pos_encoding_type = pos_encoding
        self.head_dim = embed_dim // num_heads
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize positional encoding based on type
        if pos_encoding == 'absolute':
            self.pos_embed = AbsolutePositionalEncoding(embed_dim)
            self.use_pos_embed_in_forward = True  # Add to embeddings in forward pass
            self.use_rope = False
        elif pos_encoding == 'relative':
            self.pos_embed = RelativePositionalEncoding(self.num_patches, num_heads)
            self.use_pos_embed_in_forward = False  # Applied within attention mechanism
            self.use_rope = False
        elif pos_encoding == 'polynomial':
            self.pos_embed = PolynomialRPE(self.num_patches, num_heads=num_heads)
            self.use_pos_embed_in_forward = False  # Applied within attention mechanism
            self.use_rope = False
        elif pos_encoding == 'rope-axial':
            self.pos_embed = RoPEAxial(self.head_dim, theta=rope_theta)
            self.use_pos_embed_in_forward = False  # Applied within attention mechanism
            self.use_rope = True
        elif pos_encoding == 'rope-mixed':
            self.pos_embed = RoPEMixed(self.head_dim, num_heads=num_heads, theta=rope_theta)
            self.use_pos_embed_in_forward = False  # Applied within attention mechanism
            self.use_rope = True
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding}")
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Set positional encoding for all blocks if not absolute
        if not self.use_pos_embed_in_forward:
            for block in self.blocks:
                block.set_pos_encoding(self.pos_embed)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """
        Initialize weights for different layer types.
        
        Args:
            m (nn.Module): Module to initialize weights for
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward_features(self, x):
        """
        Extract features from input images through the transformer blocks.
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Features after transformer blocks
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, C, H, W] -> [B, E, H/P, W/P]
        h, w = H // self.patch_size, W // self.patch_size
        x = x.flatten(2).transpose(1, 2)  # [B, E, H/P*W/P] -> [B, H/P*W/P, E]
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional encoding if absolute
        if self.use_pos_embed_in_forward:
            x = self.pos_embed(x)
        
        # For RoPE variants, precompute frequency components
        freqs_cis = None
        if self.use_rope:
            if self.pos_encoding_type == 'rope-axial':
                freqs_cis = self.pos_embed.get_freqs_cis(h * w, x.device)
                # Ensure freqs_cis is on the same device as x
                if freqs_cis.device != x.device:
                    freqs_cis = freqs_cis.to(x.device)
            elif self.pos_encoding_type == 'rope-mixed':
                freqs_cis = self.pos_embed.get_freqs_cis(h * w, x.device)
                # Ensure freqs_cis is on the same device as x
                if freqs_cis.device != x.device:
                    freqs_cis = freqs_cis.to(x.device)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis)
            
        return x
    
    def forward(self, x):
        """
        Forward pass through the complete Vision Transformer.
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.head(x[:, 0])  # Use class token for classification
        return x 