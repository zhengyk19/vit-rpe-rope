import torch
import torch.nn as nn
import math

class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute Positional Encoding (APE)
    
    Creates fixed sinusoidal position embeddings and adds them to patch embeddings.
    Class token doesn't receive positional encoding.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to all tokens except the class token
        x[:, 1:] = x[:, 1:] + self.pe[:, :x.size(1)-1]
        return x

class RelativePositionalEncoding(nn.Module):
    """
    Relative Position Bias (RPB) for attention mechanism
    
    Implements relative position bias that is added to attention logits,
    not to token embeddings directly. This uses 2L-1 learnable parameters
    for representing all possible relative positions.
    """
    def __init__(self, num_patches, num_heads=8):
        super().__init__()
        self.num_patches = num_patches
        self.num_heads = num_heads
        
        # Initialize relative position bias table
        # One bias parameter per relative position per head
        # Add +1 for class token
        self.seq_length = num_patches + 1  # Include class token
        table_size = 2 * self.seq_length - 1  # Full size for all possible relative positions
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_heads, table_size)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Store relative position index matrix for reuse
        coords = torch.arange(self.seq_length)
        relative_coords = coords.unsqueeze(1) - coords.unsqueeze(0)  # [L+1, L+1]
        # Shift to make values non-negative from 0 to 2L-2
        relative_coords = relative_coords + (self.seq_length - 1)
        
        # Clamp to ensure all indices are valid
        relative_coords = torch.clamp(relative_coords, 0, table_size - 1)
        
        self.register_buffer("relative_position_index", relative_coords)
        
    def forward(self, x):
        # This module doesn't modify x directly
        # It only stores position bias information that will be used in attention
        return x
        
    def get_bias(self):
        """
        Retrieve the relative position bias matrix for attention computation
        
        Returns:
            torch.Tensor: Position bias of shape [num_heads, seq_len, seq_len]
        """
        # Retrieve the pre-computed relative position indices
        relative_coords = self.relative_position_index
        
        # Get position bias from table using relative position indices
        bias = self.relative_position_bias_table[:, relative_coords]  # [num_heads, L+1, L+1]
        
        return bias

class PolynomialRPE(nn.Module):
    """
    Polynomial-based Relative Position Bias for attention
    
    Computes position bias as a polynomial function of L1 distance between
    patch coordinates, which is added to attention logits.
    """
    def __init__(self, num_patches, degree=3, num_heads=8, shared_across_heads=True):
        super().__init__()
        self.num_patches = num_patches
        self.degree = degree
        self.num_heads = num_heads
        self.shared_across_heads = shared_across_heads
        
        # Calculate patch grid size
        self.grid_size = int(math.sqrt(num_patches))
        
        # Initialize polynomial coefficients
        if shared_across_heads:
            self.coefficients = nn.Parameter(torch.randn(degree + 1))
        else:
            self.coefficients = nn.Parameter(torch.randn(num_heads, degree + 1))
        
    def forward(self, x):
        # This module doesn't modify x directly
        # It only stores polynomial bias information
        return x
        
    def get_bias(self):
        """
        Compute polynomial-based relative position bias matrix for attention
        
        Returns:
            torch.Tensor: Position bias of shape [num_heads, seq_len, seq_len]
        """
        # Create grid coordinates
        grid_size = self.grid_size
        y_coords = torch.arange(grid_size, device=self.coefficients.device).repeat(grid_size)
        x_coords = torch.arange(grid_size, device=self.coefficients.device).repeat_interleave(grid_size)
        
        # Compute L1 distances between all pairs of positions
        y_dist = (y_coords.unsqueeze(1) - y_coords.unsqueeze(0)).abs()
        x_dist = (x_coords.unsqueeze(1) - x_coords.unsqueeze(0)).abs()
        l1_dist = y_dist + x_dist  # [num_patches, num_patches]
        
        # Compute polynomial features
        poly_features = torch.stack([l1_dist.pow(i) for i in range(self.degree + 1)], dim=-1)
        
        # Calculate bias
        if self.shared_across_heads:
            bias = (poly_features @ self.coefficients).unsqueeze(0)  # [1, num_patches, num_patches]
            bias = bias.expand(self.num_heads, -1, -1)  # [num_heads, num_patches, num_patches]
        else:
            # Each head has its own polynomial
            coeffs = self.coefficients.unsqueeze(1).unsqueeze(1)  # [num_heads, 1, 1, degree+1]
            bias = (poly_features.unsqueeze(0) @ coeffs).squeeze(-1)  # [num_heads, num_patches, num_patches]
        
        # Add a row and column for class token
        # For simplicity, we set class token's relative position bias to 0
        padding = torch.zeros(self.num_heads, self.num_patches + 1, self.num_patches + 1, 
                             device=bias.device)
        padding[:, 1:, 1:] = bias
        
        return padding

class RoPEAxial(nn.Module):
    """
    Rotary Position Embedding with Axial Frequencies (RoPE-Axial)
    
    In this variant, we apply different rotation frequencies to x and y coordinates,
    dividing the embedding dimensions into two halves. This is suitable for 2D image data
    where positions have both row and column coordinates.
    
    Each axis (x and y) uses half of the available frequency bands, allowing
    the model to capture spatial relationships in both dimensions.
    """
    def __init__(self, dim, theta=100.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        half_dim = dim // 4  # Quarter of dimensions for each axis component (sin/cos)
        
        # Create less frequent bands for 2D - note the division by 4 instead of 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float) / half_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x):
        # RoPE doesn't modify input embeddings directly
        # It only stores frequency information for attention computation
        return x
    
    def init_t_xy(self, h, w, device):
        """
        Generate x and y position indices for a 2D grid
        
        Args:
            h (int): Height of the grid
            w (int): Width of the grid
            device: Device to create tensors on
            
        Returns:
            tuple: (t_x, t_y) position indices
        """
        t = torch.arange(h * w, device=device, dtype=torch.float32)
        t_x = (t % w).float()  # x coordinates (columns)
        t_y = torch.div(t, w, rounding_mode='floor').float()  # y coordinates (rows)
        return t_x, t_y
    
    def get_freqs_cis(self, seq_len, device):
        """
        Compute cosine and sine components for rotary embeddings using axial frequencies
        
        Args:
            seq_len (int): Sequence length (h*w)
            device: Device to create tensors on
            
        Returns:
            tuple: (cos, sin) tensors for rotary embeddings [seq_len, dim/2]
        """
        # For a typical patch-based image, we need the grid dimensions
        grid_size = int(math.sqrt(seq_len))
        
        # Get x and y coordinates for each position
        t_x, t_y = self.init_t_xy(grid_size, grid_size, device)
        
        # Separate frequencies for x and y dimensions
        freqs_x = torch.outer(t_x, self.inv_freq)  # [seq_len, dim/4]
        freqs_y = torch.outer(t_y, self.inv_freq)  # [seq_len, dim/4]
        
        # Combine into a single tensor with all frequencies
        # Each position has separate x and y frequencies
        freqs = torch.cat([freqs_x, freqs_y], dim=-1)  # [seq_len, dim/2]
        
        # Return cosine and sine components separately
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin

class RoPEMixed(nn.Module):
    """
    Rotary Position Embedding with Mixed Learnable Frequencies (RoPE-Mixed)
    
    In this variant, frequencies for x and y dimensions are mixed, allowing the model
    to capture diagonal relationships. The frequencies can be learned during training,
    unlike standard RoPE which uses fixed frequencies.
    
    This implementation initializes random frequencies for each attention head and supports
    learned frequencies that can adapt during training.
    """
    def __init__(self, dim, num_heads, theta=10.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.theta = theta
        
        # Initialize learnable frequencies for each axis and head
        # The frequencies are initialized randomly and will be learned during training
        freqs_x = []
        freqs_y = []
        half_dim = dim // 2
        
        # Create magnitude based on theta
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        
        # Initialize random angles for each head
        for _ in range(num_heads):
            # Random initial angle for this head
            angles = torch.rand(1) * 2 * torch.pi
            
            # Create x and y frequency components with 90-degree phase shift
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
            
            freqs_x.append(fx)
            freqs_y.append(fy)
            
        freqs_x = torch.stack(freqs_x, dim=0)  # [num_heads, dim/2]
        freqs_y = torch.stack(freqs_y, dim=0)  # [num_heads, dim/2]
        
        # Stack x and y frequencies for learnable parameters
        freqs = torch.stack([freqs_x, freqs_y], dim=0)  # [2, num_heads, dim/2]
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
        
    def forward(self, x):
        # RoPE doesn't modify input embeddings directly
        return x
    
    def init_t_xy(self, h, w, device):
        """
        Generate x and y position indices for a 2D grid
        
        Args:
            h (int): Height of the grid
            w (int): Width of the grid
            device: Device to create tensors on
            
        Returns:
            tuple: (t_x, t_y) position indices
        """
        t = torch.arange(h * w, device=device, dtype=torch.float32)
        t_x = (t % w).float()  # x coordinates (columns)
        t_y = torch.div(t, w, rounding_mode='floor').float()  # y coordinates (rows)
        return t_x, t_y
    
    def get_freqs_cis(self, seq_len, device):
        """
        Compute mixed frequency components for all attention heads
        
        Args:
            seq_len (int): Sequence length (h*w)
            device: Device to create tensors on
            
        Returns:
            tuple: (cos, sin) tensors with mixed frequencies [num_heads, seq_len, dim/2]
        """
        # For a typical patch-based image, we need the grid dimensions
        grid_size = int(math.sqrt(seq_len))
        
        # Get x and y coordinates for each position
        t_x, t_y = self.init_t_xy(grid_size, grid_size, device)
        t_x = t_x.to(device=self.freqs.device)
        t_y = t_y.to(device=self.freqs.device)
        
        # Calculate mixed frequencies for all heads
        # Use matrix multiplication for efficient computation
        with torch.cuda.amp.autocast(enabled=False):
            # Compute frequency components for each head
            # Multiply positions with frequency components for both x and y
            freqs_x = (t_x.unsqueeze(-1) @ self.freqs[0].unsqueeze(-2))  # [seq_len, num_heads, dim/2]
            freqs_y = (t_y.unsqueeze(-1) @ self.freqs[1].unsqueeze(-2))  # [seq_len, num_heads, dim/2]
            
            # Reshape and sum the components
            freqs_x = freqs_x.view(seq_len, self.num_heads, -1).permute(1, 0, 2)  # [num_heads, seq_len, dim/2]
            freqs_y = freqs_y.view(seq_len, self.num_heads, -1).permute(1, 0, 2)  # [num_heads, seq_len, dim/2]
            
            # Calculate combined phase
            freqs = freqs_x + freqs_y
            
            # Return cosine and sine components separately
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
        
        return cos, sin 