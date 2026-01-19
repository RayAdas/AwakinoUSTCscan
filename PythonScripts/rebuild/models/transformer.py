"""Transformer model for depth reconstruction."""

import torch
import torch.nn as nn
import math


class PositionalEncoding2D(nn.Module):
    """2D positional encoding: PE_{i,j} = PE_i^{row} + PE_j^{col}"""
    
    def __init__(self, spatial_size: int, embed_dim: int):
        """
        Args:
            spatial_size: Size of spatial dimension (H and W)
            embed_dim: Dimension of the embedding space
        """
        super().__init__()
        self.spatial_size = spatial_size
        self.embed_dim = embed_dim
        
        # Create learnable row and column positional encodings
        self.row_pe = nn.Parameter(torch.zeros(spatial_size, embed_dim))
        self.col_pe = nn.Parameter(torch.zeros(spatial_size, embed_dim))
        
        # Initialize with sinusoidal encodings
        self._init_sinusoidal()
    
    def _init_sinusoidal(self):
        """Initialize with sinusoidal positional encodings."""
        position = torch.arange(self.spatial_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * 
                            (-math.log(10000.0) / self.embed_dim))
        
        # Row encoding
        self.row_pe.data[:, 0::2] = torch.sin(position * div_term)
        self.row_pe.data[:, 1::2] = torch.cos(position * div_term)
        
        # Column encoding
        self.col_pe.data[:, 0::2] = torch.sin(position * div_term)
        self.col_pe.data[:, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, H*W, embed_dim)
        
        Returns:
            Tensor with added positional encoding of shape (B, H*W, embed_dim)
        """
        B, seq_len, embed_dim = x.shape
        H = W = self.spatial_size
        
        # Create 2D position encoding grid
        # PE_{i,j} = PE_i^{row} + PE_j^{col}
        row_indices = torch.arange(H, device=x.device).repeat_interleave(W)
        col_indices = torch.arange(W, device=x.device).repeat(H)
        
        pos_encoding = self.row_pe[row_indices] + self.col_pe[col_indices]  # (H*W, embed_dim)
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, embed_dim)
        
        return x + pos_encoding


class TransformerDepthReconstructor(nn.Module):
    """Transformer Encoder-Only model for depth reconstruction."""
    
    def __init__(
        self,
        wave_len: int = 128,
        spatial_size: int = 41,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            wave_len: Length of input wave signal
            spatial_size: Size of spatial dimension (H and W)
            embed_dim: Dimension of the embedding space
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
        """
        super().__init__()
        self.wave_len = wave_len
        self.spatial_size = spatial_size
        self.d_model = d_model
        
        # Learnable embedding network
        self.embedding = nn.Linear(d_model, d_model)
        
        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2D(spatial_size, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output head for depth prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input wave tensor of shape (B, H, W, wave_len) or (B, wave_len, H, W)
        
        Returns:
            Depth map of shape (B, H, W) or (B, 1, H, W) depending on input format
        """
        # Handle both (B, H, W, wave_len) and (B, wave_len, H, W) formats
        if x.shape[1] == self.wave_len:
            # CNN format: (B, wave_len, H, W) -> (B, H, W, wave_len)
            x = x.permute(0, 2, 3, 1)
            cnn_format = True
        else:
            cnn_format = False
        
        B, H, W, _ = x.shape
        
        # 1. Embedding: (B, H, W, wave_len) -> (B, H*W, embed_dim)
        x = self.embedding(x)
        x = x.reshape(B, H * W, self.d_model)
        
        # 2. Add positional encoding: (B, H*W, embed_dim) -> (B, H*W, embed_dim)
        x = self.pos_encoding(x)
        
        # 3. Transformer encoder: (B, H*W, embed_dim) -> (B, H*W, embed_dim)
        x = self.transformer_encoder(x)
        
        # 4. Output head: (B, H*W, embed_dim) -> (B, H*W, 1)
        x = self.output_head(x)
        
        # 5. Reshape to spatial dimensions: (B, H*W, 1) -> (B, H, W)
        x = x.squeeze(-1).reshape(B, H, W)
        
        # Return in the same format as input
        if cnn_format:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        return x
