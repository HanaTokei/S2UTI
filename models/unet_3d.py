"""
3D U-Net with Temporal Attention

Enhanced 3D U-Net for video generation with temporal attention modules.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List


class TemporalAttention(nn.Module):
    """
    Temporal self-attention along time dimension.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        
        # Reshape: [B, C, T, H, W] -> [B*H*W, C, T]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        
        # Normalize
        x_norm = self.norm(x_flat)
        
        # QKV projection
        qkv = self.qkv(x_norm)  # [B*H*W, 3C, T]
        qkv = qkv.reshape(B * H * W, 3, self.num_heads, C // self.num_heads, T)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each [B*H*W, heads, C//heads, T]
        
        # Attention
        q = q.permute(0, 1, 3, 2)  # [B*H*W, heads, T, C//heads]
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)  # [B*H*W, heads, T, C//heads]
        
        # Concatenate heads
        out = out.permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        out = out.permute(0, 2, 1)  # [B*H*W, C, T]
        
        # Project
        out = self.proj(out)
        
        # Reshape back
        out = out.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]
        
        # Residual
        return x + out


class ResBlock3D(nn.Module):
    """3D Residual block with GroupNorm."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        if time_embed_dim:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)
            )
        else:
            self.time_mlp = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None):
        h = self.conv1(torch.relu(self.norm1(x)))
        
        if self.time_mlp and time_emb is not None:
            h = h + self.time_mlp(time_emb)[:, :, None, None, None]
        
        h = self.conv2(self.dropout(torch.relu(self.norm2(h))))
        
        return h + self.shortcut(x)


class UNet3D(nn.Module):
    """
    3D U-Net with temporal attention.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        model_channels: Base channel count
        channel_mult: Channel multipliers per level
        num_res_blocks: Number of residual blocks per level
        attn_resolutions: Resolutions to apply temporal attention
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [16, 8],
        dropout: float = 0.1,
        use_temporal_attn: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.use_temporal_attn = use_temporal_attn
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input convolution
        self.input_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        
        ch = model_channels
        input_ch = [ch]
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                block = ResBlock3D(ch, out_ch, time_embed_dim, dropout)
                self.encoder_blocks.append(block)
                ch = out_ch
                input_ch.append(ch)
                
                # Add temporal attention
                if use_temporal_attn and (2 ** i) in attn_resolutions:
                    self.encoder_attns.append(TemporalAttention(ch))
                else:
                    self.encoder_attns.append(None)
            
            # Downsample
            if i < len(channel_mult) - 1:
                self.encoder_blocks.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
                input_ch.append(ch)
                self.encoder_attns.append(None)
        
        # Middle
        self.middle_block1 = ResBlock3D(ch, ch, time_embed_dim, dropout)
        self.middle_attn = TemporalAttention(ch) if use_temporal_attn else None
        self.middle_block2 = ResBlock3D(ch, ch, time_embed_dim, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            
            for j in range(num_res_blocks + 1):
                skip_ch = input_ch.pop()
                block = ResBlock3D(ch + skip_ch, out_ch, time_embed_dim, dropout)
                self.decoder_blocks.append(block)
                ch = out_ch
                
                # Add temporal attention
                if use_temporal_attn and (2 ** (len(channel_mult) - 1 - i)) in attn_resolutions:
                    self.decoder_attns.append(TemporalAttention(ch))
                else:
                    self.decoder_attns.append(None)
            
            # Upsample
            if i < len(channel_mult) - 1:
                self.decoder_blocks.append(nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1))
                self.decoder_attns.append(None)
        
        # Output
        self.output_norm = nn.GroupNorm(32, ch)
        self.output_conv = nn.Conv3d(ch, out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy input [B, C, T, H, W]
            timesteps: Diffusion timesteps [B]
            cond: Optional conditioning [B, cond_dim]
        
        Returns:
            x_pred: Predicted noise or x_0 [B, C, T, H, W]
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder with skip connections
        hs = [h]
        for i, (block, attn) in enumerate(zip(self.encoder_blocks, self.encoder_attns)):
            if isinstance(block, ResBlock3D):
                h = block(h, t_emb)
            else:  # Downsample
                h = block(h)
            
            if attn is not None:
                h = attn(h)
            
            hs.append(h)
        
        # Middle
        h = self.middle_block1(h, t_emb)
        if self.middle_attn is not None:
            h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        # Decoder with skip connections
        for block, attn in zip(self.decoder_blocks, self.decoder_attns):
            if isinstance(block, ResBlock3D):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, t_emb)
            else:  # Upsample
                h = block(h)
            
            if attn is not None:
                h = attn(h)
        
        # Output
        h = self.output_conv(torch.relu(self.output_norm(h)))
        
        return h
    
    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


if __name__ == "__main__":
    # Test 3D U-Net
    batch_size = 2
    num_frames = 15
    height, width = 112, 112
    
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        model_channels=128,
        channel_mult=[1, 2, 4],
        num_res_blocks=2,
        use_temporal_attn=True
    )
    
    # Test forward
    x = torch.randn(batch_size, 1, num_frames, height, width)
    t = torch.randint(0, 1000, (batch_size,))
    
    out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
