"""
Module 3: EDM Diffusion Framework

Implements Elucidated Diffusion Model (Karras et al., 2022)
with reference frame conditioning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from .unet_3d import UNet3D
from torchvision.models import resnet18


class EDMDiffusion(nn.Module):
    """
    EDM-based diffusion model for UTI generation.
    
    Features:
    - EDM preconditioning
    - 3D U-Net with temporal attention
    - Reference frame conditioning
    - Heun sampling
    """
    
    def __init__(
        self,
        unet_config: dict,
        sigma_min: float = 0.002,
        sigma_max: float = 160.0,
        sigma_data: float = 0.25,
        rho: float = 7.0,
        reference_feat_dim: int = 512
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        
        # 3D U-Net denoiser
        self.unet = UNet3D(**unet_config)
        
        # Reference frame encoder (ResNet18)
        self.ref_encoder = resnet18(pretrained=True)
        self.ref_encoder.fc = nn.Linear(512, reference_feat_dim)
        
        # Reference conditioning via AdaGN
        self.ref_modulation = nn.Sequential(
            nn.Linear(reference_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, unet_config['model_channels'] * 2)  # gamma_r, beta_r
        )
    
    def preconditioning(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        EDM preconditioning.
        
        Returns:
            c_skip, c_out, c_in, c_noise
        """
        sigma = sigma.view(-1, 1, 1, 1, 1)
        sigma_data = self.sigma_data
        
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_noise = 0.25 * torch.log(sigma)
        
        return c_skip, c_out, c_in, c_noise.squeeze()
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        ref_frame: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Denoising forward pass.
        
        Args:
            x_noisy: Noisy input [B, C, T, H, W]
            sigma: Noise level [B]
            cond: Conditioning from fusion module [B, cond_dim]
            ref_frame: Reference frame [B, C, H, W]
        
        Returns:
            x_pred: Denoised prediction [B, C, T, H, W]
        """
        # Preconditioning
        c_skip, c_out, c_in, c_noise = self.preconditioning(x_noisy, sigma)
        
        # Encode reference frame
        if ref_frame is not None:
            # Extract features
            B = ref_frame.size(0)
            ref_feat = self.ref_encoder(ref_frame)  # [B, ref_feat_dim]
            
            # Generate modulation parameters
            ref_mod = self.ref_modulation(ref_feat)  # [B, 2*C]
            gamma_r, beta_r = torch.chunk(ref_mod, 2, dim=1)
        else:
            gamma_r, beta_r = None, None
        
        # U-Net denoising
        F_theta = self.unet(
            c_in * x_noisy,
            c_noise,
            cond=cond
        )
        
        # Apply reference frame conditioning if available
        if gamma_r is not None and beta_r is not None:
            # Reshape for broadcasting [B, C, 1, 1, 1]
            gamma_r = gamma_r.view(B, -1, 1, 1, 1)
            beta_r = beta_r.view(B, -1, 1, 1, 1)
            
            # AdaGN modulation
            F_theta = gamma_r * F_theta + beta_r
        
        # EDM output
        D_theta = c_skip * x_noisy + c_out * F_theta
        
        return D_theta
    
    def loss(
        self,
        x_0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        ref_frame: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        EDM training loss.
        
        Args:
            x_0: Clean data [B, C, T, H, W]
            cond: Conditioning [B, cond_dim]
            ref_frame: Reference frame [B, C, H, W]
        
        Returns:
            loss: Scalar loss
        """
        B = x_0.size(0)
        
        # Sample noise level
        rnd_normal = torch.randn(B, device=x_0.device)
        sigma = (
            self.sigma_max ** (1 / self.rho)
            + rnd_normal * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        
        # Add noise
        noise = torch.randn_like(x_0)
        x_noisy = x_0 + sigma.view(-1, 1, 1, 1, 1) * noise
        
        # Denoise
        D_theta = self.forward(x_noisy, sigma, cond, ref_frame)
        
        # Loss weight
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        weight = weight.view(-1, 1, 1, 1, 1)
        
        # MSE loss
        loss = weight * (D_theta - x_0) ** 2
        
        return loss.mean()
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        ref_frame: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        sampler: str = "heun",
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Generate samples using EDM sampling.
        
        Args:
            shape: Output shape [B, C, T, H, W]
            cond: Conditioning [B, cond_dim]
            ref_frame: Reference frame [B, C, H, W]
            num_steps: Number of sampling steps
            sampler: Sampling method ('heun' or 'euler')
            device: Device
        
        Returns:
            samples: Generated samples [B, C, T, H, W]
        """
        # Initialize from noise
        x = torch.randn(shape, device=device) * self.sigma_max
        
        # Time schedule
        t = torch.linspace(self.sigma_max, self.sigma_min, num_steps, device=device)
        
        if sampler == "heun":
            return self._sample_heun(x, t, cond, ref_frame)
        elif sampler == "euler":
            return self._sample_euler(x, t, cond, ref_frame)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
    
    def _sample_heun(
        self,
        x: torch.Tensor,
        t_schedule: torch.Tensor,
        cond: Optional[torch.Tensor],
        ref_frame: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Second-order Heun sampler."""
        for i in range(len(t_schedule) - 1):
            t_cur = t_schedule[i]
            t_next = t_schedule[i + 1]
            
            # First evaluation
            sigma_cur = torch.full((x.size(0),), t_cur, device=x.device)
            d_cur = (x - self.forward(x, sigma_cur, cond, ref_frame)) / t_cur
            
            # Euler step
            x_next = x + (t_next - t_cur) * d_cur
            
            # Second evaluation (if not last step)
            if i < len(t_schedule) - 2:
                sigma_next = torch.full((x.size(0),), t_next, device=x.device)
                d_next = (x_next - self.forward(x_next, sigma_next, cond, ref_frame)) / t_next
                
                # Correct with midpoint
                x_next = x + (t_next - t_cur) * (d_cur + d_next) / 2
            
            x = x_next
        
        return x
    
    def _sample_euler(
        self,
        x: torch.Tensor,
        t_schedule: torch.Tensor,
        cond: Optional[torch.Tensor],
        ref_frame: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """First-order Euler sampler."""
        for i in range(len(t_schedule) - 1):
            t_cur = t_schedule[i]
            t_next = t_schedule[i + 1]
            
            sigma = torch.full((x.size(0),), t_cur, device=x.device)
            d = (x - self.forward(x, sigma, cond, ref_frame)) / t_cur
            x = x + (t_next - t_cur) * d
        
        return x


if __name__ == "__main__":
    # Test EDM diffusion
    batch_size = 2
    num_frames = 15
    height, width = 112, 112
    
    unet_config = {
        'in_channels': 1,
        'out_channels': 1,
        'model_channels': 64,
        'channel_mult': [1, 2, 4],
        'num_res_blocks': 2,
        'use_temporal_attn': True
    }
    
    model = EDMDiffusion(
        unet_config=unet_config,
        sigma_min=0.002,
        sigma_max=160.0
    )
    
    # Test training loss
    x_0 = torch.randn(batch_size, 1, num_frames, height, width)
    ref_frame = torch.randn(batch_size, 3, height, width)
    
    loss = model.loss(x_0, ref_frame=ref_frame)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    model.eval()
    samples = model.sample(
        shape=(batch_size, 1, num_frames, height, width),
        ref_frame=ref_frame,
        num_steps=10,
        device='cpu'
    )
    print(f"Generated samples shape: {samples.shape}")
