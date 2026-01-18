"""
Module 1: Supervised Mode Embedding

Maps discrete mode labels (normal/pathological) to continuous embeddings.
"""

import torch
import torch.nn as nn


class ModeEmbedding(nn.Module):
    """
    Supervised mode embedding module.
    
    Args:
        num_modes (int): Number of modes (2 for normal/pathological)
        embed_dim (int): Dimension of mode embedding
        hidden_dim (int): Hidden dimension of MLP
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        num_modes: int = 2,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_modes = num_modes
        self.embed_dim = embed_dim
        
        # MLP: one-hot -> hidden -> embed
        self.mlp = nn.Sequential(
            nn.Linear(num_modes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, mode_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mode_labels (torch.Tensor): Mode labels [B]
                0 = normal, 1 = pathological
        
        Returns:
            mode_embeds (torch.Tensor): Mode embeddings [B, embed_dim]
        """
        # One-hot encode
        mode_onehot = torch.nn.functional.one_hot(
            mode_labels, num_classes=self.num_modes
        ).float()  # [B, num_modes]
        
        # Pass through MLP
        mode_embeds = self.mlp(mode_onehot)  # [B, embed_dim]
        
        return mode_embeds
    
    def get_mode_statistics(self, mode_labels: torch.Tensor):
        """
        Get statistics of mode embeddings for analysis.
        
        Args:
            mode_labels (torch.Tensor): Mode labels [B]
        
        Returns:
            stats (dict): Statistics including mean, std per mode
        """
        with torch.no_grad():
            embeds = self.forward(mode_labels)
            
            stats = {}
            for mode in range(self.num_modes):
                mask = mode_labels == mode
                if mask.any():
                    mode_embeds = embeds[mask]
                    stats[f'mode_{mode}_mean'] = mode_embeds.mean(dim=0)
                    stats[f'mode_{mode}_std'] = mode_embeds.std(dim=0)
            
            return stats


if __name__ == "__main__":
    # Test mode embedding
    batch_size = 4
    
    # Create module
    mode_embed = ModeEmbedding(
        num_modes=2,
        embed_dim=128,
        hidden_dim=64
    )
    
    # Test forward pass
    mode_labels = torch.tensor([0, 1, 0, 1])  # normal, pathological, normal, pathological
    mode_embeds = mode_embed(mode_labels)
    
    print(f"Mode labels: {mode_labels}")
    print(f"Mode embeds shape: {mode_embeds.shape}")
    print(f"Mode embeds:\n{mode_embeds}")
    
    # Test statistics
    stats = mode_embed.get_mode_statistics(mode_labels)
    print(f"\nMode statistics:")
    for k, v in stats.items():
        print(f"{k}: shape={v.shape}, mean={v.mean():.4f}")
