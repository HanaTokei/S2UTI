"""
Module 2: Adaptive Condition Fusion

Dynamically integrates audio, text, and mode embeddings with
adaptive modulation mechanism.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel
from typing import Optional


class AdaptiveFusion(nn.Module):
    """
    Adaptive multimodal fusion module.
    
    Integrates:
    - Audio features (Wav2Vec2.0)
    - Text features (BERT)
    - Mode embeddings
    
    With dynamic weight adjustment based on input type.
    """
    
    def __init__(
        self,
        audio_model_name: str = "facebook/wav2vec2-large-960h",
        text_model_name: str = "bert-base-uncased",
        audio_feat_dim: int = 1024,
        text_feat_dim: int = 768,
        mode_embed_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoders: bool = True
    ):
        super().__init__()
        
        # Audio encoder (Wav2Vec2.0)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        if freeze_encoders:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        if freeze_encoders:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Feature projections
        self.audio_proj = nn.Linear(audio_feat_dim, hidden_dim)
        self.text_proj = nn.Linear(text_feat_dim, hidden_dim)
        
        # Cross-attention for audio-text fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Modulation network (learns α, β, γ)
        self.modulation_net = nn.Sequential(
            nn.Linear(hidden_dim + mode_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3 * hidden_dim)  # α, β, γ
        )
        
        # Condition attention for mode integration
        self.mode_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        mode_embeds: torch.Tensor,
        feature_map: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            audio: Raw audio waveform [B, T_audio]
            text_input_ids: BERT input IDs [B, T_text]
            text_attention_mask: BERT attention mask [B, T_text]
            mode_embeds: Mode embeddings [B, mode_dim]
            feature_map: Optional intermediate features [B, C, H, W] for modulation
        
        Returns:
            fused_features: Fused condition features [B, T, hidden_dim]
            modulation_params: (alpha, beta, gamma) for AdaGN
        """
        B = audio.size(0)
        
        # Extract audio features
        with torch.no_grad() if not self.audio_encoder.training else torch.enable_grad():
            audio_outputs = self.audio_encoder(audio)
            audio_feats = audio_outputs.last_hidden_state  # [B, T_a, 1024]
        
        # Extract text features
        with torch.no_grad() if not self.text_encoder.training else torch.enable_grad():
            text_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_feats = text_outputs.last_hidden_state  # [B, T_t, 768]
        
        # Project to common dimension
        audio_feats = self.audio_proj(audio_feats)  # [B, T_a, D]
        text_feats = self.text_proj(text_feats)     # [B, T_t, D]
        
        # Cross-attention: audio queries text
        fused_feats, _ = self.cross_attn(
            query=audio_feats,
            key=text_feats,
            value=text_feats,
            key_padding_mask=~text_attention_mask.bool() if text_attention_mask is not None else None
        )  # [B, T_a, D]
        
        # Pool to single vector for modulation
        fused_pooled = fused_feats.mean(dim=1)  # [B, D]
        
        # Concatenate with mode embedding
        mode_cond = torch.cat([fused_pooled, mode_embeds], dim=1)  # [B, D + mode_dim]
        
        # Generate modulation parameters
        modulation = self.modulation_net(mode_cond)  # [B, 3*D]
        alpha, beta, gamma = torch.chunk(modulation, 3, dim=1)  # Each [B, D]
        
        # If feature map provided, apply modulation
        if feature_map is not None:
            # Reshape for broadcasting
            alpha = alpha.view(B, -1, 1, 1)
            beta = beta.view(B, -1, 1, 1)
            
            # AdaGN modulation
            modulated_feats = (1 + alpha) * feature_map + beta
            
            return modulated_feats, (alpha, beta, gamma)
        
        # Otherwise return fused features and params
        return fused_feats, (alpha, beta, gamma)
    
    def get_modulation_weights(self, mode_labels: torch.Tensor):
        """
        Analyze modulation weight distribution across modes.
        
        Args:
            mode_labels: Mode labels [B]
        
        Returns:
            stats: Statistics of gamma (mode attention weight)
        """
        stats = {}
        # This would need actual forward pass with data
        # Placeholder for analysis
        return stats


if __name__ == "__main__":
    # Test adaptive fusion
    batch_size = 2
    audio_len = 16000  # 1 second
    text_len = 32
    
    # Create module
    fusion = AdaptiveFusion(
        audio_feat_dim=1024,
        text_feat_dim=768,
        mode_embed_dim=128,
        hidden_dim=512
    )
    
    # Create dummy inputs
    audio = torch.randn(batch_size, audio_len)
    text_ids = torch.randint(0, 1000, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len)
    mode_embeds = torch.randn(batch_size, 128)
    
    # Forward pass
    fused_feats, (alpha, beta, gamma) = fusion(
        audio=audio,
        text_input_ids=text_ids,
        text_attention_mask=text_mask,
        mode_embeds=mode_embeds
    )
    
    print(f"Fused features shape: {fused_feats.shape}")
    print(f"Alpha shape: {alpha.shape}")
    print(f"Beta shape: {beta.shape}")
    print(f"Gamma shape: {gamma.shape}")
    print(f"\nGamma statistics:")
    print(f"  Mean: {gamma.mean():.4f}")
    print(f"  Std: {gamma.std():.4f}")
