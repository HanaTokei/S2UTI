"""
Inference script for generating UTI from speech.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt \
                       --audio_path sample.wav \
                       --output_dir outputs/
"""

import argparse
import torch
import torchaudio
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from PIL import Image

from models.mode_embedding import ModeEmbedding
from models.adaptive_fusion import AdaptiveFusion
from models.edm_diffusion import EDMDiffusion
from utils.visualization import save_uti_video


@torch.no_grad()
def inference(
    checkpoint_path: str,
    audio_path: str,
    text: str = None,
    mode: str = 'normal',
    output_dir: str = 'outputs/',
    num_steps: int = 50,
    device: str = 'cuda'
):
    """
    Generate UTI sequence from audio.
    
    Args:
        checkpoint_path: Path to model checkpoint
        audio_path: Path to input audio file
        text: Optional text transcript (if None, use ASR)
        mode: 'normal' or 'pathological'
        output_dir: Output directory
        num_steps: Number of diffusion steps
        device: Device to use
    """
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Please ensure you have either: "
            f"1) Completed training and saved checkpoints, or "
            f"2) Downloaded pretrained weights (not included in repository). "
            f"Note: Checkpoint files are typically >10GB and not included in the repo."
        )
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = ckpt['config']
    
    # Build models
    print("Building models...")
    mode_embed = ModeEmbedding(
        num_modes=config.model.mode_embedding.num_modes,
        embed_dim=config.model.mode_embedding.embed_dim,
        hidden_dim=config.model.mode_embedding.hidden_dim
    ).to(device)
    
    fusion = AdaptiveFusion(
        audio_model_name=config.model.audio_encoder.model_name,
        text_model_name=config.model.text_encoder.model_name,
        audio_feat_dim=config.model.audio_encoder.feat_dim,
        text_feat_dim=config.model.text_encoder.feat_dim,
        mode_embed_dim=config.model.mode_embedding.embed_dim,
        hidden_dim=config.model.fusion.cross_attn_dim,
        num_heads=config.model.fusion.cross_attn_heads
    ).to(device)
    
    diffusion = EDMDiffusion(
        unet_config=OmegaConf.to_container(config.model.unet),
        sigma_min=config.model.diffusion.sigma_min,
        sigma_max=config.model.diffusion.sigma_max,
        sigma_data=config.model.diffusion.sigma_data
    ).to(device)
    
    # Load weights
    mode_embed.load_state_dict(ckpt['mode_embed'])
    fusion.load_state_dict(ckpt['fusion'])
    diffusion.load_state_dict(ckpt['diffusion'])
    
    # Set to eval mode
    mode_embed.eval()
    fusion.eval()
    diffusion.eval()
    
    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = torchaudio.load(audio_path)
    if sr != config.data.audio_sr:
        audio = torchaudio.functional.resample(audio, sr, config.data.audio_sr)
    audio = audio.mean(dim=0, keepdim=True)  # Convert to mono
    audio = audio.unsqueeze(0).to(device)  # [1, 1, T]
    
    # Get text
    if text is None:
        print("Using ASR to transcribe audio...")
        # TODO: Implement ASR
        text = "placeholder text"
    
    # Tokenize text
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.model_name)
    text_encoded = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    text_ids = text_encoded['input_ids'].to(device)
    text_mask = text_encoded['attention_mask'].to(device)
    
    # Mode label
    mode_label = torch.tensor([0 if mode == 'normal' else 1]).to(device)
    
    # Generate conditioning
    print("Generating conditioning...")
    mode_embeds = mode_embed(mode_label)
    cond, (alpha, beta, gamma) = fusion(
        audio=audio.squeeze(1),
        text_input_ids=text_ids,
        text_attention_mask=text_mask,
        mode_embeds=mode_embeds
    )
    
    print(f"Mode weight (gamma): {gamma.mean().item():.4f}")
    
    # Generate UTI
    print(f"Generating UTI with {num_steps} steps...")
    shape = (1, 1, config.data.num_frames, *config.data.uti_resolution)
    
    uti_frames = diffusion.sample(
        shape=shape,
        cond=cond,
        ref_frame=None,  # TODO: Use reference frame if available
        num_steps=num_steps,
        sampler='heun',
        device=device
    )
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as images
    print(f"Saving results to {output_dir}")
    uti_frames = uti_frames.squeeze().cpu().numpy()  # [T, H, W]
    
    for t, frame in enumerate(uti_frames):
        # Normalize to [0, 255]
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        img = Image.fromarray(frame, mode='L')
        img.save(output_dir / f'frame_{t:03d}.png')
    
    # Save as video
    video_path = output_dir / 'uti_sequence.mp4'
    save_uti_video(uti_frames, video_path, fps=30)
    
    print(f"✓ Done! Saved to {output_dir}")
    print(f"  - Individual frames: frame_*.png")
    print(f"  - Video: uti_sequence.mp4")


def main():
    parser = argparse.ArgumentParser(description='Generate UTI from speech')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio_path', type=str, required=True,
                       help='Path to input audio file')
    parser.add_argument('--text', type=str, default=None,
                       help='Optional text transcript')
    parser.add_argument('--mode', type=str, default='normal',
                       choices=['normal', 'pathological'],
                       help='Speech mode')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                       help='Output directory')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    inference(
        checkpoint_path=args.checkpoint,
        audio_path=args.audio_path,
        text=args.text,
        mode=args.mode,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        device=args.device
    )


if __name__ == "__main__":
    main()
