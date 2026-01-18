"""
Training script for Pathological UTI Generation.

Usage:
    python train.py --config configs/default.yaml --gpus 0,1,2,3
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from data.auspeech_dataset import AUSpeechDataset
from models.mode_embedding import ModeEmbedding
from models.adaptive_fusion import AdaptiveFusion
from models.edm_diffusion import EDMDiffusion
from utils.metrics import compute_metrics
from utils.logging import setup_logger

# Load additional configuration from pretrained weights
# Note: Pretrained encoder configs are loaded automatically from HuggingFace
# Ensure you have access to facebook/wav2vec2-large-960h and bert-base-uncased


class Trainer:
    """Main trainer class."""
    
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        # Setup logger
        self.logger = setup_logger(config.logging.log_dir, rank=rank)
        
        # Build models
        self.build_models()
        
        # Build dataloaders
        self.build_dataloaders()
        
        # Setup optimizer
        self.build_optimizer()
        
        # Setup logging
        if self.is_main:
            if config.logging.use_wandb:
                wandb.init(
                    project=config.logging.project_name,
                    name=config.logging.experiment_name,
                    config=OmegaConf.to_container(config)
                )
        
        self.global_step = 0
    
    def build_models(self):
        """Build all model components."""
        cfg = self.config.model
        
        # Module 1: Mode Embedding
        self.mode_embed = ModeEmbedding(
            num_modes=cfg.mode_embedding.num_modes,
            embed_dim=cfg.mode_embedding.embed_dim,
            hidden_dim=cfg.mode_embedding.hidden_dim,
            dropout=cfg.mode_embedding.dropout
        )
        
        # Module 2: Adaptive Fusion
        self.fusion = AdaptiveFusion(
            audio_model_name=cfg.audio_encoder.model_name,
            text_model_name=cfg.text_encoder.model_name,
            audio_feat_dim=cfg.audio_encoder.feat_dim,
            text_feat_dim=cfg.text_encoder.feat_dim,
            mode_embed_dim=cfg.mode_embedding.embed_dim,
            hidden_dim=cfg.fusion.cross_attn_dim,
            num_heads=cfg.fusion.cross_attn_heads,
            dropout=cfg.fusion.dropout,
            freeze_encoders=cfg.audio_encoder.freeze
        )
        
        # Module 3: EDM Diffusion
        self.diffusion = EDMDiffusion(
            unet_config=OmegaConf.to_container(cfg.unet),
            sigma_min=cfg.diffusion.sigma_min,
            sigma_max=cfg.diffusion.sigma_max,
            sigma_data=cfg.diffusion.sigma_data,
            rho=cfg.diffusion.rho,
            reference_feat_dim=cfg.reference_encoder.feat_dim
        )
        
        # Move to GPU
        device = torch.device(f'cuda:{self.rank}')
        self.mode_embed = self.mode_embed.to(device)
        self.fusion = self.fusion.to(device)
        self.diffusion = self.diffusion.to(device)
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            self.mode_embed = DDP(self.mode_embed, device_ids=[self.rank])
            self.fusion = DDP(self.fusion, device_ids=[self.rank])
            self.diffusion = DDP(self.diffusion, device_ids=[self.rank])
    
    def build_dataloaders(self):
        """Build training and validation dataloaders."""
        cfg = self.config.data
        
        # Training dataset
        train_dataset = AUSpeechDataset(
            data_dir=cfg.data_dir,
            split='train',
            audio_sr=cfg.audio_sr,
            uti_resolution=cfg.uti_resolution,
            num_frames=cfg.num_frames,
            augmentation=cfg.augmentation
        )
        
        # Validation dataset
        val_dataset = AUSpeechDataset(
            data_dir=cfg.data_dir,
            split='val',
            audio_sr=cfg.audio_sr,
            uti_resolution=cfg.uti_resolution,
            num_frames=cfg.num_frames,
            augmentation=None
        )
        
        # Samplers for distributed training
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank
        ) if self.world_size > 1 else None
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
    
    def build_optimizer(self):
        """Build optimizer and scheduler."""
        cfg = self.config.training
        
        # Collect parameters
        params = (
            list(self.mode_embed.parameters()) +
            list(self.fusion.parameters()) +
            list(self.diffusion.parameters())
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.max_iterations
        )
    
    def train_step(self, batch):
        """Single training step."""
        # Unpack batch
        audio = batch['audio'].cuda(self.rank)
        text_ids = batch['text_ids'].cuda(self.rank)
        text_mask = batch['text_mask'].cuda(self.rank)
        mode_labels = batch['mode_label'].cuda(self.rank)
        uti_frames = batch['uti_frames'].cuda(self.rank)
        ref_frame = batch['ref_frame'].cuda(self.rank)
        
        # Forward pass
        # 1. Mode embedding
        mode_embeds = self.mode_embed(mode_labels)
        
        # 2. Adaptive fusion
        cond, (alpha, beta, gamma) = self.fusion(
            audio=audio,
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            mode_embeds=mode_embeds
        )
        
        # 3. Diffusion loss
        loss = self.diffusion.loss(
            x_0=uti_frames,
            cond=cond,
            ref_frame=ref_frame
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.diffusion.parameters(),
                self.config.training.grad_clip
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'gamma_mean': gamma.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def validate(self):
        """Validation."""
        self.mode_embed.eval()
        self.fusion.eval()
        self.diffusion.eval()
        
        val_losses = []
        
        for batch in tqdm(self.val_loader, desc='Validation', disable=not self.is_main):
            # Similar to train_step but no backward
            audio = batch['audio'].cuda(self.rank)
            text_ids = batch['text_ids'].cuda(self.rank)
            text_mask = batch['text_mask'].cuda(self.rank)
            mode_labels = batch['mode_label'].cuda(self.rank)
            uti_frames = batch['uti_frames'].cuda(self.rank)
            ref_frame = batch['ref_frame'].cuda(self.rank)
            
            mode_embeds = self.mode_embed(mode_labels)
            cond, _ = self.fusion(audio, text_ids, text_mask, mode_embeds)
            loss = self.diffusion.loss(uti_frames, cond, ref_frame)
            
            val_losses.append(loss.item())
        
        self.mode_embed.train()
        self.fusion.train()
        self.diffusion.train()
        
        return {'val_loss': sum(val_losses) / len(val_losses)}
    
    def train(self):
        """Main training loop."""
        cfg = self.config.training
        
        self.logger.info("Starting training...")
        self.logger.info(f"Max iterations: {cfg.max_iterations}")
        self.logger.info(f"Batch size: {cfg.batch_size}")
        self.logger.info(f"Learning rate: {cfg.learning_rate}")
        
        pbar = tqdm(range(cfg.max_iterations), disable=not self.is_main)
        
        for iteration in pbar:
            # Get batch
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Train step
            metrics = self.train_step(batch)
            
            # Logging
            if iteration % cfg.log_interval == 0 and self.is_main:
                pbar.set_postfix(metrics)
                if self.config.logging.use_wandb:
                    wandb.log(metrics, step=iteration)
            
            # Validation
            if iteration % cfg.eval_interval == 0 and iteration > 0:
                val_metrics = self.validate()
                if self.is_main:
                    self.logger.info(f"Step {iteration}: {val_metrics}")
                    if self.config.logging.use_wandb:
                        wandb.log(val_metrics, step=iteration)
            
            # Checkpointing
            if iteration % cfg.save_interval == 0 and iteration > 0 and self.is_main:
                self.save_checkpoint(iteration)
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, iteration):
        """Save checkpoint."""
        ckpt_dir = os.path.join(self.config.logging.log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_{iteration:07d}.pt')
        
        torch.save({
            'iteration': iteration,
            'mode_embed': self.mode_embed.state_dict(),
            'fusion': self.fusion.state_dict(),
            'diffusion': self.diffusion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }, ckpt_path)
        
        self.logger.info(f"Saved checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs')
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Setup distributed training
    if config.training.distributed:
        # Initialize process group
        dist.init_process_group(backend=config.training.backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device(rank)
    
    # Create trainer
    trainer = Trainer(config, rank=rank, world_size=world_size)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
