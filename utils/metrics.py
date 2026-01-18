"""
Evaluation metrics for UTI generation.

Implements:
- LPIPS (Learned Perceptual Image Patch Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Fréchet Inception Distance)
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class LPIPSMetric:
    """LPIPS metric using AlexNet."""
    
    def __init__(self, net='alex', device='cuda'):
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.device = device
    
    def __call__(self, pred, target):
        """
        Compute LPIPS between predicted and target images.
        
        Args:
            pred: [B, C, H, W] in range [-1, 1] or [0, 1]
            target: [B, C, H, W]
        
        Returns:
            lpips_score: Mean LPIPS distance
        """
        # Ensure range [-1, 1]
        if pred.min() >= 0:
            pred = pred * 2 - 1
        if target.min() >= 0:
            target = target * 2 - 1
        
        # Expand to 3 channels if grayscale
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            dist = self.loss_fn(pred, target)
        
        return dist.mean().item()


def compute_psnr(pred, target):
    """
    Compute PSNR.
    
    Args:
        pred: [B, C, H, W] numpy array
        target: [B, C, H, W] numpy array
    
    Returns:
        psnr: Mean PSNR
    """
    psnr_values = []
    
    for i in range(len(pred)):
        p = pred[i].squeeze()
        t = target[i].squeeze()
        
        psnr = peak_signal_noise_ratio(
            t, p,
            data_range=t.max() - t.min()
        )
        psnr_values.append(psnr)
    
    return np.mean(psnr_values)


def compute_ssim(pred, target):
    """
    Compute SSIM.
    
    Args:
        pred: [B, C, H, W] numpy array
        target: [B, C, H, W] numpy array
    
    Returns:
        ssim: Mean SSIM
    """
    ssim_values = []
    
    for i in range(len(pred)):
        p = pred[i].squeeze()
        t = target[i].squeeze()
        
        ssim = structural_similarity(
            t, p,
            data_range=t.max() - t.min()
        )
        ssim_values.append(ssim)
    
    return np.mean(ssim_values)


def compute_fid(pred_features, target_features):
    """
    Compute FID (Fréchet Inception Distance).
    
    Args:
        pred_features: [N, D] numpy array of features
        target_features: [M, D] numpy array of features
    
    Returns:
        fid: FID score
    """
    mu1 = np.mean(pred_features, axis=0)
    sigma1 = np.cov(pred_features, rowvar=False)
    
    mu2 = np.mean(target_features, axis=0)
    sigma2 = np.cov(target_features, rowvar=False)
    
    # Compute FID
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid


def compute_metrics(pred, target, lpips_metric=None):
    """
    Compute all metrics.
    
    Args:
        pred: Predicted images [B, C, H, W] torch.Tensor
        target: Target images [B, C, H, W] torch.Tensor
        lpips_metric: Precomputed LPIPS metric
    
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # LPIPS
    if lpips_metric is not None:
        metrics['lpips'] = lpips_metric(pred, target)
    
    # Convert to numpy for other metrics
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # PSNR
    metrics['psnr'] = compute_psnr(pred_np, target_np)
    
    # SSIM
    metrics['ssim'] = compute_ssim(pred_np, target_np)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    import scipy.linalg
    
    batch_size = 4
    height, width = 112, 112
    
    # Create dummy data
    pred = torch.randn(batch_size, 1, height, width)
    target = torch.randn(batch_size, 1, height, width)
    
    # Compute metrics
    lpips_metric = LPIPSMetric(device='cpu')
    metrics = compute_metrics(pred, target, lpips_metric)
    
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
