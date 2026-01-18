"""Visualization utilities for UTI sequences."""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def save_uti_video(frames, output_path, fps=30):
    """
    Save UTI frame sequence as video.
    
    Args:
        frames: [T, H, W] numpy array
        output_path: Output video path
        fps: Frames per second
    """
    T, H, W = frames.shape
    
    # Normalize to [0, 255]
    frames = ((frames - frames.min()) / (frames.max() - frames.min()) * 255).astype(np.uint8)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H), isColor=False)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Saved video: {output_path}")


def visualize_comparison(real, generated, save_path=None):
    """
    Visualize real vs generated UTI sequences.
    
    Args:
        real: [T, H, W] numpy array
        generated: [T, H, W] numpy array
        save_path: Optional path to save figure
    """
    T = min(real.shape[0], generated.shape[0], 5)  # Show first 5 frames
    
    fig, axes = plt.subplots(2, T, figsize=(3*T, 6))
    
    for t in range(T):
        # Real
        axes[0, t].imshow(real[t], cmap='gray')
        axes[0, t].set_title(f'Real Frame {t}')
        axes[0, t].axis('off')
        
        # Generated
        axes[1, t].imshow(generated[t], cmap='gray')
        axes[1, t].set_title(f'Generated Frame {t}')
        axes[1, t].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to training log
        save_path: Optional path to save figure
    """
    # TODO: Implement log parsing and plotting
    pass


if __name__ == "__main__":
    # Test visualization
    T, H, W = 15, 112, 112
    
    real = np.random.rand(T, H, W)
    generated = np.random.rand(T, H, W)
    
    # Save video
    save_uti_video(generated, 'test_output.mp4', fps=30)
    
    # Visualize comparison
    visualize_comparison(real, generated, save_path='test_comparison.png')
