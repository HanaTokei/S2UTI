# Joint Training Strategy for Pathological Speech Ultrasound Tongue Image Generation

PyTorch implementation of "Joint Training Strategy for Pathological Speech Ultrasound Tongue Image Generation".

## Introduction

This repository implements a diffusion-based framework for generating ultrasound tongue images (UTI) from speech, addressing the data scarcity challenge in pathological speech generation through joint training with normal speech data.

The model consists of three core modules:
- Supervised mode embedding for speech pattern distinction
- Adaptive multimodal fusion combining audio, text, and mode information
- EDM-based diffusion generation with temporal attention and reference frame conditioning

## Model Architecture

```
Input: Audio Waveform + Text + Mode Label (0=normal, 1=pathological)
         |                |              |
         v                v              v
    ┌─────────────────────────────────────────┐
    │  Module 1: Supervised Mode Embedding    │
    │  - One-hot encoding                     │
    │  - MLP (2 → 64 → 128)                   │
    └─────────────────────────────────────────┘
                        |
                        | mode embedding (128-dim)
                        v
    ┌─────────────────────────────────────────┐
    │  Module 2: Adaptive Fusion              │
    │  - Wav2Vec2.0 (audio features)          │
    │  - BERT (text features)                 │
    │  - Cross-attention fusion               │
    │  - Modulation network → α, β, γ         │
    └─────────────────────────────────────────┘
                        |
                        | fused condition (α, β, γ)
                        v
    ┌─────────────────────────────────────────┐
    │  Module 3: EDM Diffusion                │
    │  - 3D U-Net backbone                    │
    │  - Temporal attention                   │
    │  - Reference frame conditioning         │
    │  - Heun sampler (50 steps)              │
    └─────────────────────────────────────────┘
                        |
                        v
         Output: UTI Sequence (15 × 112 × 112)
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/pathological-uti-generation.git
cd pathological-uti-generation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Note: Pretrained models (Wav2Vec2.0, BERT) will be downloaded automatically from HuggingFace on first use.

## Data Preparation

This project uses the AUSpeech dataset. The dataset is not included in this repository.

Organize the data as follows:

```
data/
└── auspeech/
    ├── audio/
    │   ├── normal/
    │   └── pathological/
    ├── uti_frames/
    │   ├── normal/
    │   └── pathological/
    └── metadata.csv
```

Preprocessing:

```bash
python scripts/preprocess_auspeech.py --data_dir /path/to/raw/auspeech
```

Required metadata.csv columns: `speaker_id`, `utterance_id`, `mode`, `text`, `split`

## Training

Train the model:

```bash
python train.py --config configs/default.yaml
```

For multi-GPU training:

```bash
python train.py --config configs/default.yaml --gpus 0,1,2,3
```

Checkpoints are saved in `checkpoints/` every 50K iterations.

## Inference

Generate UTI from audio:

```bash
python inference.py --checkpoint checkpoints/best_model.pt \
                    --audio_path samples/audio.wav \
                    --output_dir outputs/
```

Note: Pretrained checkpoint files are not included in this repository.

## Project Structure

```
pathological-uti-generation/
├── configs/              # Configuration files
├── data/                 # Dataset loaders
├── models/              # Model implementations
│   ├── mode_embedding.py
│   ├── adaptive_fusion.py
│   ├── edm_diffusion.py
│   └── unet_3d.py
├── utils/               # Utility functions
├── scripts/             # Preprocessing scripts
├── train.py
└── inference.py
```

## Configuration

Key hyperparameters are defined in `configs/default.yaml`:

```yaml
mode_embed_dim: 128
audio_feat_dim: 1024
text_feat_dim: 768
sigma_min: 0.002
sigma_max: 160
batch_size: 4
learning_rate: 5e-4
max_iterations: 5000000
```

## Status

This is a course project currently under development. Some components are still being finalized.

## License

MIT License
