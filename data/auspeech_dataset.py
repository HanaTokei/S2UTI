"""
AUSpeech Dataset for Pathological UTI Generation

Loads synchronized audio-UTI pairs with mode labels.
"""

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import librosa
import cv2
from pathlib import Path
from typing import Optional, Dict


class AUSpeechDataset(Dataset):
    """
    AUSpeech dataset for joint training with normal and pathological speech.
    
    Dataset structure:
        data_dir/
        ├── audio/
        │   ├── normal/
        │   │   ├── speaker001_utterance001.wav
        │   │   └── ...
        │   └── pathological/
        │       ├── speaker044_utterance001.wav
        │       └── ...
        ├── uti_frames/
        │   ├── normal/
        │   │   ├── speaker001_utterance001/
        │   │   │   ├── frame_000.png
        │   │   │   └── ...
        │   └── pathological/
        │       └── ...
        └── metadata.csv  # Columns: speaker_id, utterance_id, mode, text
    
    Args:
        data_dir: Root data directory
        split: 'train' or 'val'
        audio_sr: Audio sampling rate
        uti_resolution: UTI image resolution (H, W)
        num_frames: Number of frames per sequence
        augmentation: Data augmentation config
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        audio_sr: int = 16000,
        uti_resolution: tuple = (112, 112),
        num_frames: int = 15,
        augmentation: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_sr = audio_sr
        self.uti_resolution = uti_resolution
        self.num_frames = num_frames
        self.augmentation = augmentation or {}
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                f"Please ensure the dataset has been properly preprocessed. "
                f"Run: python scripts/preprocess_auspeech.py --data_dir {data_dir}"
            )
        
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by split
        if 'split' not in self.metadata.columns:
            raise ValueError(
                "Metadata CSV must contain 'split' column. "
                "This is generated during preprocessing."
            )
        
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        if len(self.metadata) == 0:
            raise ValueError(
                f"No samples found for split '{split}'. "
                f"Please check your metadata.csv and ensure preprocessing completed successfully."
            )
        
        print(f"Loaded {len(self)} samples for {split}")
        print(f"  Normal: {(self.metadata['mode'] == 'normal').sum()}")
        print(f"  Pathological: {(self.metadata['mode'] == 'pathological').sum()}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        speaker_id = row['speaker_id']
        utterance_id = row['utterance_id']
        mode = row['mode']  # 'normal' or 'pathological'
        text = row['text']
        
        # Load audio
        audio_path = self.data_dir / 'audio' / mode / f'{speaker_id}_{utterance_id}.wav'
        audio, sr = librosa.load(audio_path, sr=self.audio_sr, mono=True)
        
        # Apply audio augmentation
        if self.augmentation.get('time_masking', False) and self.split == 'train':
            audio = self._time_masking(audio)
        
        # Load UTI frames
        uti_dir = self.data_dir / 'uti_frames' / mode / f'{speaker_id}_{utterance_id}'
        uti_frames = self._load_uti_frames(uti_dir)
        
        # Get reference frame (first frame)
        ref_frame = uti_frames[0].copy()
        
        # Mode label: 0=normal, 1=pathological
        mode_label = 0 if mode == 'normal' else 1
        
        # Convert to tensors
        audio = torch.from_numpy(audio).float()
        uti_frames = torch.from_numpy(uti_frames).float().unsqueeze(0)  # [1, T, H, W]
        ref_frame = torch.from_numpy(ref_frame).float().unsqueeze(0)    # [1, H, W]
        mode_label = torch.tensor(mode_label).long()
        
        return {
            'audio': audio,
            'text': text,
            'text_ids': None,  # Will be tokenized in collate_fn
            'text_mask': None,
            'uti_frames': uti_frames,
            'ref_frame': ref_frame.repeat(3, 1, 1),  # Convert grayscale to RGB for ResNet
            'mode_label': mode_label,
            'speaker_id': speaker_id,
            'utterance_id': utterance_id
        }
    
    def _load_uti_frames(self, uti_dir: Path) -> np.ndarray:
        """Load and process UTI frame sequence."""
        # Get all frame paths
        frame_paths = sorted(uti_dir.glob('frame_*.png'))
        
        # Sample frames evenly to get num_frames
        if len(frame_paths) > self.num_frames:
            indices = np.linspace(0, len(frame_paths) - 1, self.num_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        elif len(frame_paths) < self.num_frames:
            # Pad with last frame
            frame_paths = frame_paths + [frame_paths[-1]] * (self.num_frames - len(frame_paths))
        
        # Load frames
        frames = []
        for path in frame_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.uti_resolution)
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            frames.append(img)
        
        frames = np.stack(frames, axis=0)  # [T, H, W]
        
        return frames
    
    def _time_masking(self, audio: np.ndarray, max_mask_size: int = 1600) -> np.ndarray:
        """SpecAugment-style time masking in time domain."""
        mask_size = np.random.randint(0, max_mask_size)
        mask_start = np.random.randint(0, max(1, len(audio) - mask_size))
        audio[mask_start:mask_start + mask_size] = 0
        return audio
    
    @staticmethod
    def collate_fn(batch, tokenizer):
        """
        Custom collate function with text tokenization.
        
        Args:
            batch: List of samples
            tokenizer: BERT tokenizer
        
        Returns:
            Batched samples
        """
        # Tokenize texts
        texts = [sample['text'] for sample in batch]
        text_encoded = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Stack tensors
        audio = torch.nn.utils.rnn.pad_sequence(
            [sample['audio'] for sample in batch],
            batch_first=True
        )
        
        uti_frames = torch.stack([sample['uti_frames'] for sample in batch])
        ref_frame = torch.stack([sample['ref_frame'] for sample in batch])
        mode_label = torch.stack([sample['mode_label'] for sample in batch])
        
        return {
            'audio': audio,
            'text_ids': text_encoded['input_ids'],
            'text_mask': text_encoded['attention_mask'],
            'uti_frames': uti_frames,
            'ref_frame': ref_frame,
            'mode_label': mode_label
        }


if __name__ == "__main__":
    # Test dataset
    from transformers import BertTokenizer
    
    dataset = AUSpeechDataset(
        data_dir='data/auspeech',
        split='train',
        num_frames=15
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  UTI frames shape: {sample['uti_frames'].shape}")
    print(f"  Reference frame shape: {sample['ref_frame'].shape}")
    print(f"  Mode: {sample['mode_label']}")
    print(f"  Text: {sample['text']}")
    
    # Test collate
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    batch = [dataset[i] for i in range(4)]
    batch = AUSpeechDataset.collate_fn(batch, tokenizer)
    
    print(f"\nBatch:")
    print(f"  Audio shape: {batch['audio'].shape}")
    print(f"  Text IDs shape: {batch['text_ids'].shape}")
    print(f"  UTI frames shape: {batch['uti_frames'].shape}")
