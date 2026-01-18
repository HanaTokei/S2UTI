"""
AUSpeech Dataset Preprocessing Script

This script preprocesses the raw AUSpeech dataset into the format required by the training pipeline.

Usage:
    python preprocess_auspeech.py --data_dir /path/to/raw/auspeech --output_dir data/auspeech

Note: This script requires the raw AUSpeech dataset, which must be obtained separately.
      See the AUSpeech paper for dataset access information.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Preprocess AUSpeech dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to raw AUSpeech dataset')
    parser.add_argument('--output_dir', type=str, default='data/auspeech',
                       help='Output directory for preprocessed data')
    parser.add_argument('--audio_sr', type=int, default=16000,
                       help='Target audio sampling rate')
    parser.add_argument('--uti_resolution', type=int, nargs=2, default=[112, 112],
                       help='Target UTI image resolution')
    
    args = parser.parse_args()
    
    # Check if raw dataset exists
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            f"Raw dataset not found at {args.data_dir}. "
            f"Please download the AUSpeech dataset first. "
            f"Refer to the original AUSpeech paper for access details."
        )
    
    print("Starting preprocessing...")
    print(f"Input: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    
    # TODO: Implement preprocessing pipeline
    # Expected steps:
    # 1. Load raw audio and UTI data
    # 2. Resample audio to target_sr
    # 3. Extract and resize UTI frames
    # 4. Create train/val/test splits
    # 5. Generate metadata.csv
    
    raise NotImplementedError(
        "Preprocessing pipeline is dataset-specific and requires manual implementation. "
        "Please refer to the paper's supplementary materials for preprocessing details. "
        "Expected output format:\n"
        "  data/auspeech/\n"
        "  ├── audio/normal/*.wav\n"
        "  ├── audio/pathological/*.wav\n"
        "  ├── uti_frames/normal/speaker_utt/frame_*.png\n"
        "  ├── uti_frames/pathological/speaker_utt/frame_*.png\n"
        "  └── metadata.csv (columns: speaker_id, utterance_id, mode, text, split)"
    )


if __name__ == "__main__":
    main()
