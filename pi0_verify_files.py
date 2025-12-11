#!/usr/bin/env python3
"""
Pi0.5 Checkpoint Files Verifier

This script verifies that all required checkpoint files are present
in the checkpoint directory.

Usage:
    python pi0_verify_files.py --checkpoint path/to/checkpoint/directory
"""

import argparse
import os
from pathlib import Path


def verify_checkpoint_files(checkpoint_dir):
    """
    Verify all required checkpoint files exist.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        bool: True if all files are present, False otherwise
    """
    required_files = [
        "config.json",
        "model.safetensors",
        "preprocessor.safetensors",
        "postprocessor.safetensors",
    ]

    print(f"Verifying checkpoint files in: {checkpoint_dir}\n")
    
    all_present = True
    for file in required_files:
        filepath = os.path.join(checkpoint_dir, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_mb = size / 1024 / 1024
            print(f"✓ {file} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {file} - MISSING")
            all_present = False

    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Verify Pi0.5 checkpoint files are present"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory containing config.json, model.safetensors, etc.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory does not exist: {args.checkpoint}")
        return 1

    if not checkpoint_path.is_dir():
        print(f"Error: Path is not a directory: {args.checkpoint}")
        return 1

    all_present = verify_checkpoint_files(args.checkpoint)

    print()
    if all_present:
        print("✓ All required files are present!")
        return 0
    else:
        print("✗ Some required files are missing!")
        return 1


if __name__ == "__main__":
    exit(main())
