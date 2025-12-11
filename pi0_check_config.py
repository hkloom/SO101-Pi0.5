#!/usr/bin/env python3
"""
Pi0.5 Configuration Checker

This script loads and displays the configuration from a Pi0.5 checkpoint,
helping you verify the model architecture and input/output shapes.

Usage:
    python pi0_check_config.py --checkpoint path/to/checkpoint/directory
"""

import argparse
import json
import os
from pathlib import Path


def check_checkpoint_files(checkpoint_dir):
    """Check if all required checkpoint files exist."""
    required_files = [
        "config.json",
        "model.safetensors",
        "preprocessor.safetensors",
        "postprocessor.safetensors",
    ]

    print("Checking checkpoint files...")
    all_present = True
    for file in required_files:
        filepath = os.path.join(checkpoint_dir, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {file} ({size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  ✗ {file} - MISSING")
            all_present = False

    return all_present


def main():
    parser = argparse.ArgumentParser(description="Check Pi0.5 checkpoint configuration")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory containing config.json, model.safetensors, etc.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path / "config.json"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory does not exist: {args.checkpoint}")
        return

    if not config_path.exists():
        print(f"Error: config.json not found in {args.checkpoint}")
        return

    # Check all files
    print(f"\nCheckpoint directory: {args.checkpoint}\n")
    files_ok = check_checkpoint_files(args.checkpoint)
    print()

    # Load and display config
    print("Loading configuration...")
    with open(config_path, "r") as f:
        config = json.load(f)

    print("\n" + "=" * 60)
    print("Configuration Details")
    print("=" * 60)

    # Policy type
    policy_type = config.get("policy_type", "Unknown")
    print(f"\nPolicy Type: {policy_type}")

    # Input shapes
    input_shapes = config.get("input_shapes", {})
    if input_shapes:
        print("\nInput Shapes:")
        for key, shape in input_shapes.items():
            print(f"  {key}: {shape}")

    # Output shapes
    output_shapes = config.get("output_shapes", {})
    if output_shapes:
        print("\nOutput Shapes:")
        for key, shape in output_shapes.items():
            print(f"  {key}: {shape}")

    # Model architecture details
    if "model" in config:
        model_config = config["model"]
        print("\nModel Architecture:")
        if "name" in model_config:
            print(f"  Name: {model_config['name']}")
        if "hidden_dim" in model_config:
            print(f"  Hidden Dimension: {model_config['hidden_dim']}")
        if "num_layers" in model_config:
            print(f"  Number of Layers: {model_config['num_layers']}")

    # Full config (pretty printed)
    print("\n" + "=" * 60)
    print("Full Configuration (JSON)")
    print("=" * 60)
    print(json.dumps(config, indent=2))

    if not files_ok:
        print("\n⚠ Warning: Some required files are missing!")


if __name__ == "__main__":
    main()
