#!/usr/bin/env python3
"""
PI-0.5 Checkpoint Download & Setup

Downloads checkpoint from HuggingFace, patches config.json, verifies files,
and displays configuration details at the end.

Usage:
    python ./pi/ckpt/download.py --repo username/PIv2
    python ./pi/ckpt/download.py --repo username/PIv2 --output-dir ./pi/ckpt/custom_name
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Files to download from the HuggingFace repo
CHECKPOINT_FILES = [
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json", # defines how to normalize inputs
    "policy_postprocessor.json", # defines how to denormalize outputs
    "policy_preprocessor_step_2_normalizer_processor.safetensors", # normalization stats for inputs
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors", # denormalization stats for outputs
]


def fix_config(checkpoint_dir):
    """Patch config.json type field from 'pi05' to 'pi0' so LeRobot can load it."""
    config_path = os.path.join(checkpoint_dir, "config.json")

    with open(config_path) as f:
        config = json.load(f)
    config["type"] = "pi0"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def check_files(checkpoint_dir):
    """Check if all required checkpoint files exist and return status."""
    print(f"\nVerifying checkpoint files in: {checkpoint_dir}\n")

    missing = []
    for file in CHECKPOINT_FILES:
        filepath = Path(checkpoint_dir) / file
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {file} ({size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  ✗ {file} - MISSING")
            missing.append(file)

    print()
    return len(missing) == 0, missing


def display_config(config: dict) -> None:
    """Display formatted configuration details."""
    print("\n" + "=" * 60)
    print("Configuration Details")
    print("=" * 60)

    # Policy type
    policy_type = config.get("type", "Unknown")
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


def download_checkpoint(repo: str, output_dir: str) -> None:
    """Download checkpoint, fix config, verify files, and display configuration."""
    # Download checkpoint
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading model files...")
    cmd = ["huggingface-cli", "download", repo] + CHECKPOINT_FILES + ["--local-dir", output_dir]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("Error: huggingface-cli download failed.", file=sys.stderr)
        sys.exit(1)

    # Fix config
    print("Fixing config.json (changing pi05 -> pi0)...")
    fix_config(output_dir)

    # Check files
    files_ok, missing = check_files(output_dir)

    # Load and display config
    config_path = Path(output_dir) / "config.json"
    print("Loading configuration...")
    with open(config_path) as f:
        config = json.load(f)

    display_config(config)

    # Final status
    print(f"\n{'=' * 60}")
    if files_ok:
        print(f"✓ Checkpoint successfully downloaded to: {output_dir}")
        print(f"  Use in your inference scripts: --policy.path={output_dir}")
    else:
        print("⚠ Warning: Some required files are missing:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download and setup a trained PI 0.5 checkpoint from HuggingFace")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g., username/PIv2)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: ./pi/ckpt/<repo-name>)")

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        repo_name = args.repo.split("/")[-1]
        output_dir = f"./pi/ckpt/{repo_name}"
    else:
        output_dir = args.output_dir

    download_checkpoint(args.repo, output_dir)


if __name__ == "__main__":
    main()
