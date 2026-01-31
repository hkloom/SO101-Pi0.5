#!/usr/bin/env python3
"""
PI 0.5 Training Script for Lambda Cloud
Trains on dataset converted to the new v3.0 quantile format
"""

import subprocess
import sys
import os

# =========================================================================== #
#                    Configuration - Edit values as needed                    #
# =========================================================================== #

CONFIG = {
    # Dataset (pre-converted v3.0 format)
    "dataset_repo_id": "bdhillon/PI-01.07.26-v3-quantiles",
    "dataset_root": os.path.expanduser("~/lerobot-training/dataset"),
    # uncomment this line if using a local dataset instead of a dataset from HF
    # "dataset_root": os.path.expanduser("~/lerobot-training/dataset/PI-0.5-11.19.2025-v3-quantiles"),

    # Model
    "policy_type": "pi05",
    "pretrained_path": "lerobot/pi05_base",

    # HuggingFace upload settings
    "repo_id": "bdhillon/PIv3",
    "push_to_hub": True,

    # Training hyperparameters
    "batch_size": 8,
    "policy.dtype": "bfloat16",
    "policy.use_amp": True,
    "steps": 7350,
    "eval_freq": 500,
    "log_freq": 100,
    "save_freq": 500,

    # Evaluation settings
    "eval_n_episodes": 5,
    "eval_batch_size": 5,    # Must be <= eval_n_episodes

    # Output
    "output_dir": "./PIv3",

    # Logging
    "wandb_enable": True,
}

# =========================================================================== #
#                        Build Training Command & Run                         #
# =========================================================================== #

def build_command(config):
    """Build the lerobot-train command from config."""

    cmd = ["lerobot-train"]

    # Policy settings
    cmd.append(f"--policy.type={config['policy_type']}")
    cmd.append(f"--policy.pretrained_path={config['pretrained_path']}")
    cmd.append(f"--policy.repo_id={config['repo_id']}")
    cmd.append(f"--policy.push_to_hub={'true' if config['push_to_hub'] else 'false'}")

    # Dataset settings
    cmd.append(f"--dataset.repo_id={config['dataset_repo_id']}")
    cmd.append(f"--dataset.root={config['dataset_root']}")

    # Training hyperparameters
    cmd.append(f"--batch_size={config['batch_size']}")
    cmd.append(f"--steps={config['steps']}")
    cmd.append(f"--eval_freq={config['eval_freq']}")
    cmd.append(f"--log_freq={config['log_freq']}")
    cmd.append(f"--save_freq={config['save_freq']}")

    # Evaluation settings
    cmd.append(f"--eval.n_episodes={config['eval_n_episodes']}")
    cmd.append(f"--eval.batch_size={config['eval_batch_size']}")

    # Output
    cmd.append(f"--output_dir={config['output_dir']}")

    # WandB
    cmd.append(f"--wandb.enable={'true' if config['wandb_enable'] else 'false'}")

    return cmd


def main():

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Print configuration
    print("=" * 60)
    print("LeRobot PI 0.5 Training")
    print("=" * 60)
    print("\nConfiguration:")

    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    print()

    # Build command
    cmd = build_command(CONFIG)

    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print()

    # Run training
    try:
        result = subprocess.run(cmd, check=True)

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"\nModel saved to: {CONFIG['output_dir']}")

        if CONFIG['push_to_hub']:
            print(f"Model uploaded to: https://huggingface.co/{CONFIG['repo_id']}")

    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
    