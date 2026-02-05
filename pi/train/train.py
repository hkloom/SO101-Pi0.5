#!/usr/bin/env python3
"""
PI 0.5 Training Script for Lambda Cloud
Trains on dataset converted to the new v3.0 quantile format

When you run this script it will build and run the following command:

lerobot-train \
    --dataset.repo_id=bdhillon/PI-01.07.26-v3-quantiles \
    --dataset.root=/home/ubuntu/lerobot-training/dataset \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.repo_id=bdhillon/PIv3 \
    --policy.push_to_hub=true \
    --policy.normalization_mapping={"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "NONE"} \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --batch_size=8 \
    --steps=7350 \
    --eval_freq=500 \
    --log_freq=100 \
    --save_freq=500 \
    --eval.n_episodes=5 \
    --eval.batch_size=5 \
    --output_dir=./PIv3 \
    --wandb.enable=true
"""

import subprocess
import sys
import os

# =========================================================================== #
#                    Configuration - Edit values as needed                    #
# =========================================================================== #

CONFIG = {
    # Dataset (pre-converted v3.0 format with quantile stats)
    "dataset_repo_id": "bdhillon/PI-01.07.26-v3-quantiles",
    "dataset_root": os.path.expanduser("~/lerobot-training/dataset"),

    # Model
    "policy_type": "pi05",
    "pretrained_path": "lerobot/pi05_base",

    # HuggingFace upload settings
    "repo_id": "bdhillon/PIv3",
    "push_to_hub": True,

    # This tells LeRobot how to normalize inputs and denormalize outputs.
    # The stats will be saved with the checkpoint for use during inference.
    #
    # Options:
    #   - For datasets WITH quantile stats:  {"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "NONE"}
    #   - For datasets WITHOUT quantile stats: {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}
    # ==========================================================================
    "normalization_mapping": '{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "NONE"}',

    # Training hyperparameters
    "batch_size": 8,
    "steps": 7350,
    "eval_freq": 500,
    "log_freq": 100,
    "save_freq": 500,

    # Policy settings
    "compile_model": True,
    "gradient_checkpointing": True,
    "dtype": "bfloat16",

    # Evaluation settings
    "eval_n_episodes": 5,
    "eval_batch_size": 5,

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

    # Dataset settings
    cmd.append(f"--dataset.repo_id={config['dataset_repo_id']}")
    cmd.append(f"--dataset.root={config['dataset_root']}")

    # Policy settings
    cmd.append(f"--policy.type={config['policy_type']}")
    cmd.append(f"--policy.pretrained_path={config['pretrained_path']}")
    cmd.append(f"--policy.repo_id={config['repo_id']}")
    cmd.append(f"--policy.push_to_hub={'true' if config['push_to_hub'] else 'false'}")

    # Normalization mapping
    if 'normalization_mapping' in config:
        cmd.append(f"--policy.normalization_mapping={config['normalization_mapping']}")

    # Policy optimization settings
    if config.get('compile_model'):
        cmd.append("--policy.compile_model=true")
    if config.get('gradient_checkpointing'):
        cmd.append("--policy.gradient_checkpointing=true")
    if config.get('dtype'):
        cmd.append(f"--policy.dtype={config['dtype']}")

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
