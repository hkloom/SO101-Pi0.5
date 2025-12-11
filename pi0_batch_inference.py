#!/usr/bin/env python3
"""
Pi0.5 Batch Inference Script

This script demonstrates batch processing of multiple observations at once,
which is more efficient than processing them individually.

Usage:
    python pi0_batch_inference.py --checkpoint path/to/checkpoint/directory --batch-size 4
"""

import argparse
import torch
import numpy as np
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig


def load_policy_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load a pretrained policy from checkpoint directory.
    This helper function matches the tutorial's simpler API.
    """
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_path)
    policy_config.pretrained_path = checkpoint_path
    policy = make_policy(cfg=policy_config, env_cfg=None)
    policy = policy.to(device)
    return policy


def batch_inference(policy, images, states):
    """
    Process multiple observations at once.
    
    Args:
        policy: Loaded Pi0 policy
        images: numpy array (batch, H, W, 3) or (batch, 3, H, W)
        states: numpy array (batch, state_dim)
    
    Returns:
        actions: numpy array (batch, action_dim)
    """
    # Convert to tensors
    images_tensor = torch.from_numpy(images).float()
    
    # Rearrange if needed
    if images_tensor.shape[-1] == 3:
        images_tensor = images_tensor.permute(0, 3, 1, 2)
    
    states_tensor = torch.from_numpy(states).float()
    
    # Move to device
    observation = {
        "observation.image": images_tensor.to(policy.device),
        "observation.state": states_tensor.to(policy.device),
    }
    
    # Get actions
    with torch.no_grad():
        actions = policy.select_action(observation)
    
    return actions.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Batch Pi0.5 inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory containing config.json, model.safetensors, etc.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of observations to process in batch",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=224,
        help="Image height",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=224,
        help="Image width",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=7,
        help="State dimension",
    )

    args = parser.parse_args()

    # Load policy
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    policy.eval()

    # Create batch of random inputs
    # In real usage, you would load actual images and states
    images = np.random.randint(
        0, 255,
        (args.batch_size, args.image_height, args.image_width, 3),
        dtype=np.uint8
    )
    states = np.random.randn(args.batch_size, args.state_dim)

    print(f"\nProcessing batch of {args.batch_size} observations...")
    print(f"Images shape: {images.shape}")
    print(f"States shape: {states.shape}")

    # Process batch
    actions = batch_inference(policy, images, states)

    print(f"\nBatch actions shape: {actions.shape}")
    print(f"Actions:\n{actions}")


if __name__ == "__main__":
    main()
