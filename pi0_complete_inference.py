#!/usr/bin/env python3
"""
Complete Pi0.5 Inference Function

This script provides a complete inference function that handles image and state
preprocessing, policy loading, and action prediction.

Usage:
    python pi0_complete_inference.py --checkpoint path/to/checkpoint/directory
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


def run_pi0_inference(checkpoint_path, image, state, device="cuda"):
    """
    Run inference with Pi0.5 policy.
    
    Args:
        checkpoint_path: Path to directory containing checkpoint files
        image: numpy array (H, W, 3) with values 0-255 or (3, H, W) float
        state: numpy array (state_dim,)
        device: "cuda" or "cpu"
    
    Returns:
        action: numpy array of predicted actions
    """
    # Load policy
    policy = load_policy_from_checkpoint(checkpoint_path, device=device)
    policy.eval()
    
    # Convert image to tensor
    image_tensor = torch.from_numpy(image).float()
    
    # Rearrange to (C, H, W) if needed
    if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    
    # Create observation dict
    observation = {
        "observation.image": image_tensor,
        "observation.state": state_tensor,
    }
    
    # Get action
    with torch.no_grad():
        action = policy.select_action(observation)
    
    return action.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Complete Pi0.5 inference")
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

    # Example random inputs
    # In real usage, you would load actual images and states
    image = np.random.randint(0, 255, (args.image_height, args.image_width, 3), dtype=np.uint8)
    state = np.random.randn(args.state_dim)
    
    print(f"Running inference with checkpoint: {args.checkpoint}")
    print(f"Image shape: {image.shape}, State shape: {state.shape}")
    
    action = run_pi0_inference(args.checkpoint, image, state, device=args.device)
    
    print(f"Predicted action shape: {action.shape}")
    print(f"Predicted action: {action}")


if __name__ == "__main__":
    main()
