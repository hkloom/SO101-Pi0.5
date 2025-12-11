#!/usr/bin/env python3
"""
Basic Pi0.5 Inference Script

This script demonstrates loading and inference with a Pi0.5 policy checkpoint
using live camera feeds and robot state.

Usage:
    python pi0_basic_inference.py --checkpoint path/to/checkpoint/directory
"""

import argparse
import cv2
import torch
import numpy as np
from lerobot.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig


def load_policy_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load a pretrained policy from checkpoint directory.
    This helper function matches the tutorial's simpler API.
    """
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_path)
    policy_config.pretrained_path = checkpoint_path
    policy_config.device = device

    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=checkpoint_path, config=policy_config)
    # Use half precision only when running on CUDA to reduce VRAM.
    if device.startswith("cuda"):
        policy = policy.half()
    policy = policy.to(device)
    return policy


def open_cam(path: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not available: {path}")
    return cap


def get_frame(cap: cv2.VideoCapture, target_hw: tuple[int, int]) -> torch.Tensor:
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # cv2.resize expects (width, height); target_hw is (H, W)
    rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    chw = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return chw


def get_robot_state(state_dim: int) -> np.ndarray:
    """
    TODO: Replace with real robot state from /dev/ttyACM0.
    Currently returns zeros of the expected length as a placeholder.
    """
    return np.zeros(state_dim, dtype=np.float32)


def make_live_observation(policy, device, caps, image_hw, state_dim):
    """
    Build an observation using live camera frames and robot state.
    """
    dtype = next(policy.parameters()).dtype
    return {
        "observation.images.main": get_frame(caps["main"], image_hw).unsqueeze(0).to(device).to(dtype),
        "observation.images.secondary_0": get_frame(caps["secondary_0"], image_hw).unsqueeze(0).to(device).to(dtype),
        "observation.images.secondary_1": get_frame(caps["secondary_1"], image_hw).unsqueeze(0).to(device).to(dtype),
        "observation.state": torch.tensor(get_robot_state(state_dim), device=device, dtype=dtype).view(1, -1),
    }


def main():
    parser = argparse.ArgumentParser(description="Basic Pi0.5 inference")
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
        default=240,
        help="Image height",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=320,
        help="Image width",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=6,
        help="State dimension",
    )
    parser.add_argument(
        "--main-cam",
        type=str,
        default="/dev/video4",  # Innomaker-U20CAM-720P
        help="Device path for main camera (Innomaker)",
    )
    parser.add_argument(
        "--secondary0-cam",
        type=str,
        default="/dev/video2",  # C922 Pro
        help="Device path for secondary_0 camera (C922 Pro)",
    )
    parser.add_argument(
        "--secondary1-cam",
        type=str,
        default="/dev/video0",  # C270
        help="Device path for secondary_1 camera (C270)",
    )

    args = parser.parse_args()

    # Load the policy (automatically loads all files)
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    policy.eval()

    image_hw = (args.image_height, args.image_width)
    caps = {
        "main": open_cam(args.main_cam, args.image_width, args.image_height),
        "secondary_0": open_cam(args.secondary0_cam, args.image_width, args.image_height),
        "secondary_1": open_cam(args.secondary1_cam, args.image_width, args.image_height),
    }

    try:
        observation = make_live_observation(
            policy=policy,
            device=args.device,
            caps=caps,
            image_hw=image_hw,
            state_dim=args.state_dim,
        )
        print("Observation:")
        print(observation)

        # Run inference
        print("Running inference...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda"), dtype=torch.float16):
                action = policy.select_action(observation)
    finally:
        for cap in caps.values():
            cap.release()

    print(f"Action shape: {action.shape}")
    print(f"Action: {action.cpu().numpy()}")


if __name__ == "__main__":
    main()
