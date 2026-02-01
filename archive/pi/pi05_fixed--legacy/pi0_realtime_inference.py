#!/usr/bin/env python3
"""
Real-Time Pi0.5 Inference Loop

This script demonstrates a real-time inference loop for continuous control.
You need to implement get_camera_image() and get_robot_state() functions
for your specific robot setup.

Usage:
    python pi0_realtime_inference.py --checkpoint path/to/checkpoint/directory
"""

import argparse
import time
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


def get_camera_image():
    """
    Get image from camera.
    
    Returns:
        image: numpy array (H, W, 3) uint8 with values 0-255
    """
    # TODO: Implement your camera capture function
    # Example: return cv2.imread("image.jpg")
    # For now, return dummy data
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


def get_robot_state():
    """
    Get current robot state.
    
    Returns:
        state: numpy array (state_dim,) float
    """
    # TODO: Implement your robot state reading function
    # For now, return dummy data
    return np.random.randn(7).astype(np.float32)


def execute_robot_action(action):
    """
    Execute action on robot.
    
    Args:
        action: numpy array of actions to execute
    """
    # TODO: Implement your robot action execution function
    print(f"Executing action: {action}")


def main():
    parser = argparse.ArgumentParser(description="Real-time Pi0.5 inference loop")
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
        "--fps",
        type=float,
        default=30.0,
        help="Target control frequency (Hz)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration to run inference (seconds)",
    )

    args = parser.parse_args()

    # Load policy once
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    policy.eval()

    target_dt = 1.0 / args.fps
    start_time = time.time()
    step_count = 0

    print(f"Starting real-time inference loop (target FPS: {args.fps})...")
    print("Press Ctrl+C to stop")

    try:
        while (time.time() - start_time) < args.duration:
            step_start = time.time()

            # 1. Get observation from your robot/camera
            image = get_camera_image()  # Returns (H, W, 3) uint8
            state = get_robot_state()   # Returns (state_dim,) float
            
            # 2. Convert to tensors
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            # 3. Move to device
            observation = {
                "observation.image": image_tensor.to(policy.device),
                "observation.state": state_tensor.to(policy.device),
            }
            
            # 4. Get action
            with torch.no_grad():
                action = policy.select_action(observation)
            
            # 5. Execute action on robot
            execute_robot_action(action.cpu().numpy())

            step_count += 1

            # Maintain target FPS
            step_dt = time.time() - step_start
            if step_dt < target_dt:
                time.sleep(target_dt - step_dt)

            actual_dt = time.time() - step_start
            if step_count % int(args.fps) == 0:  # Log every second
                print(
                    f"Step {step_count} | "
                    f"dt: {actual_dt*1000:.1f}ms ({1/actual_dt:.1f}Hz) | "
                    f"Elapsed: {time.time()-start_time:.1f}s"
                )

    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    finally:
        print(f"Inference complete. Total steps: {step_count}")


if __name__ == "__main__":
    main()
