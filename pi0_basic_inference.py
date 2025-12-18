#!/usr/bin/env python3
"""
Pi0.5 Inference Script for SO-101 Robot

This script runs Pi0.5 policy inference with:
- Live camera feeds (3 cameras)
- Real robot state from SO-101 arm
- Continuous control loop with action execution

Usage:
    python pi0_basic_inference.py --checkpoint ./pi05_fixed --duration 60

Hardware Setup:
    - Robot: SO-101 follower arm on /dev/ttyACM0
    - Cameras: main (/dev/video4), secondary_0 (/dev/video2), secondary_1 (/dev/video0)
"""

import argparse
import time
import cv2
import torch
import numpy as np

from lerobot.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


# =============================================================================
# POLICY LOADING
# =============================================================================

def load_policy_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load a pretrained Pi0.5 policy from checkpoint directory.
    
    Args:
        checkpoint_path: Path to directory with config.json, model.safetensors, etc.
        device: "cuda" or "cpu"
    
    Returns:
        policy: Loaded policy ready for inference
    """
    print(f"Loading policy from {checkpoint_path}...")
    
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_path)
    policy_config.pretrained_path = checkpoint_path
    policy_config.device = device

    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=checkpoint_path,
        config=policy_config
    )
    
    # Use half precision on CUDA to reduce VRAM
    if device.startswith("cuda"):
        policy = policy.half()
    
    policy = policy.to(device)
    policy.eval()
    
    print(f"  Policy type: {policy_config.type}")
    print(f"  Device: {device}")
    
    return policy


# =============================================================================
# CAMERA FUNCTIONS
# =============================================================================

def open_camera(path: str, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera and configure resolution."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not available: {path}")
    print(f"  Opened camera: {path}")
    return cap


def get_frame(cap: cv2.VideoCapture, target_hw: tuple[int, int]) -> torch.Tensor:
    """
    Read a frame from camera and convert to tensor.
    
    Returns:
        Tensor of shape (3, H, W) with values in [0, 1]
    """
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # cv2.resize expects (width, height); target_hw is (H, W)
    rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    # Convert to (C, H, W) tensor normalized to [0, 1]
    chw = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return chw


# =============================================================================
# ROBOT FUNCTIONS
# =============================================================================

def connect_robot(port: str, max_relative_target: float = 0.05) -> SO101Follower:
    """
    Connect to the SO-101 robot arm.
    
    Args:
        port: Serial port (e.g., "/dev/ttyACM0")
        max_relative_target: Safety limit for movement speed
    
    Returns:
        Connected robot instance
    """
    print(f"Connecting to robot on {port}...")
    
    config = SO101FollowerConfig(
        port=port,
        id="so101_follower",
        max_relative_target=max_relative_target,
    )
    
    robot = SO101Follower(config)
    robot.connect()
    
    print(f"  Robot connected!")
    print(f"  Motors: {list(robot.bus.motors.keys())}")
    
    return robot


def get_robot_state(robot: SO101Follower) -> np.ndarray:
    """
    Read current joint positions from the robot.
    
    Returns:
        numpy array of shape (6,) with joint positions
    """
    obs = robot.get_observation()
    # Extract just the motor positions (not camera images)
    motor_names = list(robot.bus.motors.keys())
    state = np.array([obs[f"{name}.pos"] for name in motor_names], dtype=np.float32)
    return state


def execute_action(robot: SO101Follower, action: np.ndarray) -> None:
    """
    Send action to the robot.
    
    Args:
        robot: Connected SO101Follower instance
        action: numpy array of shape (6,) with target joint positions
    """
    motor_names = list(robot.bus.motors.keys())
    action_dict = {f"{name}.pos": float(action[i]) for i, name in enumerate(motor_names)}
    robot.send_action(action_dict)


# =============================================================================
# OBSERVATION BUILDING
# =============================================================================

def make_observation(
    policy,
    device: str,
    caps: dict,
    image_hw: tuple[int, int],
    robot: SO101Follower,
) -> dict:
    """
    Build observation dict from cameras and robot state.
    
    Returns:
        Dictionary with image tensors and state tensor, ready for policy
    """
    dtype = next(policy.parameters()).dtype
    
    # Get camera frames
    main_img = get_frame(caps["main"], image_hw).unsqueeze(0).to(device).to(dtype)
    sec0_img = get_frame(caps["secondary_0"], image_hw).unsqueeze(0).to(device).to(dtype)
    sec1_img = get_frame(caps["secondary_1"], image_hw).unsqueeze(0).to(device).to(dtype)
    
    # Get robot state
    state = get_robot_state(robot)
    state_tensor = torch.tensor(state, device=device, dtype=dtype).view(1, -1)
    
    return {
        "observation.images.main": main_img,
        "observation.images.secondary_0": sec0_img,
        "observation.images.secondary_1": sec1_img,
        "observation.state": state_tensor,
    }


# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================

def run_inference_loop(
    policy,
    device: str,
    caps: dict,
    image_hw: tuple[int, int],
    robot: SO101Follower,
    duration_s: float,
    fps: float,
) -> None:
    """
    Main control loop: observe -> predict -> execute.
    
    Args:
        policy: Loaded Pi0.5 policy
        device: torch device string
        caps: Dictionary of camera captures
        image_hw: Target (height, width) for images
        robot: Connected SO101Follower instance
        duration_s: How long to run (seconds)
        fps: Target control frequency
    """
    target_dt = 1.0 / fps
    start_time = time.time()
    step_count = 0
    
    print(f"\n{'='*60}")
    print(f"Starting inference loop")
    print(f"  Duration: {duration_s}s")
    print(f"  Target FPS: {fps}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    try:
        while (time.time() - start_time) < duration_s:
            step_start = time.time()
            
            # 1. Build observation from cameras + robot
            observation = make_observation(
                policy=policy,
                device=device,
                caps=caps,
                image_hw=image_hw,
                robot=robot,
            )
            
            print(f"Observation: {observation}")

            # 2. Run policy inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=device.startswith("cuda"),
                    dtype=torch.float16
                ):
                    action = policy.select_action(observation)
            
            # 3. Execute action on robot
            action_np = action.cpu().numpy().flatten()

            print(f"Action: {action_np}")
            
            execute_action(robot, action_np)
            
            step_count += 1
            
            # 4. Maintain target FPS
            step_dt = time.time() - step_start
            if step_dt < target_dt:
                time.sleep(target_dt - step_dt)
            
            # 5. Log performance
            actual_dt = time.time() - step_start
            if step_count % int(fps) == 0:  # Log every second
                elapsed = time.time() - start_time
                actual_hz = 1.0 / actual_dt if actual_dt > 0 else 0
                print(
                    f"Step {step_count:4d} | "
                    f"dt: {actual_dt*1000:5.1f}ms ({actual_hz:5.1f}Hz) | "
                    f"Elapsed: {elapsed:5.1f}s / {duration_s}s"
                )
    
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user (Ctrl+C)")
    
    finally:
        elapsed = time.time() - start_time
        avg_hz = step_count / elapsed if elapsed > 0 else 0
        print(f"\n{'='*60}")
        print(f"Inference complete!")
        print(f"  Total steps: {step_count}")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Average Hz: {avg_hz:.1f}")
        print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pi0.5 inference on SO-101 robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., ./pi05_fixed)",
    )
    
    # Robot
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO-101 robot",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=0.05,
        help="Safety limit for movement speed (lower = slower/safer)",
    )
    
    # Control loop
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration to run inference (seconds)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target control frequency (Hz)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    
    # Images
    parser.add_argument(
        "--image-height",
        type=int,
        default=240,
        help="Image height for policy input",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=320,
        help="Image width for policy input",
    )
    
    # Cameras
    parser.add_argument(
        "--main-cam",
        type=str,
        default="/dev/video4",
        help="Device path for main camera",
    )
    parser.add_argument(
        "--secondary0-cam",
        type=str,
        default="/dev/video2",
        help="Device path for secondary_0 camera",
    )
    parser.add_argument(
        "--secondary1-cam",
        type=str,
        default="/dev/video0",
        help="Device path for secondary_1 camera",
    )

    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    print("="*60)
    print("Pi0.5 Inference for SO-101")
    print("="*60)
    
    # Load policy
    policy = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    
    # Open cameras
    print("\nOpening cameras...")
    image_hw = (args.image_height, args.image_width)
    caps = {
        "main": open_camera(args.main_cam, args.image_width, args.image_height),
        "secondary_0": open_camera(args.secondary0_cam, args.image_width, args.image_height),
        "secondary_1": open_camera(args.secondary1_cam, args.image_width, args.image_height),
    }
    
    # Connect to robot
    print()
    robot = connect_robot(args.robot_port, args.max_relative_target)
    
    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    try:
        run_inference_loop(
            policy=policy,
            device=args.device,
            caps=caps,
            image_hw=image_hw,
            robot=robot,
            duration_s=args.duration,
            fps=args.fps,
        )
    finally:
        # Cleanup
        print("\nCleaning up...")
        for cap in caps.values():
            cap.release()
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
