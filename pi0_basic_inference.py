#!/usr/bin/env python3
"""
Pi0.5 Inference Script for SO-101 Robot

This script runs Pi0.5 policy inference with:
- Live camera feeds (3 cameras)
- Real robot state from SO-101 arm
- Continuous control loop with action execution
- Task/language instruction for the VLA model

Usage:
    python pi0_basic_inference.py --checkpoint ./pi05_fixed --task "Pick up the red lego." --duration 60

Hardware Setup:
    - Robot: SO-101 follower arm on /dev/ttyACM0
    - Cameras: main (/dev/video4), secondary_0 (/dev/video2), secondary_1 (/dev/video0)
"""

import argparse
import time
import cv2
import torch
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque

from lerobot.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.processor.pipeline import DataProcessorPipeline


# Default task for lego picking
DEFAULT_TASK = "Pick up the red lego and place it on the paper plate."


# =============================================================================
# POLICY AND PROCESSOR LOADING
# =============================================================================

def load_postprocessor(checkpoint_path: str, device: str = "cuda"):
    """
    Load the postprocessor pipeline that contains unnormalization stats.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load to
    
    Returns:
        Postprocessor pipeline, or None if not found
    """
    import os
    postprocessor_json = os.path.join(checkpoint_path, "policy_postprocessor.json")
    
    if not os.path.exists(postprocessor_json):
        print(f"  ⚠️  No policy_postprocessor.json found at {checkpoint_path}")
        return None
    
    try:
        postprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_path,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
        print(f"  ✓ Loaded postprocessor from {checkpoint_path}")
        
        # Check if it has unnormalization stats
        for step in postprocessor.steps:
            if hasattr(step, '_tensor_stats') and step._tensor_stats:
                print(f"    Stats loaded for: {list(step._tensor_stats.keys())}")
                # Show action stats in detail
                if 'action' in step._tensor_stats:
                    action_stats = step._tensor_stats['action']
                    print(f"    Action stats keys: {list(action_stats.keys())}")
                    for stat_name, stat_val in action_stats.items():
                        print(f"      {stat_name}: {stat_val.cpu().numpy()}")
        
        return postprocessor
    except Exception as e:
        print(f"  ⚠️  Failed to load postprocessor: {e}")
        return None


def load_policy_from_checkpoint(checkpoint_path: str, device: str = "cuda", n_action_steps: int = None):
    """
    Load a pretrained Pi0.5 policy from checkpoint directory.
    
    Args:
        checkpoint_path: Path to directory with config.json, model.safetensors, etc.
        device: "cuda" or "cpu"
    
    Returns:
        policy: Loaded policy ready for inference
    """
    print(f"Loading policy from {checkpoint_path}...")
    

    #  HERE?
    # from_pretrained orig_config = draccus.parse(cls, config_file, args=[])
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_path)


    policy_config.pretrained_path = checkpoint_path
    policy_config.device = device
    
    # Override n_action_steps if specified
    if n_action_steps is not None:
        policy_config.n_action_steps = n_action_steps
        print(f"  Overriding n_action_steps: {n_action_steps}")

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
    
    # Debug: Check if normalization stats are loaded
    if hasattr(policy, 'unnormalize_outputs'):
        stats = policy.unnormalize_outputs._tensor_stats
        if stats:
            print(f"  ✓ Unnormalization stats loaded for: {list(stats.keys())}")
            for key, stat_dict in stats.items():
                print(f"    {key}: {list(stat_dict.keys())}")
        else:
            print(f"  ⚠️  WARNING: No unnormalization stats loaded!")
            print(f"     Actions will NOT be denormalized properly.")
            print(f"     Make sure model.safetensors contains 'unnormalize_outputs.*' keys.")
    
    if hasattr(policy, 'normalize_inputs'):
        stats = policy.normalize_inputs._tensor_stats
        if stats:
            print(f"  ✓ Normalization stats loaded for: {list(stats.keys())}")
        else:
            print(f"  ⚠️  WARNING: No normalization stats loaded for inputs!")
    
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


def execute_action(robot: SO101Follower, action: np.ndarray, debug: bool = False) -> dict:
    """
    Send action to the robot.
    
    Args:
        robot: Connected SO101Follower instance
        action: numpy array of shape (6,) with target joint positions
        debug: If True, print debug info
    
    Returns:
        The action dict that was actually sent (may be clipped)
    """
    print("INSIDE execute_action")
    motor_names = list(robot.bus.motors.keys())
    action_dict = {f"{name}.pos": float(action[i]) for i, name in enumerate(motor_names)}
    print("action_dict: ", action_dict)

    
    if debug:
        # Get current position for comparison
        current_state = get_robot_state(robot)
        print(f"  Current state: {current_state}")
        print(f"  Requested action: {action}")
        print(f"  Diff: {action - current_state}")
    
    print("CALLING send_action")
    sent_action = robot.send_action(action_dict)
    print("send_action returned")
    
    if debug:
        print(f"  Sent action: {list(sent_action.values())}")
    
    return sent_action


# =============================================================================
# OBSERVATION BUILDING
# =============================================================================

def make_observation(
    policy,
    device: str,
    caps: dict,
    image_hw: tuple[int, int],
    robot: SO101Follower,
    task: str,
) -> dict:
    """
    Build observation dict from cameras and robot state.
    
    Args:
        policy: The loaded policy (for dtype)
        device: torch device
        caps: Dictionary of camera captures
        image_hw: Target (height, width) for images
        robot: Connected robot instance
        task: Task description for the VLA model
    
    Returns:
        Dictionary with image tensors, state tensor, and task, ready for policy
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
        "task": task,  # Pi0.5 needs task instruction
    }


# =============================================================================
# ASYNC INFERENCE
# =============================================================================

class AsyncInferenceRunner:
    """
    Runs model inference in a background thread for smooth robot control.
    
    While executing current action chunk, the next chunk is being generated.
    """
    
    def __init__(self, policy, device: str):
        self.policy = policy
        self.device = device
        self.action_queue = deque()  # Actions ready to execute
        self.pending_observation = None  # Observation waiting for inference
        self.inference_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.chunk_count = 0
        
    def start(self):
        """Start the async inference thread."""
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        print("[ASYNC] Inference thread started")
        
    def stop(self):
        """Stop the async inference thread."""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        print("[ASYNC] Inference thread stopped")
        
    def submit_observation(self, observation: dict):
        """Submit an observation for inference (non-blocking)."""
        with self.lock:
            self.pending_observation = observation
            
    def get_action(self) -> np.ndarray | None:
        """Get the next action (non-blocking). Returns None if no action available."""
        with self.lock:
            if len(self.action_queue) > 0:
                return self.action_queue.popleft()
            return None
    
    def queue_size(self) -> int:
        """Return number of actions waiting in queue."""
        with self.lock:
            return len(self.action_queue)
            
    def _inference_loop(self):
        """Background thread that runs inference when observations are available."""
        while self.running:
            # Check if we need more actions and have an observation
            with self.lock:
                need_actions = len(self.action_queue) < 10  # Start generating when queue gets low
                obs = self.pending_observation if need_actions else None
                if obs is not None:
                    self.pending_observation = None
                    
            if obs is None:
                time.sleep(0.01)  # Small sleep to avoid busy waiting
                continue
                
            # Run inference
            try:
                t0 = time.time()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=self.device.startswith("cuda"), dtype=torch.float16):
                        action = self.policy.select_action(obs)
                
                inference_time = time.time() - t0
                
                # Add action to queue
                with self.lock:
                    action_np = action.cpu().numpy().flatten()
                    self.action_queue.append(action_np)
                    self.chunk_count += 1
                    
                if self.chunk_count % 10 == 0:
                    print(f"[ASYNC] Queue: {len(self.action_queue)} actions, last inference: {inference_time*1000:.0f}ms")
                    
            except Exception as e:
                print(f"[ASYNC] Inference error: {e}")
                time.sleep(0.1)


# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================

def run_inference_loop(
    policy,
    device: str,
    caps: dict,
    image_hw: tuple[int, int],
    robot: SO101Follower,
    task: str,
    duration_s: float,
    fps: float,
    action_scale: float | None = None,
    postprocessor=None,
    async_inference: bool = False,
) -> None:
    """
    Main control loop: observe -> predict -> execute.
    
    Args:
        policy: Loaded Pi0.5 policy
        device: torch device string
        caps: Dictionary of camera captures
        image_hw: Target (height, width) for images
        robot: Connected SO101Follower instance
        task: Task description for the VLA model
        duration_s: How long to run (seconds)
        fps: Target control frequency
    """
    target_dt = 1.0 / fps
    start_time = time.time()
    step_count = 0
    
    # Reset policy before starting (clears action queue)
    if hasattr(policy, 'reset'):
        policy.reset()
        print("Policy reset.")
    
    print(f"\n{'='*60}")
    print(f"Starting inference loop")
    print(f"  Task: {task}")
    print(f"  Duration: {duration_s}s")
    print(f"  Target FPS: {fps}")
    print(f"  Async inference: {'✓ ENABLED' if async_inference else 'disabled'}")
    if postprocessor is not None:
        print(f"  Action unnormalization: ✓ Using postprocessor")
    elif action_scale is not None:
        print(f"  Action unnormalization: Manual scale {action_scale}x")
    else:
        print(f"  Action unnormalization: ⚠️  NONE - actions may be wrong scale!")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    # Setup async inference if enabled
    async_runner = None
    if async_inference:
        async_runner = AsyncInferenceRunner(policy, device)
        async_runner.start()
    
    # Get initial state for debugging
    initial_state = get_robot_state(robot)
    print(f"Initial robot state: {initial_state}")
    print()
    
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
                task=task,
            )

            # 2. Run policy inference (sync or async)
            if async_runner is not None:
                # Async mode: submit observation, get action from queue
                async_runner.submit_observation(observation)
                action_raw = async_runner.get_action()
                
                if action_raw is None:
                    # No action ready yet, wait a bit
                    time.sleep(0.01)
                    continue
            else:
                # Sync mode: blocking inference
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=device.startswith("cuda"), dtype=torch.float16):
                        action = policy.select_action(observation)
                action_raw = action.cpu().numpy().flatten()
            
            # Apply unnormalization via postprocessor if available
            if postprocessor is not None:
                # Find the unnormalizer step in the postprocessor
                unnormalizer = None
                for step in postprocessor.steps:
                    if hasattr(step, '_tensor_stats') and 'action' in step._tensor_stats:
                        unnormalizer = step
                        break
                
                if unnormalizer is not None:
                    # Get stats for action
                    stats = unnormalizer._tensor_stats.get('action', {})
                    if 'q01' in stats and 'q99' in stats:
                        # Policy outputs in [-1, 1] range
                        # QUANTILES unnormalization: (x + 1) / 2 * (q99 - q01) + q01
                        # This maps [-1, 1] -> [q01, q99]
                        q01 = stats['q01'].cpu().numpy()
                        q99 = stats['q99'].cpu().numpy()
                        action_01 = (action_raw + 1.0) / 2.0  # [-1,1] -> [0,1]
                        action_unnorm = action_01 * (q99 - q01) + q01  # [0,1] -> [q01,q99]
                        
                        # Stats are in radians, robot expects degrees
                        # Convert: 1 radian = 57.2958 degrees
                        action_np = action_unnorm * 57.2958
                        unnorm_method = "postprocessor (QUANTILES [-1,1]->rad->deg)"
                    elif 'mean' in stats and 'std' in stats:
                        # MEAN_STD unnormalization: x * std + mean
                        mean = stats['mean'].cpu().numpy()
                        std = stats['std'].cpu().numpy()
                        action_unnorm = action_raw * std + mean
                        # Convert radians to degrees
                        action_np = action_unnorm * 57.2958
                        unnorm_method = "postprocessor (MEAN_STD + rad->deg)"
                    else:
                        action_np = action_raw
                        unnorm_method = f"postprocessor (no matching stats, keys={list(stats.keys())})"
                else:
                    action_np = action_raw
                    unnorm_method = "postprocessor (no unnormalizer step found)"
            elif action_scale is not None:
                # Fallback: manual scaling
                action_np = action_raw * action_scale
                unnorm_method = f"manual scale {action_scale}x"
            else:
                # No unnormalization
                action_np = action_raw
                unnorm_method = "none (raw)"
            
            # Debug: print state and action on first few steps
            if step_count < 5:
                print(f"\n--- Step {step_count} ---")
                print(f"  Unnormalization: {unnorm_method}")
                print(f"  Raw action: {action_raw}")
                print(f"  Processed action: {action_np}")
                print("CALLING execute_action")
                execute_action(robot, action_np, debug=True)
            else:
                print("CALLING execute_action")
                execute_action(robot, action_np, debug=False)
            
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
                current_state = get_robot_state(robot)
                print(
                    f"Step {step_count:4d} | "
                    f"dt: {actual_dt*1000:5.1f}ms ({actual_hz:5.1f}Hz) | "
                    f"state: {current_state} | "
                    f"Elapsed: {elapsed:5.1f}s / {duration_s}s"
                )
    
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user (Ctrl+C)")
    
    finally:
        # Stop async runner if enabled
        if async_runner is not None:
            async_runner.stop()
            
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
    
    # Task
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Task description for the VLA model",
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
    
    # Action scaling (workaround when normalization stats are missing)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=None,
        help="Manual action scaling factor. If policy outputs [-1,1] and robot expects [-100,100], use 100. "
             "Set to None to use policy's built-in unnormalization (requires stats in model).",
    )
    
    # Action chunking
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Number of actions per chunk. Default is 50. Lower = more frequent inference, smoother but slower. "
             "Higher = less frequent inference, faster but jerkier during chunk transitions.",
    )
    
    # Async inference
    parser.add_argument(
        "--async",
        dest="async_inference",
        action="store_true",
        help="Enable async inference (run model in background thread for smoother motion)",
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
    policy = load_policy_from_checkpoint(
        args.checkpoint, 
        device=args.device,
        n_action_steps=args.n_action_steps
    )
    
    # Load postprocessor (contains unnormalization stats)
    print("\nLoading postprocessor...")
    postprocessor = load_postprocessor(args.checkpoint, device=args.device)
    
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
            task=args.task,
            duration_s=args.duration,
            fps=args.fps,
            action_scale=args.action_scale,
            postprocessor=postprocessor,
            async_inference=args.async_inference,
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
