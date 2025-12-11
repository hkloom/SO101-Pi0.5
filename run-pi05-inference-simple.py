#!/usr/bin/env python3
"""
Simple inference script for PI 0.5 policy on SO-101 robot.
This provides a cleaner interface for running inference without recording to a dataset.

Usage:
    python run-pi05-inference-simple.py --task "Your task description here" --duration 60
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.pipeline_features import create_initial_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(
    policy_path: str = "./pi05_fixed",
    robot_port: str = "/dev/ttyACM0",
    task: str = "Pick up the red lego and place it on the paper plate, please.",
    duration_s: float = 60.0,
    fps: int = 30,
    display: bool = True,
):
    """
    Run PI 0.5 policy inference on SO-101 robot.

    Args:
        policy_path: HuggingFace model path for the policy
        robot_port: Serial port for the SO-101 robot
        task: Task description for the policy
        duration_s: Duration to run inference (seconds)
        fps: Target control frequency
        display: Whether to display camera feeds
    """
    init_logging()

    # Configure robot with 3 cameras
    robot_config = SO101FollowerConfig(
        port=robot_port,
        max_relative_target=0.05,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path="/dev/video0",
                width=640,
                height=480,
                fps=fps,
            ),
            "side": OpenCVCameraConfig(
                index_or_path="/dev/video2",
                width=640,
                height=480,
                fps=fps,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/video4",
                width=640,
                height=480,
                fps=fps,
            ),
        },
    )

    logger.info(f"Loading policy from {policy_path}...")
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path

    # Get device
    device = get_safe_torch_device(policy_config.device, log=True)

    # Create policy
    logger.info("Initializing policy...")
    policy = make_policy(cfg=policy_config, env_cfg=None)
    policy.eval()

    # Create preprocessors and postprocessors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Initialize robot
    logger.info("Connecting to robot...")
    robot = SO101Follower(robot_config)
    robot.connect()

    # Get initial observation to determine action space
    obs = robot.get_observation()

    # Create features for observation formatting
    features = create_initial_features(
        robot_type="so101_follower",
        observation_space=robot.observation_space,
        action_space=robot.action_space,
    )

    # Reset policy
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    logger.info(f"Starting inference for {duration_s}s...")
    logger.info(f"Task: {task}")

    start_time = time.time()
    step_count = 0
    target_dt = 1.0 / fps

    try:
        while (time.time() - start_time) < duration_s:
            step_start = time.time()

            # Get observation from robot
            obs = robot.get_observation()

            # Build observation frame
            observation_frame = build_dataset_frame(features, obs, prefix="observation")

            # Predict action
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy_config.use_amp,
                task=task,
                robot_type="so101_follower",
            )

            # Convert action tensor to robot action dict
            action_names = features["action"]["names"]
            action_dict = {
                name: float(action_tensor[i])
                for i, name in enumerate(action_names)
            }

            # Send action to robot
            robot.send_action(action_dict)

            step_count += 1

            # Maintain target FPS
            step_dt = time.time() - step_start
            if step_dt < target_dt:
                time.sleep(target_dt - step_dt)

            actual_dt = time.time() - step_start
            if step_count % 30 == 0:  # Log every second
                logger.info(
                    f"Step {step_count} | "
                    f"dt: {actual_dt*1000:.1f}ms ({1/actual_dt:.1f}Hz) | "
                    f"Elapsed: {time.time()-start_time:.1f}s"
                )

    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise
    finally:
        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info(f"Inference complete. Total steps: {step_count}")


def main():
    parser = argparse.ArgumentParser(description="Run PI 0.5 inference on SO-101")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="./pi05_fixed",
        help="Local or HuggingFace model path",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyACM0",
        help="Robot serial port",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the red lego and place it on the paper plate, please.",
        help="Task description",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable camera display",
    )

    args = parser.parse_args()

    run_inference(
        policy_path=args.policy_path,
        robot_port=args.robot_port,
        task=args.task,
        duration_s=args.duration,
        fps=args.fps,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
