#!/bin/bash
# basic.sh
# - Shell wrapper that calls LeRobot's built-in record module with the right flags for our hardware
# - Does not use our `async.py` at all, this alternative approach delegates to LeRobot's framework
# - The key flag is --policy.path=./ckpt/PIv3. When lerobot.record sees a policy path, it switches from teleoperation to policy inference
#
# lerobot.record (src/lerobot/record.py)
# - LeRobot's general-purpose control loop at src/lerobot/record.py:235 
# - It's designed for both data collection (teleop) and policy execution
# - Core loop (record_loop) on each tick:
#   1. robot.get_observation() → raw camera frames + joint state
#   2. Runs observations through a processor pipeline
#   3. Gets actions from either a teleoperator (human) or a policy (model inference via predict_action)
#   4. Runs actions through a processor pipeline (unnormalization, clipping)
#   5. robot.send_action() → moves the robot
#   6. Saves the frame to a LeRobotDataset
#
# How it compares to async.py
# - infer/async.py is a stripped-down version of the same idea — observe, infer, execute
# - With infer/async.py we bypass LeRobot's dataset/recording infrastructure and add async inference threading
# - With infer/basic.sh (this file) we use LeRobot's full pipeline, which gives us dataset recording but doesn't support async inference

python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower \
    --robot.max_relative_target=0.05 \
    --robot.cameras='{
        main: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30},
        secondary_0: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30},
        secondary_1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}
    }' \
    --display_data=true \
    --dataset.repo_id=local/pi05_inference_test \
    --dataset.single_task="Pick up the red lego and place it on the paper plate, please." \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=false \
    --dataset.root=./inference_data \
    --policy.path=./ckpt/PIv3
