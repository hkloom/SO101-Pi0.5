#!/bin/bash
# Inference script with automatic reset to starting position
# Uses the eval-with-reset.py wrapper for automatic return to start

python eval-with-reset.py \
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
    --dataset.repo_id=local/pi05_inference_with_reset \
    --dataset.single_task="Pick up the red lego and place it on the paper plate, please." \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=false \
    --dataset.root=./inference_data \
    --policy.path=bdhillon/PIv2
