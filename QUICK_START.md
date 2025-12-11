# Quick Start Guide - PI 0.5 Inference on SO-101

inference files:
- /home/aiclub/dev/lerobot/pi05_fixed

source ~/miniforge3/bin/activate lerobot

python -m lerobot.record \ 
  --robot.type=so101_follower \ 
  --robot.port=/dev/ttyACM0 \ 
  --robot.id=so101_follower \ 
  --robot.max_relative_target=0.05 \ 
  --robot.cameras='{main: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, secondary_0: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, secondary_1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.repo_id=local/pi05_inference_test --dataset.single_task="Pick up the red lego and place it on the paper plate, please." \
   --dataset.num_episodes=1 \ 
   --dataset.episode_time_s=60 \ 
   --dataset.reset_time_s=30 \ 
   --dataset.push_to_hub=false \ 
   --dataset.root=./inference_data 
   --policy.path=./pi05_fixed
   
   
   
## New infernece command

### Reason:
  1. Policy (the trained model): --policy.path=./pi05_fixed
    - This is your trained model from bdhillon/PIv2
    - Contains the 14GB model weights
    - This was trained on bdhillon/PI-0.5-11.19.2025-v3-quantiles
    
  2. Dataset for saving inference data: --dataset.repo_id=eval_pi05_inference_test --dataset.root=./inference_data
    - This is where lerobot will SAVE the robot's actions during inference
    - It's a NEW local dataset that records what the robot does
    - The eval_ prefix is just a naming convention



 source ~/miniforge3/bin/activate lerobot
 
 python -m lerobot.record 
 
 --robot.type=so101_follower 
 --robot.port=/dev/ttyACM0 
 --robot.id=so101_follower 
 --robot.max_relative_target=0.05 
 --robot.cameras='{main: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, secondary_0: {type: opencv,
   index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, secondary_1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' 
 --display_data=true 
 --dataset.repo_id=eval_pi05_inference_test 
 --dataset.single_task="Pick up the red lego and place it on the paper plate, please."
 --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.push_to_hub=false --dataset.root=./inference_data --policy.path="./pi05_fixed"
   Run inference with eval_ dataset prefix
   
   
   
## Latest:

 Run PI 0.5 inference after authentication

source ~/miniforge3/bin/activate lerobot && python -m lerobot.record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower --robot.max_relative_target=0.05 --robot.cameras='{main: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, secondary_0: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, secondary_1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.repo_id=local/eval_pi05_inference_test --dataset.single_task="Pick up the red lego and place it on the paper plate, please." --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.push_to_hub=false --dataset.root=./inference_data --policy.path="/home/aiclub/dev/lerobot/src/pi05fixed"
 
 




## Current Status

✓ Your model `bdhillon/PIv2` has all required files
✓ Hardware is ready (3 cameras, robot, GPU)
✓ Software is installed
⚠️ Config needs minor fix (`"pi05"` → `"pi0"`)

## Issue & Fix

**Problem:** LeRobot recognizes `"pi0"` but not `"pi05"` as a policy type.

**Solution:** Use the local fixed model at `./pi05_fixed/` (currently downloading)

---

## Once Download Completes

The script will automatically:
1. Fix the config.json (change type to "pi0")
2. Save everything to `./pi05_fixed/`

Then run inference with:

```bash
cd /home/aiclub/dev/lerobot
conda activate lerobot

# Test the setup first
python test-pi05-setup.py

# Run inference (choose one):

# Option 1: Simple bash script
./run-pi05-inference.sh

# Option 2: Python script with custom task
python run-pi05-inference-simple.py \
    --task "Pick up the blue cube" \
    --duration 30 \
    --policy-path ./pi05_fixed

# Option 3: With auto-reset to starting position
./run-pi05-with-reset.sh
```

---

## Important: Update Scripts to Use Local Model

After download completes, update your inference scripts to use `./pi05_fixed` instead of `bdhillon/PIv2`:

### Edit run-pi05-inference.sh:
```bash
# Change this line:
--policy.path=bdhillon/PIv2

# To this:
--policy.path=./pi05_fixed
```

### For Python script:
```bash
python run-pi05-inference-simple.py --policy-path ./pi05_fixed
```

---

## Files Created

All in `/home/aiclub/dev/lerobot/`:

1. **run-pi05-inference.sh** - Quick inference script
2. **run-pi05-inference-simple.py** - Python inference with options
3. **run-pi05-with-reset.sh** - Inference with auto-reset
4. **test-pi05-setup.py** - Verify environment
5. **check-model-files.py** - Check HuggingFace model
6. **fix-pi05-model.sh** - Download and fix model (running now)
7. **PI05_INFERENCE_GUIDE.md** - Detailed guide
8. **MINIMUM_INFERENCE_BUNDLE.md** - Required files explanation
9. **QUICK_START.md** - This file

---

## Download Progress

Check status with:
```bash
# Check file sizes
du -sh pi05_fixed/

# Check if complete
ls -lh pi05_fixed/model.safetensors
```

Model is ~3.4GB, currently downloading...

---

## Safety Reminders

- Always supervise the robot
- Press Ctrl+C to stop anytime
- The robot has speed limits (`max_relative_target=0.05`)
- Ensure clear workspace

---

## Troubleshooting

### Download taking too long?
Check your internet connection:
```bash
speedtest-cli
```

### After download completes but scripts fail?
1. Verify files: `ls -lh pi05_fixed/`
2. Check config was fixed: `grep '"type"' pi05_fixed/config.json`
3. Should show: `"type": "pi0"`

### Need to re-download?
```bash
rm -rf pi05_fixed/
./fix-pi05-model.sh
```
