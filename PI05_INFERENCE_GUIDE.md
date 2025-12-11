# PI 0.5 Inference Guide for SO-101 Robot

This guide explains how to run inference using a trained PI 0.5 policy on your SO-101 robotic arms.

## Prerequisites

1. **Conda Environment**: Activate the lerobot environment
   ```bash
   conda activate lerobot
   ```

2. **Hardware Setup**:
   - SO-101 robot connected to `/dev/ttyACM0`
   - 3 cameras connected:
     - Front camera: `/dev/video0`
     - Side camera: `/dev/video2`
     - Wrist camera: `/dev/video4`

3. **Model**: Your trained PI 0.5 model at `bdhillon/PIv2` on HuggingFace

## Method 1: Using the Bash Script (Recommended for Quick Testing)

The bash script uses LeRobot's `record` command with a policy, which will:
- Run inference using your PI 0.5 policy
- Save observations locally (not pushed to HuggingFace)
- Display camera feeds in real-time

```bash
cd /home/aiclub/dev/lerobot
./run-pi05-inference.sh
```

### Customizing the Bash Script

Edit `run-pi05-inference.sh` to change:
- `--dataset.single_task`: Update the task description
- `--dataset.episode_time_s`: Change duration (default: 60s)
- `--robot.port`: Change robot port if needed
- `--policy.path`: Use a different model

## Method 2: Using the Python Script (More Control)

The Python script provides a cleaner interface for running inference without dataset recording overhead.

### Basic Usage

```bash
cd /home/aiclub/dev/lerobot
python run-pi05-inference-simple.py
```

### With Custom Parameters

```bash
python run-pi05-inference-simple.py \
    --task "Pick up the blue cube and place it in the bin" \
    --duration 120 \
    --fps 30 \
    --robot-port /dev/ttyACM0 \
    --policy-path bdhillon/PIv2
```

### Command-line Options

- `--policy-path`: HuggingFace model path (default: `bdhillon/PIv2`)
- `--robot-port`: Serial port for robot (default: `/dev/ttyACM0`)
- `--task`: Task description for the policy (default: "Pick up the red lego...")
- `--duration`: Duration in seconds (default: 60.0)
- `--fps`: Control frequency (default: 30)
- `--no-display`: Disable camera display

## Troubleshooting

### Camera Issues

If cameras aren't working, check which cameras are available:
```bash
ls -la /dev/video*
v4l2-ctl --list-devices
```

### Robot Connection Issues

Check if the robot is connected:
```bash
ls -la /dev/ttyACM*
```

If the robot is on a different port, update the `--robot-port` parameter.

### Model Download Issues

The first run will download the model from HuggingFace. Ensure you have:
1. Internet connection
2. HuggingFace token configured (if model is private)

```bash
huggingface-cli login
```

### Permission Issues

If you get permission errors for cameras or serial ports:
```bash
sudo usermod -a -G dialout,video $USER
# Then log out and log back in
```

## Monitoring Performance

The scripts will log performance metrics including:
- Control loop frequency (Hz)
- Camera read times
- Robot communication times

Target is 30 Hz. If you see yellow warnings, the system isn't keeping up.

## Safety Notes

1. **Emergency Stop**: Press Ctrl+C to stop the robot at any time
2. **Safe Space**: Ensure the robot has clear workspace
3. **Supervision**: Always supervise the robot during operation
4. **Speed Limits**: The `max_relative_target=0.05` limits movement speed for safety

## Next Steps

1. **Test with Different Tasks**: Modify the task description to test different behaviors
2. **Adjust Camera Positions**: Optimize camera placement for your workspace
3. **Fine-tune Parameters**: Adjust FPS if you need different speed/smoothness tradeoffs
4. **Add Reset Functionality**: Use the `eval-with-reset.py` pattern to automatically return to starting position

## File Locations

- Bash script: `/home/aiclub/dev/lerobot/run-pi05-inference.sh`
- Python script: `/home/aiclub/dev/lerobot/run-pi05-inference-simple.py`
- This guide: `/home/aiclub/dev/lerobot/PI05_INFERENCE_GUIDE.md`
- Inference data (if saved): `/home/aiclub/dev/lerobot/inference_data/`
