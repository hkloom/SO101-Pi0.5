## 12.17.2025 - Wednesday, December 17th

- Inference loop is working now working, see: `pi0_basic_inference.py`
- Converted radians to degrees
- Added async inference w-/ policy server
- Looks like half the trajectory is running
- Normalization does not seem to be working:
```bash
Warning: Could not remap state dict keys: Error(s) in loading state_dict for PI05OpenPIPolicy:
Missing key(s) in state_dict: "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight". 

Policy type: pi05_openpi

Device: cuda

⚠️  WARNING: No unnormalization stats loaded!
    Actions will NOT be denormalized properly.
    Make sure model.safetensors contains 'unnormalize_outputs.*' keys.
⚠️  WARNING: No normalization stats loaded for inputs!
```

## 01.16.2026 - Friday, January 16th

- Ran inference on PIv3 -- trained on 42 trajectories 
- At the beginning of the day we were getting actions but robot was not moving 
  - Added print statements to robot.send_action()
  - Turns out we needed to reinstall lerobot, then robot.send_action() started working properly again
- Robot stood up, moved forward towards the lego, and opened the gripper.
- Played around with:
  - `chunk_size` (config.json > line 55)
  - `n_action_steps` (config.json > line 56)
  - `--max-relative-target` (pi0_basic_inference > line 623)
- Turns out the downloaded and USB transferred PIv3 was missing 200mb
  - Error while deserializing header: incomplete metadata, file not fully covered
    - Ensure any postprocessor stats files are present
    - It should list tensor keys instead of erroring
- Tensor Key check:
```bash
cd /home/aiclub/dev/SO101-Pi0.5 && python -m pip install safetensors && python - <<'PY'
from safetensors import safe_open
path = './pi05_fixed/model.safetensors'
with safe_open(path, framework='pt') as f:
    print('tensors:', list(f.keys()))
PY
```

## 01.21.2026 - Wednesday, January 21st

- WiFi has been reconfigured.
- Successfully downloaded 14gb PIv3 with uncorrupted tensors
- Ran inference 
  - Robot stands up, rotates its base to the right, descends partially, and opens and closes its gripper
  - Depth is still off, robot cannot descend completely
  - Movement is not random, there seems to be some learned training signal 
- Try another recording + training while:
  - Making sure camera/image resolutions are correct when recording, training, and inferencing (224x224 matches what the VLM PaliGemma was trained on) 
  - Recording a solid 50-100 trajectory dataset with proper camera configuration, the cable not in the way, and no edits after

## 1.30.2026 - Friday, January 30th

- Confirmed camera mappings are consistent through recording-training-inference
- Realized that calibration needs to be constant through recording-training-inference as well 
- Realized that the dataset should be created by only watching the camera feeds when performing the task
- Realized recording and training happen at 30fps and inference happens at 12-15hz
  - Is the policy experiencing a "dropped frame" every other frame?
  - The dropped frames may be adding jank in the observation input leading to jank in the action output
- Switching to ACT and getting that running at 30hz may be a better approach