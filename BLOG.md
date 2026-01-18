## 12.17.2025

Accomplished:
- Inference loop is working!
- Normalization applied!
- Converted radians to degrees
- Looks like half the trajectory is running
- Add async inference w-/ policy server

Now:
- Trying to figure out why we still get: WARNING:root:Relative goal position magnitude had to be clamped to be safe.

------------- From: run*.log ------------

Warning: Could not remap state dict keys: Error(s) in loading state_dict for PI05OpenPIPolicy:
Missing key(s) in state_dict: "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight". 

Policy type: pi05_openpi

Device: cuda

⚠️  WARNING: No unnormalization stats loaded!
    Actions will NOT be denormalized properly.
    Make sure model.safetensors contains 'unnormalize_outputs.*' keys.
⚠️  WARNING: No normalization stats loaded for inputs!

-----------------------------------------

==============================================================================================================

## 1.17.2026

Ran inference on PIv3 -- trained on 42 trajectories 

- At the beginning of the day we were getting actions but robot was not moving 
  - added print statements to robot.send_action()
  - turns out we needed to reinstall lerobot, then robot.send_action() started working properly
- Robot stood up, moved forward towards the lego, and opened the gripper.
- Played around with:
  - `chunk_size` (config.json > line 55)
  - `n_action_steps` (config.json > line 56)
  - `--max-relative-target` (pi0_basic_inference > line 623)

FROM LLM:
```text
safetensors is installed. The file is indeed corrupt:
  Error while deserializing header: incomplete metadata, file not fully covered

  "So ./pi05_fixed/model.safetensors is truncated. You’ll need to replace it with a complete copy of the checkpoint (and ensure any postprocessor stats files are present). Once you have a good file, re-run the same check; it should list tensor keys instead of erroring."

  Tensor Check:

    cd /home/aiclub/dev/SO101-Pi0.5 && python -m pip install safetensors && python - <<'PY'
    from safetensors import safe_open
    path = './pi05_fixed/model.safetensors'
    with safe_open(path, framework='pt') as f:
        print('tensors:', list(f.keys()))
    PY
```

NEXT
- Try to download uncorrupted tensors (should contain tensor keys) and run inference with that
- Try another recording + training while:
  - Making sure camera/image resolutions are correct when recording, training, and inferencing (224x224 or 448x448) 
  - Recording a solid 50-100 trajectory dataset with proper camera configuration, the cable not in the way, and no edits after
