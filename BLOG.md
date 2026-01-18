## 12.17.2025

Accomplished:
  - Inference loop is working!
  - Normalization applied!
  - Converted radians to degrees
  - Looks like half the trajectory is running
  - Add async inference w-/ policy server

Now:
  - Trying to figure out why we still get: WARNING:root:Relative goal position magnitude had to be clamped to be safe.

  --- From: run*.log ---

    Warning: Could not remap state dict keys: Error(s) in loading state_dict for PI05OpenPIPolicy:
	  Missing key(s) in state_dict: "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight". 

    Policy type: pi05_openpi

    Device: cuda

    ⚠️  WARNING: No unnormalization stats loaded!
       Actions will NOT be denormalized properly.
       Make sure model.safetensors contains 'unnormalize_outputs.*' keys.
    ⚠️  WARNING: No normalization stats loaded for inputs!



Change:

## 1.17.2026

- Ran inference on PIv3 -- trained on 42 trajectories 
- Robot stood up, moved forward towards the lego, and opened the gripper.

- Cursor says

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


