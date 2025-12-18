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

