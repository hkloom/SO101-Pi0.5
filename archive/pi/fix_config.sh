#!/bin/bash
# fix_config.sh — Download the Pi0.5 checkpoint from HuggingFace (bdhillon/PIv2)
# and patch its config.json so LeRobot can load it.
#
# The upstream checkpoint sets "type": "pi05", but our LeRobot fork registers
# the policy class as "pi0". Without the patch, get_policy_class("pi05") fails.
# This script rewrites the type field to "pi0" after download.
#
# Downloads: config.json, model.safetensors, pre/post-processor JSON + safetensors
# Output:    ./ckpt/PIv3/ (ready to pass as --checkpoint to async.py)
#
# Only needs to be run once. Re-run to re-download or recreate the checkpoint.

set -e

MODEL_DIR="./ckpt/PIv3"

echo "Creating directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

echo "Downloading model files..."
huggingface-cli download bdhillon/PIv2 \
    config.json \
    model.safetensors \
    policy_preprocessor.json \
    policy_postprocessor.json \
    policy_preprocessor_step_2_normalizer_processor.safetensors \
    policy_postprocessor_step_0_unnormalizer_processor.safetensors \
    --local-dir "$MODEL_DIR"

echo "Fixing config.json (changing pi05 -> pi0)..."
python -c "
import json
config_path = '$MODEL_DIR/config.json'
with open(config_path) as f:
    config = json.load(f)
config['type'] = 'pi0'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print('✓ Config fixed!')
"

echo ""
echo "✓ Model fixed and saved to: $MODEL_DIR"
echo ""
echo "Now use this in your inference scripts:"
echo "  --policy.path=./ckpt/PIv3"
