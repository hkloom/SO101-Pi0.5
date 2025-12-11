#!/bin/bash
# Download and fix the PI 0.5 model locally

set -e

MODEL_DIR="./pi05_fixed"

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
echo "  --policy.path=./pi05_fixed"
