# Minimum Inference Bundle for PI 0.5 Policy

## Required Files in `bdhillon/PIv2` Repository

Based on LeRobot's code analysis, the **minimum files** required for inference are:

### 1. **config.json** (Required)
The policy configuration file containing:
- Policy type (must be `"pi0"` for PI 0.5)
- Model architecture settings
- Device configuration (cpu/cuda)
- Input/output feature definitions
- Any policy-specific hyperparameters

**Location:** Root of repository
**Loaded by:** `PreTrainedConfig.from_pretrained()`

### 2. **model.safetensors** (Required)
The actual trained model weights in SafeTensors format.

**Location:** Root of repository
**Loaded by:** `hf_hub_download()` with filename `SAFETENSORS_SINGLE_FILE`
**Note:** This is the main model file and will be the largest file (typically several GB)

### 3. **policy_preprocessor.json** (Required)
Preprocessor configuration that defines how to transform raw observations before feeding to the policy.

Includes:
- Image normalization parameters
- State normalization statistics (mean, std)
- Device settings
- Data type conversions

**Location:** Root of repository
**Loaded by:** `PolicyProcessorPipeline.from_pretrained()` with filename `"policy_preprocessor.json"`

### 4. **policy_postprocessor.json** (Required)
Postprocessor configuration that defines how to transform policy outputs to robot actions.

Includes:
- Action denormalization parameters
- Output scaling/clipping
- Action space mapping

**Location:** Root of repository
**Loaded by:** `PolicyProcessorPipeline.from_pretrained()` with filename `"policy_postprocessor.json"`

---

## Complete File Structure

```
bdhillon/PIv2/
├── config.json                    # Policy configuration
├── model.safetensors             # Model weights
├── policy_preprocessor.json      # Input preprocessing config
└── policy_postprocessor.json     # Output postprocessing config
```

---

## Optional Files (Recommended but not required)

### README.md
Documentation about the model, training details, usage instructions.

### .gitattributes
HuggingFace configuration for LFS (Large File Storage) for the safetensors file.

---

## How LeRobot Loads These Files

### From the Code Analysis:

1. **Policy Loading** (`pretrained.py:70-128`):
   ```python
   # Loads config.json
   config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)

   # Downloads and loads model.safetensors
   model_file = hf_hub_download(
       repo_id=model_id,
       filename="model.safetensors"
   )
   policy = cls._load_as_safetensor(instance, model_file, device)
   ```

2. **Preprocessor Loading** (`factory.py:202-211`):
   ```python
   preprocessor = PolicyProcessorPipeline.from_pretrained(
       pretrained_model_name_or_path=pretrained_path,
       config_filename="policy_preprocessor.json"
   )
   ```

3. **Postprocessor Loading** (`factory.py:212-221`):
   ```python
   postprocessor = PolicyProcessorPipeline.from_pretrained(
       pretrained_model_name_or_path=pretrained_path,
       config_filename="policy_postprocessor.json"
   )
   ```

---

## Checking Your Model Repository

To verify your `bdhillon/PIv2` repository has the minimum required files, run:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# List files in the repository
python -c "
from huggingface_hub import list_repo_files
files = list_repo_files('bdhillon/PIv2')
print('Files in bdhillon/PIv2:')
for f in sorted(files):
    print(f'  - {f}')

required = ['config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json']
print('\nRequired files check:')
for req in required:
    status = '✓' if req in files else '✗ MISSING'
    print(f'  {status} {req}')
"
```

---

## What Happens if Files Are Missing?

| Missing File | Error Message | Impact |
|-------------|---------------|---------|
| `config.json` | "config.json not found" | Cannot instantiate policy |
| `model.safetensors` | "model.safetensors not found on HuggingFace Hub" | Cannot load weights |
| `policy_preprocessor.json` | "policy_preprocessor.json not found" | Cannot preprocess observations |
| `policy_postprocessor.json` | "policy_postprocessor.json not found" | Cannot postprocess actions |

---

## Creating These Files (If Training Your Own)

If you trained your own PI 0.5 model, these files are created by LeRobot's training pipeline:

1. **During Training:**
   - `config.json` is created from your training config
   - `model.safetensors` is saved at checkpoints
   - Processor configs are created during dataset preprocessing

2. **Pushing to Hub:**
   ```python
   # LeRobot automatically includes all required files
   policy.push_model_to_hub(cfg)
   ```

3. **From Local Checkpoint:**
   If you have a local checkpoint at `outputs/train/pi0_myexp/checkpoints/005000/pretrained_model/`:
   ```bash
   # This directory should contain all 4 required files
   ls outputs/train/pi0_myexp/checkpoints/005000/pretrained_model/
   # Should show:
   # config.json
   # model.safetensors
   # policy_preprocessor.json
   # policy_postprocessor.json
   ```

---

## Summary

**Minimum inference bundle = 4 files:**
1. `config.json` - Policy architecture
2. `model.safetensors` - Trained weights
3. `policy_preprocessor.json` - Input processing
4. `policy_postprocessor.json` - Output processing

All 4 must be present in the HuggingFace repository for inference to work.
