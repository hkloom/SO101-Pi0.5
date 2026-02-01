# Pi0.5 Inference Tutorial - LeRobot

A complete guide to running inference with Pi0.5 policy checkpoints in LeRobot.

## Prerequisites

```bash
pip install lerobot torch safetensors numpy
```

## Required Checkpoint Files

Your checkpoint directory should contain these files:

```
your_checkpoint_directory/
├── config.json              # Model architecture and settings
├── model.safetensors        # Trained policy weights
├── preprocessor.safetensors # Input normalization parameters
└── postprocessor.safetensors # Output denormalization parameters
```

**What each file does:**
- `config.json` - Defines model architecture, input/output shapes
- `model.safetensors` - Contains the trained neural network weights
- `preprocessor.safetensors` - Normalizes inputs (images, states) before feeding to model
- `postprocessor.safetensors` - Denormalizes outputs (actions) to real-world values

## Basic Loading and Inference

```python
from lerobot.common.policies.factory import make_policy
import torch

# Load the policy (automatically loads all files)
checkpoint_path = "path/to/checkpoint/directory"
policy = make_policy(pretrained=checkpoint_path, device="cuda")
policy.eval()

# Prepare observation
observation = {
    "observation.image": image_tensor,  # Shape: (1, 3, H, W)
    "observation.state": state_tensor,  # Shape: (1, state_dim)
}

# Run inference
with torch.no_grad():
    action = policy.select_action(observation)

print(f"Action: {action}")
```

## Complete Inference Function

```python
import torch
import numpy as np
from lerobot.common.policies.factory import make_policy

def run_pi0_inference(checkpoint_path, image, state, device="cuda"):
    """
    Run inference with Pi0.5 policy.
    
    Args:
        checkpoint_path: Path to directory containing checkpoint files
        image: numpy array (H, W, 3) with values 0-255 or (3, H, W) float
        state: numpy array (state_dim,)
        device: "cuda" or "cpu"
    
    Returns:
        action: numpy array of predicted actions
    """
    # Load policy
    policy = make_policy(pretrained=checkpoint_path, device=device)
    policy.eval()
    
    # Convert image to tensor
    image_tensor = torch.from_numpy(image).float()
    
    # Rearrange to (C, H, W) if needed
    if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    
    # Create observation dict
    observation = {
        "observation.image": image_tensor,
        "observation.state": state_tensor,
    }
    
    # Get action
    with torch.no_grad():
        action = policy.select_action(observation)
    
    return action.cpu().numpy()

# Example usage
if __name__ == "__main__":
    checkpoint = "path/to/checkpoint/directory"
    
    # Example random inputs
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    state = np.random.randn(7)
    
    action = run_pi0_inference(checkpoint, image, state)
    print(f"Predicted action: {action}")
```

## Real-Time Inference Loop

```python
from lerobot.common.policies.factory import make_policy
import torch
import numpy as np

# Load policy once
checkpoint_path = "path/to/checkpoint/directory"
policy = make_policy(pretrained=checkpoint_path, device="cuda")
policy.eval()

# Inference loop
while True:
    # 1. Get observation from your robot/camera
    image = get_camera_image()  # Your function - returns (H, W, 3) uint8
    state = get_robot_state()   # Your function - returns (state_dim,) float
    
    # 2. Convert to tensors
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    
    # 3. Move to device
    observation = {
        "observation.image": image_tensor.to(policy.device),
        "observation.state": state_tensor.to(policy.device),
    }
    
    # 4. Get action
    with torch.no_grad():
        action = policy.select_action(observation)
    
    # 5. Execute action on robot
    execute_robot_action(action.cpu().numpy())  # Your function
```

## Checking Your Configuration

```python
import json

# Load and inspect config
with open("path/to/checkpoint/directory/config.json", "r") as f:
    config = json.load(f)

print("Policy type:", config.get("policy_type"))
print("Input shapes:", config.get("input_shapes"))
print("Output shapes:", config.get("output_shapes"))

# Example output:
# Policy type: pi0
# Input shapes: {'observation.image': [3, 224, 224], 'observation.state': [7]}
# Output shapes: {'action': [7]}
```

## Common Issues and Solutions

### 1. Wrong Input Format

**Problem:** Tensor shape mismatch

**Solution:** Ensure correct format
```python
# Images should be (batch, channels, height, width)
image = torch.from_numpy(image).float()
if image.shape[-1] == 3:  # If (H, W, 3)
    image = image.permute(2, 0, 1)  # Convert to (3, H, W)
image = image.unsqueeze(0)  # Add batch dimension -> (1, 3, H, W)

# States should be (batch, state_dim)
state = torch.from_numpy(state).float().unsqueeze(0)  # (state_dim,) -> (1, state_dim)
```

### 2. Device Mismatch

**Problem:** RuntimeError about tensors on different devices

**Solution:** Move all tensors to same device
```python
device = policy.device
observation = {
    "observation.image": image_tensor.to(device),
    "observation.state": state_tensor.to(device),
}
```

### 3. Missing Files

**Problem:** FileNotFoundError for config.json or .safetensors

**Solution:** Verify all files are present
```python
import os

checkpoint_dir = "path/to/checkpoint"
required_files = ["config.json", "model.safetensors", 
                  "preprocessor.safetensors", "postprocessor.safetensors"]

for file in required_files:
    filepath = os.path.join(checkpoint_dir, file)
    if os.path.exists(filepath):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING")
```

### 4. Image Value Range

**Problem:** Poor predictions due to incorrect image normalization

**Solution:** The preprocessor handles normalization, but ensure input format is consistent
```python
# If your images are 0-255 uint8 (most common)
image = np.array(image, dtype=np.uint8)  # Values 0-255

# If your images are already 0-1 float
image = np.array(image, dtype=np.float32)  # Values 0.0-1.0

# Convert to tensor
image_tensor = torch.from_numpy(image).float()
```

## Batch Processing Multiple Observations

```python
def batch_inference(policy, images, states):
    """
    Process multiple observations at once.
    
    Args:
        policy: Loaded Pi0 policy
        images: numpy array (batch, H, W, 3) or (batch, 3, H, W)
        states: numpy array (batch, state_dim)
    
    Returns:
        actions: numpy array (batch, action_dim)
    """
    # Convert to tensors
    images_tensor = torch.from_numpy(images).float()
    
    # Rearrange if needed
    if images_tensor.shape[-1] == 3:
        images_tensor = images_tensor.permute(0, 3, 1, 2)
    
    states_tensor = torch.from_numpy(states).float()
    
    # Move to device
    observation = {
        "observation.image": images_tensor.to(policy.device),
        "observation.state": states_tensor.to(policy.device),
    }
    
    # Get actions
    with torch.no_grad():
        actions = policy.select_action(observation)
    
    return actions.cpu().numpy()

# Example usage
batch_size = 4
images = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)
states = np.random.randn(batch_size, 7)

actions = batch_inference(policy, images, states)
print(f"Batch actions shape: {actions.shape}")
```

## Performance Tips

### Use GPU for Faster Inference
```python
# Load on GPU
policy = make_policy(pretrained=checkpoint_path, device="cuda")

# For even faster inference, use half precision (if supported)
policy = policy.half()
```

### Optimize for Real-Time Control
```python
import torch

# Disable gradient computation (already done with torch.no_grad())
# Use compiled model for faster execution (PyTorch 2.0+)
policy = torch.compile(policy)

# Warm up the model
dummy_obs = {
    "observation.image": torch.randn(1, 3, 224, 224).to(policy.device),
    "observation.state": torch.randn(1, 7).to(policy.device),
}
with torch.no_grad():
    _ = policy.select_action(dummy_obs)
```

## Debugging Checklist

When inference isn't working:

1. **Verify files exist:**
   ```python
   import os
   print(os.listdir("path/to/checkpoint"))
   ```

2. **Check config:**
   ```python
   import json
   with open("path/to/checkpoint/config.json") as f:
       print(json.dumps(json.load(f), indent=2))
   ```

3. **Test with dummy data:**
   ```python
   # Create tensors matching expected shapes from config
   dummy_obs = {
       "observation.image": torch.randn(1, 3, 224, 224).to(policy.device),
       "observation.state": torch.randn(1, 7).to(policy.device),
   }
   action = policy.select_action(dummy_obs)
   print(f"Action shape: {action.shape}")
   ```

4. **Verify device:**
   ```python
   print(f"Policy device: {policy.device}")
   print(f"Policy dtype: {next(policy.parameters()).dtype}")
   ```

## Getting Help

If you encounter issues:

1. Check the LeRobot GitHub issues: https://github.com/huggingface/lerobot/issues
2. Join the LeRobot Discord community
3. Verify you're using compatible versions:
   ```python
   import lerobot
   import torch
   print(f"LeRobot version: {lerobot.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   ```

## Additional Resources

- LeRobot Documentation: https://github.com/huggingface/lerobot
- Pi0 Paper: https://www.physicalintelligence.company/blog/pi0
- Example Models: https://huggingface.co/lerobot

---

**Last Updated:** December 2025