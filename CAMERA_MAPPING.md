# Camera Mapping Reference

## Training vs Physical Setup

### Model Expects (from training):
- `main` = wrist camera
- `secondary_0` = top/front camera
- `secondary_1` = side camera

### Your Physical Setup:
- /dev/video0 = side camera
- /dev/video2 = wrist camera
- /dev/video4 = top/front camera

### Final Mapping in Scripts:
```
main        (wrist)     → /dev/video2
secondary_0 (top/front) → /dev/video4
secondary_1 (side)      → /dev/video0
```

## Why This Matters

The policy was trained with specific camera inputs. If you swap the cameras or use different names, the policy will:
- Receive incorrect visual input
- Make wrong predictions
- Fail to perform the task correctly

**Always ensure camera names match the training configuration!**

## Quick Reference

When running inference:
```bash
--robot.cameras='{
    main: {type: opencv, index_or_path: /dev/video2, ...},         # wrist
    secondary_0: {type: opencv, index_or_path: /dev/video4, ...},  # front
    secondary_1: {type: opencv, index_or_path: /dev/video0, ...}   # side
}'
```
