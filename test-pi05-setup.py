#!/usr/bin/env python3
"""
Test script to verify PI 0.5 inference setup before running on the robot.
This checks:
1. Required packages are installed
2. Model can be loaded from HuggingFace
3. Cameras are accessible
4. Serial port is available

Usage:
    python test-pi05-setup.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing imports...")
    try:
        import torch
        import lerobot
        from lerobot.policies.factory import make_policy
        from lerobot.robots.so101_follower import SO101Follower
        logger.info("✓ All required packages imported successfully")
        logger.info(f"  - PyTorch version: {torch.__version__}")
        logger.info(f"  - LeRobot version: {lerobot.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_model_loading():
    """Test loading the PI 0.5 model from HuggingFace."""
    logger.info("\nTesting model loading...")
    try:
        from lerobot.configs.policies import PreTrainedConfig

        model_path = "bdhillon/PIv2"
        logger.info(f"  Attempting to load config from {model_path}...")

        config = PreTrainedConfig.from_pretrained(model_path)
        logger.info(f"✓ Successfully loaded model config")
        logger.info(f"  - Policy type: {config.type}")
        logger.info(f"  - Device: {config.device}")

        return True
    except Exception as e:
        logger.error(f"✗ Model loading error: {e}")
        logger.info("  Tip: Run 'huggingface-cli login' if the model is private")
        return False


def test_cameras():
    """Test camera availability."""
    logger.info("\nTesting cameras...")
    import os

    cameras = {
        "front": "/dev/video0",
        "side": "/dev/video2",
        "wrist": "/dev/video4",
    }

    all_ok = True
    for name, path in cameras.items():
        if os.path.exists(path):
            logger.info(f"✓ {name} camera found at {path}")
        else:
            logger.warning(f"✗ {name} camera NOT found at {path}")
            all_ok = False

    if not all_ok:
        logger.info("\n  Available video devices:")
        os.system("ls -la /dev/video* 2>/dev/null")

    return all_ok


def test_robot_port():
    """Test robot serial port availability."""
    logger.info("\nTesting robot serial port...")
    import os

    port = "/dev/ttyACM0"
    if os.path.exists(port):
        logger.info(f"✓ Robot port found at {port}")

        # Check permissions
        import stat
        st = os.stat(port)
        mode = st.st_mode
        if os.access(port, os.R_OK | os.W_OK):
            logger.info(f"✓ Port {port} is readable and writable")
            return True
        else:
            logger.warning(f"✗ Port {port} exists but may not have proper permissions")
            logger.info(f"  Run: sudo usermod -a -G dialout $USER")
            logger.info(f"  Then log out and log back in")
            return False
    else:
        logger.warning(f"✗ Robot port NOT found at {port}")
        logger.info("\n  Available serial ports:")
        os.system("ls -la /dev/ttyACM* /dev/ttyUSB* 2>/dev/null")
        return False


def test_gpu():
    """Test GPU availability."""
    logger.info("\nTesting GPU...")
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"✓ CUDA is available")
            logger.info(f"  - GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  - CUDA version: {torch.version.cuda}")
            logger.info(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            logger.warning(f"✗ CUDA is not available - will use CPU (slower)")
            return False
    except Exception as e:
        logger.error(f"✗ GPU test error: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("PI 0.5 Inference Setup Test")
    logger.info("=" * 60)

    results = {
        "Imports": test_imports(),
        "Model Loading": test_model_loading(),
        "Cameras": test_cameras(),
        "Robot Port": test_robot_port(),
        "GPU": test_gpu(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:20s}: {status}")

    all_critical_passed = results["Imports"] and results["Model Loading"]

    logger.info("=" * 60)
    if all_critical_passed:
        logger.info("✓ Critical tests passed! You can proceed with inference.")
        logger.info("\nTo run inference:")
        logger.info("  ./run-pi05-inference.sh")
        logger.info("  OR")
        logger.info("  python run-pi05-inference-simple.py")
        return 0
    else:
        logger.error("✗ Some critical tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
