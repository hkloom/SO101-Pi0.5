#!/usr/bin/env python3
"""
Quick script to verify that a HuggingFace model repository has all required files for inference.

Usage:
    python check-model-files.py bdhillon/PIv2
    python check-model-files.py lerobot/pi0
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_model_files(repo_id: str):
    """Check if a HuggingFace model has all required files for inference."""
    try:
        from huggingface_hub import list_repo_files, repo_info
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    logger.info(f"Checking model repository: {repo_id}")
    logger.info("=" * 60)

    # Required files for inference
    required_files = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]

    try:
        # Get repository info
        info = repo_info(repo_id, repo_type="model")
        logger.info(f"Repository: {info.id}")
        logger.info(f"Author: {info.author}")
        if hasattr(info, 'lastModified'):
            logger.info(f"Last modified: {info.lastModified}")

        # List all files
        logger.info("\n" + "=" * 60)
        logger.info("All files in repository:")
        logger.info("=" * 60)

        files = list(list_repo_files(repo_id))
        for f in sorted(files):
            logger.info(f"  {f}")

        # Check required files
        logger.info("\n" + "=" * 60)
        logger.info("Required files check:")
        logger.info("=" * 60)

        all_present = True
        for req_file in required_files:
            if req_file in files:
                logger.info(f"  ✓ {req_file}")
            else:
                logger.error(f"  ✗ MISSING: {req_file}")
                all_present = False

        # Summary
        logger.info("\n" + "=" * 60)
        if all_present:
            logger.info("✓ SUCCESS: All required files are present!")
            logger.info("This model is ready for inference.")
            return True
        else:
            logger.error("✗ FAILURE: Some required files are missing!")
            logger.error("This model cannot be used for inference until all files are present.")
            return False

    except Exception as e:
        logger.error(f"Error accessing repository: {e}")
        logger.info("\nPossible issues:")
        logger.info("  1. Repository doesn't exist")
        logger.info("  2. Repository is private (run: huggingface-cli login)")
        logger.info("  3. No internet connection")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python check-model-files.py <huggingface-repo-id>")
        print("\nExamples:")
        print("  python check-model-files.py bdhillon/PIv2")
        print("  python check-model-files.py lerobot/pi0")
        sys.exit(1)

    repo_id = sys.argv[1]
    success = check_model_files(repo_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
