#!/usr/bin/env python3
"""Script to download and prepare datasets for VLM training."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_rlhfv():
    """Download RLHF-V dataset."""
    print("Downloading RLHF-V dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openbmb/RLHF-V-Dataset")
        print(f"  RLHF-V: {len(ds['train'])} samples")
        return True
    except Exception as e:
        print(f"  Failed to download RLHF-V: {e}")
        return False


def download_pixmo():
    """Download PixMo datasets."""
    print("Downloading PixMo datasets...")
    try:
        from datasets import load_dataset

        # PixMo-Cap
        try:
            ds_cap = load_dataset("allenai/pixmo-cap", split="train")
            print(f"  PixMo-Cap: {len(ds_cap)} samples")
        except Exception as e:
            print(f"  PixMo-Cap not available: {e}")

        # PixMo-Points
        try:
            ds_points = load_dataset("allenai/pixmo-points", split="train")
            print(f"  PixMo-Points: {len(ds_points)} samples")
        except Exception as e:
            print(f"  PixMo-Points not available: {e}")

        # PixMo-Count
        try:
            ds_count = load_dataset("allenai/pixmo-count", split="train")
            print(f"  PixMo-Count: {len(ds_count)} samples")
        except Exception as e:
            print(f"  PixMo-Count not available: {e}")

        return True
    except Exception as e:
        print(f"  Failed to download PixMo: {e}")
        return False


def download_spatial_vlm():
    """Download SpatialVLM dataset."""
    print("Downloading SpatialVLM dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("remyxai/vqasynth_spacial", split="train")
        print(f"  SpatialVLM: {len(ds)} samples")
        return True
    except Exception as e:
        print(f"  Failed to download SpatialVLM: {e}")
        print("  Note: This dataset may require authentication or may not be publicly available.")
        return False


def download_vsr():
    """Download VSR benchmark."""
    print("Downloading VSR benchmark...")
    try:
        from datasets import load_dataset
        ds = load_dataset("cambridgeltl/vsr_random", split="test")
        print(f"  VSR: {len(ds)} samples")
        return True
    except Exception as e:
        print(f"  Failed to download VSR: {e}")
        return False


def download_cvbench():
    """Download CV-Bench."""
    print("Downloading CV-Bench...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
        print(f"  CV-Bench: {len(ds)} samples")
        return True
    except Exception as e:
        print(f"  Failed to download CV-Bench: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets for VLM training")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "rlhfv", "pixmo", "spatial", "vsr", "cvbench"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    args = parser.parse_args()

    if args.cache_dir:
        import os
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    print("=" * 60)
    print("Dataset Download Script")
    print("=" * 60)

    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["rlhfv", "pixmo", "spatial", "vsr", "cvbench"]

    results = {}

    if "rlhfv" in datasets_to_download:
        results["RLHF-V"] = download_rlhfv()

    if "pixmo" in datasets_to_download:
        results["PixMo"] = download_pixmo()

    if "spatial" in datasets_to_download:
        results["SpatialVLM"] = download_spatial_vlm()

    if "vsr" in datasets_to_download:
        results["VSR"] = download_vsr()

    if "cvbench" in datasets_to_download:
        results["CV-Bench"] = download_cvbench()

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)

    # Return non-zero if any download failed
    if not all(results.values()):
        print("\nSome downloads failed. Check errors above.")
        sys.exit(1)
    else:
        print("\nAll datasets downloaded successfully!")


if __name__ == "__main__":
    main()
