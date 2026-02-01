#!/usr/bin/env python3
"""Script to download and prepare datasets for VLM training."""

import argparse
import sys
from pathlib import Path

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
        print(
            "  Note: This dataset may require authentication or may not be publicly available."
        )
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


def download_llava_instruct():
    """Download LLaVA-Instruct-150K dataset."""
    print("Downloading LLaVA-Instruct-150K...")
    try:
        from datasets import load_dataset

        ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")
        print(f"  LLaVA-Instruct-150K: {len(ds)} samples")
        print("  Note: COCO images will be downloaded at training time.")
        return True
    except Exception as e:
        print(f"  Failed to download LLaVA-Instruct-150K: {e}")
        return False


def download_pixmo_points():
    """Download PixMo-Points dataset."""
    print("Downloading PixMo-Points...")
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/pixmo-points", split="train")
        print(f"  PixMo-Points: {len(ds)} samples")
        print("  Note: Images will be downloaded from URLs at training time.")
        return True
    except Exception as e:
        print(f"  Failed to download PixMo-Points: {e}")
        return False


def download_sharegpt4v():
    """Download ShareGPT4V dataset."""
    print("Downloading ShareGPT4V...")
    try:
        from datasets import load_dataset

        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train")
        print(f"  ShareGPT4V: {len(ds)} samples")
        print("  Note: COCO images will be downloaded at training time.")
        return True
    except Exception as e:
        print(f"  Failed to download ShareGPT4V: {e}")
        return False


def download_pixmo_docs():
    """Download PixMo-Docs dataset."""
    print("Downloading PixMo-Docs...")
    try:
        from datasets import load_dataset

        total = 0
        for subset in ["charts", "tables", "diagrams", "other"]:
            try:
                ds = load_dataset("allenai/pixmo-docs", subset, split="train")
                print(f"  PixMo-Docs ({subset}): {len(ds)} samples")
                total += len(ds)
            except Exception as e:
                print(f"  PixMo-Docs ({subset}) not available: {e}")
        print(f"  PixMo-Docs total: {total} samples")
        return total > 0
    except Exception as e:
        print(f"  Failed to download PixMo-Docs: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets for VLM training")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "rlhfv",
            "pixmo",
            "spatial",
            "vsr",
            "cvbench",
            "llava_instruct",
            "pixmo_points",
            "sharegpt4v",
            "pixmo_docs",
        ],
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
        datasets_to_download = [
            "rlhfv",
            "pixmo",
            "spatial",
            "vsr",
            "cvbench",
            "llava_instruct",
            "pixmo_points",
            "sharegpt4v",
            "pixmo_docs",
        ]

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

    if "llava_instruct" in datasets_to_download:
        results["LLaVA-Instruct-150K"] = download_llava_instruct()

    if "pixmo_points" in datasets_to_download:
        results["PixMo-Points"] = download_pixmo_points()

    if "sharegpt4v" in datasets_to_download:
        results["ShareGPT4V"] = download_sharegpt4v()

    if "pixmo_docs" in datasets_to_download:
        results["PixMo-Docs"] = download_pixmo_docs()

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
