"""Utility functions for Google Colab training.

This module provides helper functions for:
- GPU detection and optimization
- Google Drive path management
- Checkpoint handling
- Training time estimation
"""

import os
from pathlib import Path
from typing import Optional


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def detect_gpu_tier() -> str:
    """Detect GPU tier from available hardware.

    Returns:
        One of: "a100", "v100", "t4", "cpu"
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if "a100" in gpu_name:
        return "a100"
    elif "v100" in gpu_name:
        return "v100"
    elif "t4" in gpu_name or vram_gb < 20:
        return "t4"
    else:
        # Default to v100 settings for unknown GPUs
        return "v100"


def get_gpu_config(gpu_tier: str) -> dict:
    """Get recommended config settings for a GPU tier.

    Args:
        gpu_tier: One of "a100", "v100", "t4", "cpu"

    Returns:
        Dict with recommended settings
    """
    configs = {
        "a100": {
            "max_seq_length": 2048,
            "lora_r": 32,
            "gradient_accumulation_steps": 2,
            "dataloader_num_workers": 4,
            "grpo_generations": 4,
            "bf16": True,
        },
        "v100": {
            "max_seq_length": 1536,
            "lora_r": 32,
            "gradient_accumulation_steps": 4,
            "dataloader_num_workers": 4,
            "grpo_generations": 4,
            "bf16": True,
        },
        "t4": {
            "max_seq_length": 2048,
            "lora_r": 16,
            "gradient_accumulation_steps": 8,
            "dataloader_num_workers": 2,
            "grpo_generations": 4,
            "bf16": True,
        },
        "cpu": {
            "max_seq_length": 512,
            "lora_r": 8,
            "gradient_accumulation_steps": 16,
            "dataloader_num_workers": 0,
            "grpo_generations": 1,
            "bf16": False,
        },
    }
    return configs.get(gpu_tier, configs["t4"])


def setup_drive_paths(
    base_path: str = "/content/drive/MyDrive/qwen3_vl_training"
) -> dict:
    """Setup Google Drive paths for persistent storage.

    Args:
        base_path: Base directory on Google Drive

    Returns:
        Dict with paths for checkpoints, datasets, logs, hf_cache
    """
    paths = {
        "base": base_path,
        "checkpoints": f"{base_path}/checkpoints",
        "datasets": f"{base_path}/datasets_cache",
        "logs": f"{base_path}/logs",
        "hf_cache": f"{base_path}/hf_cache",
    }

    # Create directories
    for name, path in paths.items():
        os.makedirs(path, exist_ok=True)

    # Set environment variables for HuggingFace
    os.environ["HF_HOME"] = paths["hf_cache"]
    os.environ["HF_DATASETS_CACHE"] = paths["datasets"]
    os.environ["TRANSFORMERS_CACHE"] = paths["hf_cache"]

    return paths


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the most recent checkpoint from a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint-* folders

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, item))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest[1])


def list_checkpoints(base_dir: str) -> list:
    """List all checkpoints in a directory tree.

    Args:
        base_dir: Base directory to search

    Returns:
        List of checkpoint paths, sorted by step number
    """
    import glob

    pattern = os.path.join(base_dir, "**/checkpoint-*")
    checkpoints = glob.glob(pattern, recursive=True)

    # Sort by step number
    def get_step(path):
        try:
            return int(Path(path).name.split("-")[1])
        except (ValueError, IndexError):
            return 0

    return sorted(checkpoints, key=get_step)


def estimate_training_time(
    num_samples: int,
    num_epochs: int,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    gpu_tier: str = "t4",
) -> float:
    """Estimate training time in hours.

    Args:
        num_samples: Number of training samples
        num_epochs: Number of epochs
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        gpu_tier: GPU tier for speed estimation

    Returns:
        Estimated training time in hours
    """
    # Approximate steps per minute by GPU tier (empirical)
    steps_per_minute = {
        "a100": 8.0,
        "v100": 4.0,
        "t4": 2.0,
        "cpu": 0.1,
    }

    effective_batch_size = batch_size * gradient_accumulation
    total_steps = (num_samples * num_epochs) / effective_batch_size

    spm = steps_per_minute.get(gpu_tier, 2.0)
    hours = total_steps / (spm * 60)

    return hours


def check_session_limits(
    estimated_hours: float,
    tier: str = "free"
) -> dict:
    """Check if training fits within Colab session limits.

    Args:
        estimated_hours: Estimated training time
        tier: Colab tier ("free", "pro", "pro+")

    Returns:
        Dict with limit info and recommendations
    """
    limits = {
        "free": {"runtime": 12, "idle": 1.5},
        "pro": {"runtime": 24, "idle": 24},
        "pro+": {"runtime": 24, "idle": 24},
    }

    limit = limits.get(tier, limits["free"])
    fits = estimated_hours < limit["runtime"]

    return {
        "estimated_hours": round(estimated_hours, 2),
        "runtime_limit": limit["runtime"],
        "idle_timeout": limit["idle"],
        "fits_in_session": fits,
        "recommendation": (
            "Training should complete in one session"
            if fits
            else f"Consider reducing samples or using checkpointing "
                 f"(need {estimated_hours:.1f}h, limit {limit['runtime']}h)"
        )
    }


def get_checkpoint_size(checkpoint_path: str) -> float:
    """Get total size of a checkpoint in MB.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Size in megabytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(checkpoint_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)


def print_training_summary(
    stage: str,
    gpu_tier: str,
    num_samples: int,
    num_epochs: int,
    output_dir: str,
    resume_from: Optional[str] = None,
):
    """Print a summary of training configuration.

    Args:
        stage: Training stage (sft, dpo, grpo)
        gpu_tier: Detected GPU tier
        num_samples: Number of samples
        num_epochs: Number of epochs
        output_dir: Output directory
        resume_from: Checkpoint to resume from
    """
    gpu_config = get_gpu_config(gpu_tier)
    est_hours = estimate_training_time(
        num_samples,
        num_epochs,
        gradient_accumulation=gpu_config["gradient_accumulation_steps"],
        gpu_tier=gpu_tier
    )
    session_info = check_session_limits(est_hours)

    print("=" * 50)
    print(f"Training Summary: {stage.upper()}")
    print("=" * 50)
    print(f"GPU Tier: {gpu_tier.upper()}")
    print(f"Samples: {num_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Estimated Time: {est_hours:.1f} hours")
    print(f"Session Fit: {'Yes' if session_info['fits_in_session'] else 'No'}")
    print(f"Output: {output_dir}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 50)

    if not session_info["fits_in_session"]:
        print(f"WARNING: {session_info['recommendation']}")


if __name__ == "__main__":
    # Quick test
    print(f"Running in Colab: {is_colab()}")
    print(f"GPU Tier: {detect_gpu_tier()}")

    gpu_cfg = get_gpu_config(detect_gpu_tier())
    print(f"Recommended config: {gpu_cfg}")

    est = estimate_training_time(1000, 1)
    print(f"Estimated time for 1000 samples, 1 epoch: {est:.2f} hours")
