"""Dataset loading and preprocessing for VLM training."""

from typing import Optional, Callable
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image
import io
import json


# =============================================================================
# SFT Dataset Loading
# =============================================================================


def load_rlhfv_sft(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load RLHF-V dataset formatted for SFT (using chosen responses).

    The RLHF-V dataset contains preference pairs. For SFT, we use the chosen
    responses as ground truth.

    Args:
        split: Dataset split to load
        max_samples: Maximum number of samples to load

    Returns:
        Dataset formatted for SFT training
    """
    ds = load_dataset("openbmb/RLHF-V-Dataset", split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_for_sft(example):
        """Convert RLHF-V format to SFT format."""
        text_data = example["text"]
        # Parse JSON string if needed
        if isinstance(text_data, str):
            text_data = json.loads(text_data)
        question = text_data.get("question", "")
        chosen = text_data.get("chosen", "")

        # Handle image
        image_data = example.get("image", {})
        if isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(io.BytesIO(image_data["bytes"]))
        else:
            image = image_data

        return {
            "images": [image],  # TRL expects "images" (plural) as a list
            "question": question,
            "answer": chosen,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": chosen},
            ],
        }

    return ds.map(format_for_sft, remove_columns=ds.column_names)


def load_pixmo_cap(
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load PixMo-Cap dense captioning dataset for SFT.

    Note: PixMo-Cap contains image URLs, not direct images.
    This requires downloading images which is slow - skip for now.
    Use RLHF-V and SpatialVLM which have direct image data.

    Args:
        max_samples: Maximum number of samples

    Returns:
        Dataset formatted for SFT
    """
    # PixMo-Cap only has image URLs, not actual images
    # Skip for now as downloading images is slow
    print("Warning: PixMo-Cap contains URLs, not images. Skipping.")
    return Dataset.from_dict({
        "image": [],
        "question": [],
        "answer": [],
        "messages": [],
    })


def load_spatial_vlm(
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load SpatialVLM dataset for spatial reasoning SFT.

    SpatialVLM provides synthetic spatial VQA data. Check availability at:
    https://huggingface.co/datasets/remyxai/vqasynth_spacial

    Args:
        max_samples: Maximum number of samples

    Returns:
        Dataset formatted for SFT
    """
    try:
        # Try loading from available sources
        ds = load_dataset("remyxai/vqasynth_spacial", split="train")
    except Exception:
        print("Warning: SpatialVLM dataset not available. Using placeholder.")
        return Dataset.from_dict({
            "image": [],
            "question": [],
            "answer": [],
            "messages": [],
        })

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_for_sft(example):
        question = example.get("question", example.get("prompt", ""))
        answer = example.get("answer", example.get("response", ""))
        return {
            "image": example["image"],
            "question": question,
            "answer": answer,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    return ds.map(format_for_sft, remove_columns=ds.column_names)


def load_sft_dataset(
    datasets_to_load: list[str] = ["rlhfv", "pixmo", "spatial"],
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """Load and combine multiple datasets for SFT training.

    Args:
        datasets_to_load: List of dataset names to load
        max_samples_per_dataset: Max samples from each dataset
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed for shuffling

    Returns:
        Combined dataset for SFT training
    """
    loaders = {
        "rlhfv": load_rlhfv_sft,
        "pixmo": load_pixmo_cap,
        "spatial": load_spatial_vlm,
    }

    loaded_datasets = []
    for name in datasets_to_load:
        if name in loaders:
            print(f"Loading {name} dataset...")
            ds = loaders[name](max_samples=max_samples_per_dataset)
            if len(ds) > 0:
                loaded_datasets.append(ds)
                print(f"  Loaded {len(ds)} samples from {name}")

    if not loaded_datasets:
        raise ValueError("No datasets were successfully loaded!")

    combined = concatenate_datasets(loaded_datasets)

    if shuffle:
        combined = combined.shuffle(seed=seed)

    print(f"Total SFT dataset size: {len(combined)}")
    return combined


# =============================================================================
# Preference Dataset Loading (for DPO)
# =============================================================================


def load_rlhfv_preference(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load RLHF-V dataset formatted for DPO preference learning.

    Args:
        split: Dataset split
        max_samples: Maximum samples to load

    Returns:
        Dataset with chosen/rejected pairs for DPO
    """
    ds = load_dataset("openbmb/RLHF-V-Dataset", split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_for_dpo(example):
        """Convert to DPO format with chosen/rejected pairs."""
        text_data = example["text"]
        # Parse JSON string if needed
        if isinstance(text_data, str):
            text_data = json.loads(text_data)
        question = text_data.get("question", "")
        chosen = text_data.get("chosen", "")
        rejected = text_data.get("rejected", "")

        # Handle image
        image_data = example.get("image", {})
        if isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(io.BytesIO(image_data["bytes"]))
        else:
            image = image_data

        return {
            "image": image,
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
        }

    return ds.map(format_for_dpo, remove_columns=ds.column_names)


def load_preference_dataset(
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """Load preference dataset for DPO training.

    Currently uses RLHF-V as the primary preference dataset.

    Args:
        max_samples: Maximum samples to load
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        Dataset with preference pairs
    """
    ds = load_rlhfv_preference(max_samples=max_samples)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    print(f"Loaded {len(ds)} preference pairs for DPO")
    return ds


# =============================================================================
# Evaluation Dataset Loading
# =============================================================================


def load_vsr_benchmark(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load Visual Spatial Reasoning (VSR) benchmark.

    Args:
        split: Dataset split
        max_samples: Maximum samples

    Returns:
        VSR evaluation dataset
    """
    try:
        ds = load_dataset("cambridgeltl/vsr_random", split=split)
    except Exception:
        print("Warning: VSR benchmark not available.")
        return Dataset.from_dict({
            "image": [],
            "caption": [],
            "label": [],
        })

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


def load_cvbench(
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load CV-Bench for comprehensive VLM evaluation.

    Args:
        max_samples: Maximum samples

    Returns:
        CV-Bench evaluation dataset
    """
    try:
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    except Exception:
        print("Warning: CV-Bench not available.")
        return Dataset.from_dict({
            "image": [],
            "question": [],
            "answer": [],
        })

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
