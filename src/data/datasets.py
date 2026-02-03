"""Dataset loading and preprocessing for VLM training."""

import io
import json
from typing import Optional

import requests
from datasets import Dataset, concatenate_datasets, load_dataset
from PIL import Image


def _download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """Download an image from a URL and return as PIL Image.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        PIL Image or None if download fails
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


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
    return Dataset.from_dict(
        {
            "image": [],
            "question": [],
            "answer": [],
            "messages": [],
        }
    )


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
        return Dataset.from_dict(
            {
                "image": [],
                "question": [],
                "answer": [],
                "messages": [],
            }
        )

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


def load_llava_instruct(
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load LLaVA-Instruct-150K dataset for SFT.

    Contains 150K visual instruction-following examples with conversations.
    Images are COCO filenames downloaded at load time.

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        Dataset formatted for SFT training
    """
    ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_for_sft(example):
        # Extract first human/gpt turn from conversations
        conversations = example.get("conversations", [])
        question = ""
        answer = ""
        for turn in conversations:
            if turn["from"] == "human" and not question:
                # Strip <image>\n prefix if present
                value = turn["value"]
                value = value.replace("<image>\n", "").replace("<image>", "").strip()
                question = value
            elif turn["from"] == "gpt" and not answer:
                answer = turn["value"]

        # Download COCO image
        filename = example.get("image", "")
        url = f"http://images.cocodataset.org/train2017/{filename}"
        image = _download_image(url)

        if image is None:
            return {"images": [], "question": "", "answer": "", "messages": []}

        return {
            "images": [image],
            "question": question,
            "answer": answer,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    ds = ds.map(format_for_sft, remove_columns=ds.column_names)
    # Filter out failed downloads
    ds = ds.filter(lambda x: len(x["question"]) > 0)
    return ds


def load_pixmo_points(
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load PixMo-Points dataset for SFT.

    Contains 2.38M pointing/counting examples. Images are downloaded from URLs.
    Converted to Q&A format based on collection_method (counting vs pointing).

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        Dataset formatted for SFT training
    """
    ds = load_dataset("allenai/pixmo-points", split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_for_sft(example):
        label = example.get("label", "objects")
        count = example.get("count", 0)
        method = example.get("collection_method", "counting")

        # Create Q&A based on collection method
        if method == "counting":
            question = f"How many {label} are in this image?"
            answer = str(count)
        else:
            # Pointing - format coordinates
            points = example.get("points", [])
            question = f"Point to all {label} in this image."
            if points:
                coords = [f"({p['x']:.1f}, {p['y']:.1f})" for p in points]
                answer = f"There are {count} {label}: " + ", ".join(coords)
            else:
                answer = f"There are {count} {label} in the image."

        # Download image from URL
        url = example.get("image_url", "")
        image = _download_image(url)

        if image is None:
            return {"images": [], "question": "", "answer": "", "messages": []}

        return {
            "images": [image],
            "question": question,
            "answer": answer,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    ds = ds.map(format_for_sft, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["question"]) > 0)
    return ds


def load_sharegpt4v(
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load ShareGPT4V dataset for SFT.

    Contains 102K GPT4-Vision-powered captions for COCO images.
    Images are downloaded from COCO server at load time.

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        Dataset formatted for SFT training
    """
    ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_for_sft(example):
        # Extract first human/gpt turn
        conversations = example.get("conversations", [])
        question = ""
        answer = ""
        for turn in conversations:
            if turn["from"] == "human" and not question:
                value = turn["value"]
                value = value.replace("<image>\n", "").replace("<image>", "").strip()
                question = value
            elif turn["from"] == "gpt" and not answer:
                answer = turn["value"]

        # Parse image path to get COCO URL
        # Paths are like "coco/train2017/000000000009.jpg"
        image_path = example.get("image", "")
        if "coco/" in image_path:
            # Extract the COCO path portion
            coco_part = image_path.split("coco/", 1)[1]
            url = f"http://images.cocodataset.org/{coco_part}"
        else:
            url = ""

        image = _download_image(url) if url else None

        if image is None:
            return {"images": [], "question": "", "answer": "", "messages": []}

        return {
            "images": [image],
            "question": question,
            "answer": answer,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    ds = ds.map(format_for_sft, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["question"]) > 0)
    return ds


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
        "llava_instruct": load_llava_instruct,
        "pixmo_points": load_pixmo_points,
        "sharegpt4v": load_sharegpt4v,
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
        return Dataset.from_dict(
            {
                "image": [],
                "caption": [],
                "label": [],
            }
        )

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
        return Dataset.from_dict(
            {
                "image": [],
                "question": [],
                "answer": [],
            }
        )

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
