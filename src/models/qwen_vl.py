"""Qwen3-VL model loading utilities."""

from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)

# Handle transformers version differences
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq
from peft import LoraConfig, TaskType, get_peft_model


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_quantization_config(use_4bit: bool = True) -> Optional[BitsAndBytesConfig]:
    """Get BitsAndBytes quantization config for QLoRA.

    Note: bitsandbytes does not support MPS, returns None for non-CUDA devices.
    """
    if not torch.cuda.is_available():
        return None

    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def get_peft_config(
    r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> LoraConfig:
    """Get LoRA configuration for Qwen3-VL.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        LoraConfig for PEFT
    """
    if target_modules is None:
        # Default target modules for Qwen3-VL
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def load_qwen_vl(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    use_lora: bool = True,
    use_4bit: bool = True,
    lora_config: Optional[LoraConfig] = None,
    device_map: Optional[str] = "auto",
    attn_implementation: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> tuple[AutoModelForVision2Seq, AutoProcessor]:
    """Load Qwen3-VL model and processor.

    Args:
        model_name: HuggingFace model name or path
        use_lora: Whether to apply LoRA adapters
        use_4bit: Whether to use 4-bit quantization (CUDA only)
        lora_config: Custom LoRA configuration
        device_map: Device mapping strategy
        attn_implementation: Attention implementation (e.g., "flash_attention_2")

    Returns:
        Tuple of (model, processor)
    """
    device = get_device()

    # Processor — limit image resolution to control vision token count
    processor_kwargs = {"trust_remote_code": True}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float16,
    }

    # Quantization (CUDA only)
    quantization_config = get_quantization_config(use_4bit) if use_4bit else None
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map
    else:
        # For MPS/CPU, load without quantization
        if device == "mps":
            model_kwargs["device_map"] = None  # Manual device placement for MPS
        else:
            model_kwargs["device_map"] = device_map

    # Flash attention (CUDA only)
    if attn_implementation and device == "cuda":
        model_kwargs["attn_implementation"] = attn_implementation

    # Load model - use Auto class to handle Qwen2-VL, Qwen2.5-VL, and Qwen3-VL
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        **model_kwargs,
    )

    # Move to MPS if needed
    if device == "mps" and not quantization_config:
        model = model.to(device)

    # Apply LoRA
    if use_lora:
        if lora_config is None:
            lora_config = get_peft_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    return model, processor


def load_qwen_vl_for_inference(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    adapter_path: Optional[str] = None,
    use_4bit: bool = False,
) -> tuple[AutoModelForVision2Seq, AutoProcessor]:
    """Load Qwen3-VL for inference, optionally with trained adapter.

    Args:
        model_name: Base model name
        adapter_path: Path to trained LoRA adapter
        use_4bit: Whether to use 4-bit quantization

    Returns:
        Tuple of (model, processor)
    """
    device = get_device()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float16,
    }

    if use_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = get_quantization_config(True)
        model_kwargs["device_map"] = "auto"
    elif device == "mps":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if device == "mps":
        model = model.to(device)

    # Load adapter if provided
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, processor
