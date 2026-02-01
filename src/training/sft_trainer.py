"""SFT Trainer for Qwen3-VL using TRL."""

from typing import Optional, Any
from dataclasses import dataclass, field
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from ..models.qwen_vl import load_qwen_vl, get_peft_config, get_device
from ..data.datasets import load_sft_dataset


@dataclass
class SFTTrainingConfig:
    """Configuration for SFT training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    use_lora: bool = True
    use_4bit: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Image resolution (controls vision token count per image)
    min_pixels: Optional[int] = 256 * 28 * 28      # ~256 tokens min
    max_pixels: Optional[int] = 1280 * 28 * 28     # ~1280 tokens max

    # Training
    output_dir: str = "./outputs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 4096
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # Data
    datasets_to_load: list[str] = field(
        default_factory=lambda: ["rlhfv", "pixmo", "spatial"]
    )
    max_samples: Optional[int] = None

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

    # Logging
    report_to: str = "wandb"
    run_name: Optional[str] = None


def create_sft_trainer(
    config: SFTTrainingConfig,
    dataset: Optional[Dataset] = None,
) -> SFTTrainer:
    """Create SFT trainer with Qwen3-VL.

    Args:
        config: Training configuration
        dataset: Optional pre-loaded dataset

    Returns:
        Configured SFTTrainer
    """
    device = get_device()

    # Adjust config for device
    if device != "cuda":
        config.use_4bit = False  # bitsandbytes requires CUDA
        config.bf16 = False

    # Load model
    lora_config = get_peft_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    ) if config.use_lora else None

    model, processor = load_qwen_vl(
        model_name=config.model_name,
        use_lora=config.use_lora,
        use_4bit=config.use_4bit,
        lora_config=lora_config,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )

    # Load dataset
    if dataset is None:
        dataset = load_sft_dataset(
            datasets_to_load=config.datasets_to_load,
            max_samples_per_dataset=config.max_samples,
        )

    # Training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_length=config.max_seq_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16 and device == "cuda",
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to="none" if config.report_to == "none" else config.report_to,
        run_name=config.run_name,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    def formatting_func(examples):
        """Format examples for Qwen3-VL chat."""
        formatted = []
        for i in range(len(examples.get("question", []))):
            question = examples["question"][i]
            answer = examples["answer"][i]
            # Using Qwen chat format
            text = (
                f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>"
            )
            formatted.append(text)
        return formatted

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        formatting_func=formatting_func,
    )

    return trainer


def train_sft(
    config: SFTTrainingConfig,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Run SFT training.

    Args:
        config: Training configuration
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Path to saved model
    """
    trainer = create_sft_trainer(config)

    print(f"Starting SFT training on {get_device()}")
    print(f"Output directory: {config.output_dir}")
    if resume_from_checkpoint:
        print(f"Resuming from: {resume_from_checkpoint}")

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    trainer.save_model()

    print(f"Training complete! Model saved to {config.output_dir}")
    return config.output_dir
