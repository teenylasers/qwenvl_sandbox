"""DPO Trainer for Qwen3-VL using TRL."""

from typing import Optional
from dataclasses import dataclass, field
import torch
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from ..models.qwen_vl import load_qwen_vl, get_peft_config, get_device
from ..data.datasets import load_preference_dataset


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    sft_checkpoint: Optional[str] = None  # Path to SFT checkpoint
    use_lora: bool = True
    use_4bit: bool = True
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # DPO specific
    beta: float = 0.1  # KL penalty coefficient
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, etc.
    label_smoothing: float = 0.0

    # Training
    output_dir: str = "./outputs/dpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-7
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 2048
    max_prompt_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # Data
    max_samples: Optional[int] = None

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

    # Logging
    report_to: str = "wandb"
    run_name: Optional[str] = None


def create_dpo_trainer(
    config: DPOTrainingConfig,
    dataset: Optional[Dataset] = None,
) -> DPOTrainer:
    """Create DPO trainer with Qwen3-VL.

    Args:
        config: Training configuration
        dataset: Optional pre-loaded preference dataset

    Returns:
        Configured DPOTrainer
    """
    device = get_device()

    # Adjust config for device
    if device != "cuda":
        config.use_4bit = False
        config.bf16 = False

    # Load model
    lora_config = get_peft_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    ) if config.use_lora else None

    # If SFT checkpoint provided, load from there
    model_path = config.sft_checkpoint or config.model_name

    model, processor = load_qwen_vl(
        model_name=model_path,
        use_lora=config.use_lora,
        use_4bit=config.use_4bit,
        lora_config=lora_config,
    )

    # Load reference model (for KL divergence)
    # For LoRA, we can use the same base model
    ref_model = None
    if not config.use_lora:
        ref_model, _ = load_qwen_vl(
            model_name=config.model_name,
            use_lora=False,
            use_4bit=config.use_4bit,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # Load dataset
    if dataset is None:
        dataset = load_preference_dataset(
            max_samples=config.max_samples,
        )

    # Training arguments
    training_args = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16 and device == "cuda",
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to=config.report_to if config.report_to != "none" else None,
        run_name=config.run_name,
        remove_unused_columns=False,
        # DPO specific
        beta=config.beta,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    return trainer


def train_dpo(
    config: DPOTrainingConfig,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Run DPO training.

    Args:
        config: Training configuration
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Path to saved model
    """
    trainer = create_dpo_trainer(config)

    print(f"Starting DPO training on {get_device()}")
    print(f"Output directory: {config.output_dir}")
    print(f"Beta (KL penalty): {config.beta}")
    if resume_from_checkpoint:
        print(f"Resuming from: {resume_from_checkpoint}")

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    trainer.save_model()

    print(f"DPO training complete! Model saved to {config.output_dir}")
    return config.output_dir
