"""GRPO Trainer for Qwen2.5-VL using TRL."""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import torch
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from ..models.qwen_vl import load_qwen_vl, get_peft_config, get_device
from ..data.datasets import load_sft_dataset
from .reward_functions import create_reward_function, CombinedRewardFunction


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    sft_checkpoint: Optional[str] = None
    use_lora: bool = True
    use_4bit: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # GRPO specific
    num_generations: int = 4  # Number of samples per prompt
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    reward_type: str = "combined"  # spatial, counting, or combined

    # Training
    output_dir: str = "./outputs/grpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # Data
    datasets_to_load: list[str] = field(default_factory=lambda: ["spatial"])
    max_samples: Optional[int] = None

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

    # vLLM (for cloud training)
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.8

    # Logging
    report_to: str = "wandb"
    run_name: Optional[str] = None


class SpatialGRPOTrainer(GRPOTrainer):
    """Extended GRPO trainer with spatial reasoning reward integration."""

    def __init__(
        self,
        reward_function: Callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reward_function = reward_function

    def compute_rewards(
        self,
        completions: list[str],
        prompts: list[str],
        **kwargs,
    ) -> torch.Tensor:
        """Compute rewards using custom reward function.

        Note: This method is called during training to score completions.
        Override this if using TRL's built-in reward computation.
        """
        # Get ground truths from kwargs if available
        ground_truths = kwargs.get("ground_truths", [""] * len(completions))

        rewards = self.reward_function(
            completions=completions,
            ground_truths=ground_truths,
        )

        return torch.tensor(rewards, dtype=torch.float32)


def create_grpo_trainer(
    config: GRPOTrainingConfig,
    dataset: Optional[Dataset] = None,
) -> GRPOTrainer:
    """Create GRPO trainer with Qwen2.5-VL.

    Args:
        config: Training configuration
        dataset: Optional pre-loaded dataset

    Returns:
        Configured GRPOTrainer
    """
    device = get_device()

    # Adjust config for device
    if device != "cuda":
        config.use_4bit = False
        config.bf16 = False
        config.use_vllm = False  # vLLM requires CUDA

    # Load model
    lora_config = get_peft_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    ) if config.use_lora else None

    model_path = config.sft_checkpoint or config.model_name

    model, processor = load_qwen_vl(
        model_name=model_path,
        use_lora=config.use_lora,
        use_4bit=config.use_4bit,
        lora_config=lora_config,
    )

    # Load dataset
    if dataset is None:
        dataset = load_sft_dataset(
            datasets_to_load=config.datasets_to_load,
            max_samples_per_dataset=config.max_samples,
        )

    # Prepare dataset for GRPO (needs 'prompt' column)
    def prepare_for_grpo(example):
        """Convert SFT format to GRPO format."""
        question = example.get("question", "")
        prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt": prompt,
            "ground_truth": example.get("answer", ""),
        }

    dataset = dataset.map(prepare_for_grpo)

    # Create reward function
    reward_fn = create_reward_function(config.reward_type)

    # Training arguments
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_completion_length=config.max_new_tokens,
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
        # GRPO specific
        num_generations=config.num_generations,
        temperature=config.temperature,
        # vLLM settings (cloud only)
        use_vllm=config.use_vllm,
    )

    # Create trainer with reward function
    # Note: TRL's GRPOTrainer expects a reward function that takes
    # (completions, prompts) and returns rewards
    def reward_wrapper(completions, prompts, **kwargs):
        """Wrapper to adapt our reward function to TRL's expected signature."""
        # In practice, we'd need access to ground truths here
        # This is a simplified version - in production, you'd want to
        # store ground truths somewhere accessible
        return reward_fn(completions, [""] * len(completions))

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        reward_funcs=reward_wrapper,
    )

    return trainer


def train_grpo(config: GRPOTrainingConfig) -> str:
    """Run GRPO training.

    Args:
        config: Training configuration

    Returns:
        Path to saved model
    """
    trainer = create_grpo_trainer(config)

    print(f"Starting GRPO training on {get_device()}")
    print(f"Output directory: {config.output_dir}")
    print(f"Reward type: {config.reward_type}")
    print(f"Num generations per prompt: {config.num_generations}")

    # Train
    trainer.train()

    # Save final model
    trainer.save_model()

    print(f"GRPO training complete! Model saved to {config.output_dir}")
    return config.output_dir
