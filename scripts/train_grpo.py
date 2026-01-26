#!/usr/bin/env python3
"""GRPO training script for Qwen2.5-VL spatial reasoning."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.grpo_trainer import GRPOTrainingConfig, train_grpo


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_config_from_yaml(yaml_config: dict, args) -> GRPOTrainingConfig:
    """Create GRPOTrainingConfig from YAML config and CLI args."""
    model_cfg = yaml_config.get("model", {})
    grpo_cfg = yaml_config.get("grpo", {})
    data_cfg = yaml_config.get("data", {})
    training_cfg = yaml_config.get("training", {})
    hardware_cfg = yaml_config.get("hardware", {})
    logging_cfg = yaml_config.get("logging", {})
    debug_cfg = yaml_config.get("debug", {})

    max_samples = data_cfg.get("max_samples_per_dataset")
    if args.debug or debug_cfg.get("enabled", False):
        max_samples = args.max_samples or debug_cfg.get("max_samples", 30)

    lora_cfg = model_cfg.get("lora", {})

    return GRPOTrainingConfig(
        model_name=model_cfg.get("name", "Qwen/Qwen2.5-VL-3B-Instruct"),
        sft_checkpoint=args.sft_checkpoint or model_cfg.get("sft_checkpoint"),
        use_lora=model_cfg.get("use_lora", True),
        use_4bit=model_cfg.get("use_4bit", True),
        lora_r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        num_generations=grpo_cfg.get("num_generations", 4),
        temperature=grpo_cfg.get("temperature", 0.7),
        top_p=grpo_cfg.get("top_p", 0.9),
        max_new_tokens=grpo_cfg.get("max_new_tokens", 256),
        reward_type=args.reward_type or grpo_cfg.get("reward_type", "combined"),
        output_dir=args.output_dir or training_cfg.get("output_dir", "./outputs/grpo"),
        num_train_epochs=training_cfg.get("num_epochs", 1),
        per_device_train_batch_size=training_cfg.get("batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=training_cfg.get("learning_rate", 1e-6),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        max_length=training_cfg.get("max_length", 2048),
        logging_steps=logging_cfg.get("steps", 10),
        save_steps=logging_cfg.get("save_steps", 500),
        save_total_limit=logging_cfg.get("save_total_limit", 3),
        datasets_to_load=data_cfg.get("datasets", ["spatial"]),
        max_samples=max_samples,
        bf16=hardware_cfg.get("bf16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=hardware_cfg.get("dataloader_num_workers", 4),
        use_vllm=hardware_cfg.get("use_vllm", False),
        vllm_gpu_memory_utilization=hardware_cfg.get("vllm_gpu_memory_utilization", 0.8),
        report_to="none" if args.no_wandb else logging_cfg.get("report_to", "wandb"),
        run_name=args.run_name or logging_cfg.get("run_name"),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with GRPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default=None,
        help="Path to SFT/DPO checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        choices=["spatial", "counting", "combined"],
        default=None,
        help="Type of reward function",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use MLX for debug mode (fast local inference on Apple Silicon)",
    )
    parser.add_argument(
        "--mlx_model",
        type=str,
        default="3b-4bit",
        help="MLX model to use (3b-4bit, 7b-8bit, etc.)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples (debug)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override run name",
    )
    args = parser.parse_args()

    # Handle MLX debug mode
    if args.debug and args.mlx:
        from src.training.mlx_debug import debug_grpo_with_mlx

        yaml_config = load_config(args.config)
        data_cfg = yaml_config.get("data", {})
        grpo_cfg = yaml_config.get("grpo", {})
        max_samples = args.max_samples or 5

        results = debug_grpo_with_mlx(
            model_name=args.mlx_model,
            datasets_to_load=data_cfg.get("datasets", ["spatial"]),
            reward_type=args.reward_type or grpo_cfg.get("reward_type", "combined"),
            max_samples=max_samples,
            num_generations=grpo_cfg.get("num_generations", 2),
        )

        if results["status"] == "success":
            print("\nMLX debug validation passed!")
        else:
            print("\nMLX debug validation failed!")
            sys.exit(1)
        return

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    yaml_config = load_config(args.config)
    config = create_config_from_yaml(yaml_config, args)

    print("=" * 60)
    print("GRPO Training Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"SFT checkpoint: {config.sft_checkpoint or 'None'}")
    print(f"Reward type: {config.reward_type}")
    print(f"Num generations: {config.num_generations}")
    print(f"Temperature: {config.temperature}")
    print(f"Use vLLM: {config.use_vllm}")
    print(f"Output: {config.output_dir}")
    print(f"Max samples: {config.max_samples or 'all'}")
    print("=" * 60)

    output_path = train_grpo(config)
    print(f"\nGRPO training complete! Model saved to: {output_path}")


if __name__ == "__main__":
    main()
