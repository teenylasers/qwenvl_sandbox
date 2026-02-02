#!/usr/bin/env python3
"""Evaluation script for trained VLM models on spatial reasoning benchmarks."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on spatial reasoning benchmarks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (if using fine-tuned model)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["VSR", "CV-Bench"],
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (for quick testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (CUDA only)",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use MLX backend for fast evaluation on Apple Silicon",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="2b-4bit",
        help="MLX model shorthand or HuggingFace path (default: 2b-4bit)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VLM Spatial Reasoning Evaluation")
    print("=" * 60)

    if args.mlx:
        from src.eval.mlx_eval import evaluate_model_mlx

        print(f"Backend: MLX")
        print(f"Model: {args.mlx_model}")
        print(f"Benchmarks: {args.benchmarks}")
        print(f"Max samples: {args.max_samples or 'all'}")
        print("=" * 60)

        results = evaluate_model_mlx(
            mlx_model=args.mlx_model,
            benchmarks=args.benchmarks,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
    else:
        from src.eval.spatial_eval import evaluate_model

        print(f"Backend: PyTorch")
        print(f"Model: {args.model_path}")
        print(f"Adapter: {args.adapter_path or 'None'}")
        print(f"Benchmarks: {args.benchmarks}")
        print(f"Max samples: {args.max_samples or 'all'}")
        print("=" * 60)

        results = evaluate_model(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            benchmarks=args.benchmarks,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
