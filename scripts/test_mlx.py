#!/usr/bin/env python3
"""Test script for MLX-based local inference on Apple Silicon.

This script tests the MLX inference pipeline for Qwen3-VL, which provides
fast local inference (~200-400 tok/s) compared to PyTorch MPS (~7-9 tok/s).

Usage:
    # Quick test with default model (4b-4bit)
    python scripts/test_mlx.py

    # Test with specific model
    python scripts/test_mlx.py --model 8b-4bit

    # Test with local image
    python scripts/test_mlx.py --image /path/to/image.jpg

    # Run spatial reasoning test suite
    python scripts/test_mlx.py --spatial --image /path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mlx_inference import (
    MLXInference,
    is_mlx_available,
    MLX_MODELS,
)


def test_basic_inference(model: MLXInference, image_url: str):
    """Test basic image description."""
    print("\n" + "=" * 60)
    print("Basic Inference Test")
    print("=" * 60)

    prompt = "Describe this image in detail."
    print(f"Prompt: {prompt}")
    print(f"Image: {image_url}")

    start = time.time()
    response = model.generate(prompt, image=image_url)
    elapsed = time.time() - start

    print(f"\nResponse ({elapsed:.2f}s):")
    print(response)


def test_spatial_reasoning(model: MLXInference, image_url: str):
    """Test spatial reasoning capabilities."""
    print("\n" + "=" * 60)
    print("Spatial Reasoning Test")
    print("=" * 60)

    questions = [
        "What objects are in this image and where are they located?",
        "Describe the spatial relationships between objects (left, right, above, below).",
        "How many distinct objects can you count in this image?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        start = time.time()
        response = model.generate(question, image=image_url)
        elapsed = time.time() - start
        print(f"Response ({elapsed:.2f}s): {response[:200]}...")


def test_batch_inference(model: MLXInference, image_url: str):
    """Test batch inference performance."""
    print("\n" + "=" * 60)
    print("Batch Inference Test")
    print("=" * 60)

    prompts = [
        "What is the main subject of this image?",
        "What colors dominate this image?",
        "Is this image taken indoors or outdoors?",
    ]

    print(f"Processing {len(prompts)} prompts...")
    start = time.time()
    responses = model.batch_generate(prompts, images=[image_url] * len(prompts))
    elapsed = time.time() - start

    print(f"Total time: {elapsed:.2f}s ({elapsed/len(prompts):.2f}s per prompt)")
    for prompt, response in zip(prompts, responses):
        print(f"\nQ: {prompt}")
        print(f"A: {response[:150]}...")


def test_text_only(model: MLXInference):
    """Test text-only inference (no image)."""
    print("\n" + "=" * 60)
    print("Text-Only Test")
    print("=" * 60)

    prompt = "What are the common spatial relationships used to describe object positions in images?"

    start = time.time()
    response = model.generate(prompt)
    elapsed = time.time() - start

    print(f"Prompt: {prompt}")
    print(f"\nResponse ({elapsed:.2f}s):")
    print(response)


def main():
    parser = argparse.ArgumentParser(
        description="Test MLX inference for Qwen3-VL on Apple Silicon"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="4b-4bit",
        choices=list(MLX_MODELS.keys()),
        help="Model to use (default: 4b-4bit)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL to test image",
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Run spatial reasoning test suite",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch inference test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )
    args = parser.parse_args()

    # Check MLX availability
    if not is_mlx_available():
        print("ERROR: MLX is not available.")
        print("Install with: pip install -e '.[mlx]'")
        print("Note: MLX only works on Apple Silicon Macs.")
        sys.exit(1)

    # Default test image
    image_url = args.image or "http://images.cocodataset.org/val2017/000000039769.jpg"

    print("=" * 60)
    print("MLX Inference Test Suite")
    print("=" * 60)
    print(f"Model: {args.model} ({MLX_MODELS[args.model]})")
    print(f"Image: {image_url}")

    # Load model
    print("\nLoading model...")
    start = time.time()
    model = MLXInference(args.model)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Run tests
    if args.all:
        test_basic_inference(model, image_url)
        test_spatial_reasoning(model, image_url)
        test_batch_inference(model, image_url)
        test_text_only(model)
    elif args.spatial:
        test_spatial_reasoning(model, image_url)
    elif args.batch:
        test_batch_inference(model, image_url)
    else:
        test_basic_inference(model, image_url)

    print("\n" + "=" * 60)
    print("MLX test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
