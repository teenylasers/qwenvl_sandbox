"""MLX-based debug/validation for training pipelines.

This module provides fast local validation of training pipelines using MLX
instead of PyTorch. Use this to validate:
- Data loading and preprocessing
- Model generation quality
- Reward function computation (GRPO)
- Preference format validity (DPO)

Note: This is for validation only, not actual training.
"""

import time
from typing import Optional
from PIL import Image
import io

from ..models.mlx_inference import MLXInference, is_mlx_available
from ..data.datasets import load_sft_dataset, load_preference_dataset
from .reward_functions import create_reward_function


def _get_image_from_example(example: dict) -> Optional[Image.Image]:
    """Extract PIL Image from dataset example."""
    # Handle different image formats in datasets
    images = example.get("images", [])
    if images and len(images) > 0:
        img = images[0]
        if isinstance(img, Image.Image):
            return img
        elif isinstance(img, dict) and "bytes" in img:
            return Image.open(io.BytesIO(img["bytes"]))
    return None


def debug_sft_with_mlx(
    model_name: str = "3b-4bit",
    datasets_to_load: list[str] = ["rlhfv"],
    max_samples: int = 5,
) -> dict:
    """Validate SFT pipeline using MLX inference.

    Tests:
    - Dataset loading works
    - Images and questions are properly formatted
    - Model can generate reasonable responses

    Args:
        model_name: MLX model to use (3b-4bit, 7b-8bit, etc.)
        datasets_to_load: Datasets to test
        max_samples: Number of samples to test

    Returns:
        Dict with test results
    """
    if not is_mlx_available():
        raise ImportError("MLX not available. Install with: pip install mlx-vlm")

    print("=" * 60)
    print("SFT Debug Mode (MLX)")
    print("=" * 60)

    results = {
        "status": "success",
        "samples_tested": 0,
        "samples_with_responses": 0,
        "errors": [],
        "examples": [],
    }

    # Load dataset
    print(f"\nLoading datasets: {datasets_to_load}")
    try:
        dataset = load_sft_dataset(
            datasets_to_load=datasets_to_load,
            max_samples_per_dataset=max_samples,
        )
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Dataset loading failed: {e}")
        return results

    # Load MLX model
    print(f"\nLoading MLX model: {model_name}")
    try:
        model = MLXInference(model_name)
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Model loading failed: {e}")
        return results

    # Test each sample
    print(f"\nTesting {min(max_samples, len(dataset))} samples...")
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        results["samples_tested"] += 1
        question = example.get("question", "")
        expected = example.get("answer", "")
        image = _get_image_from_example(example)

        print(f"\n[{i+1}/{max_samples}] Q: {question[:60]}...")

        try:
            start = time.time()
            response = model.generate(question, image=image, max_tokens=100)
            elapsed = time.time() - start

            results["samples_with_responses"] += 1
            print(f"    Response ({elapsed:.2f}s): {response[:80]}...")

            results["examples"].append({
                "question": question,
                "expected": expected[:100],
                "response": response[:100],
                "has_image": image is not None,
            })
        except Exception as e:
            results["errors"].append(f"Sample {i}: {e}")
            print(f"    ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SFT Debug Summary")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Samples tested: {results['samples_tested']}")
    print(f"Successful responses: {results['samples_with_responses']}")
    print(f"Errors: {len(results['errors'])}")

    return results


def debug_dpo_with_mlx(
    model_name: str = "3b-4bit",
    max_samples: int = 5,
) -> dict:
    """Validate DPO pipeline using MLX inference.

    Tests:
    - Preference dataset loading works
    - Chosen/rejected responses are properly formatted
    - Model can generate responses for comparison

    Args:
        model_name: MLX model to use
        max_samples: Number of samples to test

    Returns:
        Dict with test results
    """
    if not is_mlx_available():
        raise ImportError("MLX not available. Install with: pip install mlx-vlm")

    print("=" * 60)
    print("DPO Debug Mode (MLX)")
    print("=" * 60)

    results = {
        "status": "success",
        "samples_tested": 0,
        "preference_pairs_valid": 0,
        "errors": [],
        "examples": [],
    }

    # Load preference dataset
    print("\nLoading preference dataset...")
    try:
        dataset = load_preference_dataset(max_samples=max_samples)
        print(f"Loaded {len(dataset)} preference pairs")
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Dataset loading failed: {e}")
        return results

    # Load MLX model
    print(f"\nLoading MLX model: {model_name}")
    try:
        model = MLXInference(model_name)
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Model loading failed: {e}")
        return results

    # Test each preference pair
    print(f"\nTesting {min(max_samples, len(dataset))} preference pairs...")
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        results["samples_tested"] += 1
        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        image = example.get("image")

        # Handle image
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))

        print(f"\n[{i+1}/{max_samples}] Prompt: {prompt[:50]}...")
        print(f"    Chosen: {chosen[:50]}...")
        print(f"    Rejected: {rejected[:50]}...")

        # Validate pair
        if chosen and rejected and chosen != rejected:
            results["preference_pairs_valid"] += 1

        # Generate model response
        try:
            start = time.time()
            response = model.generate(prompt, image=image, max_tokens=100)
            elapsed = time.time() - start
            print(f"    Model response ({elapsed:.2f}s): {response[:60]}...")

            results["examples"].append({
                "prompt": prompt[:50],
                "chosen": chosen[:50],
                "rejected": rejected[:50],
                "model_response": response[:50],
            })
        except Exception as e:
            results["errors"].append(f"Sample {i}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("DPO Debug Summary")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Samples tested: {results['samples_tested']}")
    print(f"Valid preference pairs: {results['preference_pairs_valid']}")
    print(f"Errors: {len(results['errors'])}")

    return results


def debug_grpo_with_mlx(
    model_name: str = "3b-4bit",
    datasets_to_load: list[str] = ["rlhfv"],
    reward_type: str = "combined",
    max_samples: int = 5,
    num_generations: int = 2,
) -> dict:
    """Validate GRPO pipeline using MLX inference.

    Tests:
    - Dataset loading works
    - Model can generate multiple responses per prompt
    - Reward function computes valid scores

    Args:
        model_name: MLX model to use
        datasets_to_load: Datasets to test
        reward_type: Reward function type (spatial, counting, combined)
        max_samples: Number of samples to test
        num_generations: Responses per prompt to generate

    Returns:
        Dict with test results
    """
    if not is_mlx_available():
        raise ImportError("MLX not available. Install with: pip install mlx-vlm")

    print("=" * 60)
    print("GRPO Debug Mode (MLX)")
    print("=" * 60)

    results = {
        "status": "success",
        "samples_tested": 0,
        "total_generations": 0,
        "reward_computations": 0,
        "avg_reward": 0.0,
        "errors": [],
        "examples": [],
    }

    # Load dataset
    print(f"\nLoading datasets: {datasets_to_load}")
    try:
        dataset = load_sft_dataset(
            datasets_to_load=datasets_to_load,
            max_samples_per_dataset=max_samples,
        )
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Dataset loading failed: {e}")
        return results

    # Load reward function
    print(f"\nLoading reward function: {reward_type}")
    reward_fn = create_reward_function(reward_type)

    # Load MLX model
    print(f"\nLoading MLX model: {model_name}")
    try:
        model = MLXInference(model_name)
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Model loading failed: {e}")
        return results

    # Test each sample
    all_rewards = []
    print(f"\nTesting {min(max_samples, len(dataset))} samples with {num_generations} generations each...")

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        results["samples_tested"] += 1
        question = example.get("question", "")
        ground_truth = example.get("answer", "")
        image = _get_image_from_example(example)

        print(f"\n[{i+1}/{max_samples}] Q: {question[:50]}...")
        print(f"    Ground truth: {ground_truth[:50]}...")

        generations = []
        rewards = []

        for g in range(num_generations):
            try:
                # Generate with some temperature for diversity
                response = model.generate(
                    question,
                    image=image,
                    max_tokens=100,
                    temperature=0.7,
                )
                generations.append(response)
                results["total_generations"] += 1

                # Compute reward
                reward = reward_fn([response], [ground_truth])[0]
                rewards.append(reward)
                all_rewards.append(reward)
                results["reward_computations"] += 1

                print(f"    Gen {g+1}: {response[:40]}... (reward: {reward:.3f})")

            except Exception as e:
                results["errors"].append(f"Sample {i}, Gen {g}: {e}")

        results["examples"].append({
            "question": question[:50],
            "ground_truth": ground_truth[:50],
            "generations": [g[:40] for g in generations],
            "rewards": rewards,
        })

    # Compute average reward
    if all_rewards:
        results["avg_reward"] = sum(all_rewards) / len(all_rewards)

    # Summary
    print("\n" + "=" * 60)
    print("GRPO Debug Summary")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Samples tested: {results['samples_tested']}")
    print(f"Total generations: {results['total_generations']}")
    print(f"Reward computations: {results['reward_computations']}")
    print(f"Average reward: {results['avg_reward']:.3f}")
    print(f"Errors: {len(results['errors'])}")

    return results
