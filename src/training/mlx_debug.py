"""MLX-based debug/validation for training pipelines.

This module provides fast local validation of training pipelines using MLX
instead of PyTorch. Use this to validate:
- Data loading and preprocessing
- Model generation quality
- Reward function computation (GRPO)
- Preference format validity (DPO)
- Training batch shapes and configurations

Note: This is for validation only, not actual training.
"""

import io
import time
from typing import Optional

from PIL import Image

from ..data.datasets import load_preference_dataset, load_sft_dataset
from ..models.mlx_inference import MLXInference, is_mlx_available
from .reward_functions import create_reward_function
from .validation_utils import (
    ValidationReport,
    ValidationResult,
    ValidationStatus,
    validate_batch_shapes,
    validate_gradient_accumulation,
    validate_lora_config,
)


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


def validate_sft_training(
    datasets_to_load: list[str] = ["rlhfv"],
    max_samples: int = 5,
    max_seq_length: int = 2048,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> ValidationReport:
    """Validate SFT training configuration and data pipeline.

    Checks dataset loading, collator batch shapes, LoRA config, and more.
    Does not require MLX - uses PyTorch components for validation.

    Args:
        datasets_to_load: Datasets to test loading
        max_samples: Number of samples to validate
        max_seq_length: Maximum sequence length
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport(trainer_type="SFT")

    # 1. Dataset loading
    try:
        dataset = load_sft_dataset(
            datasets_to_load=datasets_to_load,
            max_samples_per_dataset=max_samples,
        )
        report.add(
            ValidationResult(
                name="dataset_loading",
                status=ValidationStatus.PASSED,
                message=f"Loaded {len(dataset)} samples from {datasets_to_load}",
            )
        )
    except Exception as e:
        report.add(
            ValidationResult(
                name="dataset_loading",
                status=ValidationStatus.FAILED,
                message=f"Dataset loading failed: {e}",
            )
        )
        return report

    # 2. Dataset columns
    required_cols = {"question", "answer"}
    actual_cols = set(dataset.column_names)
    if not required_cols.issubset(actual_cols):
        report.add(
            ValidationResult(
                name="dataset_columns",
                status=ValidationStatus.FAILED,
                message=f"Missing required columns: {required_cols - actual_cols}",
            )
        )
    else:
        report.add(
            ValidationResult(
                name="dataset_columns",
                status=ValidationStatus.PASSED,
                message=f"Has required columns: {required_cols}",
            )
        )

    # 3. Collator batch shapes (requires processor)
    try:
        from transformers import AutoProcessor

        from ..data.collators import VLMDataCollator

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            trust_remote_code=True,
        )
        collator = VLMDataCollator(
            processor=processor,
            max_length=max_seq_length,
        )

        # Create test batch
        test_features = [dataset[i] for i in range(min(batch_size, len(dataset)))]
        batch = collator(test_features)

        # Validate shapes
        expected_keys = ["input_ids", "attention_mask", "labels"]
        shape_results = validate_batch_shapes(
            batch=batch,
            expected_keys=expected_keys,
            batch_size=min(batch_size, len(dataset)),
            max_seq_length=max_seq_length,
            has_images=True,
        )
        for result in shape_results:
            report.add(result)

    except Exception as e:
        report.add(
            ValidationResult(
                name="collator_validation",
                status=ValidationStatus.FAILED,
                message=f"Collator test failed: {e}",
            )
        )

    # 4. Sequence length analysis
    try:
        seq_lengths = []
        for i in range(min(max_samples, len(dataset))):
            example = dataset[i]
            q_len = len(example.get("question", "").split())
            a_len = len(example.get("answer", "").split())
            # Rough token estimate (words * 1.3)
            estimated_tokens = int((q_len + a_len) * 1.3)
            seq_lengths.append(estimated_tokens)

        avg_len = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
        max_len = max(seq_lengths) if seq_lengths else 0
        truncation_count = sum(1 for length in seq_lengths if length > max_seq_length)

        if truncation_count > len(seq_lengths) * 0.5:
            report.add(
                ValidationResult(
                    name="sequence_length",
                    status=ValidationStatus.WARNING,
                    message=f">{truncation_count}/{len(seq_lengths)} samples may be truncated",
                    details={"avg_estimated_tokens": int(avg_len), "max_estimated": max_len},
                )
            )
        else:
            report.add(
                ValidationResult(
                    name="sequence_length",
                    status=ValidationStatus.PASSED,
                    message=f"Avg ~{int(avg_len)} tokens, max ~{max_len}",
                )
            )
    except Exception as e:
        report.add(
            ValidationResult(
                name="sequence_length",
                status=ValidationStatus.FAILED,
                message=f"Sequence analysis failed: {e}",
            )
        )

    # 5. LoRA configuration
    report.add(validate_lora_config(lora_r, lora_alpha, lora_dropout))

    # 6. Gradient accumulation
    report.add(validate_gradient_accumulation(batch_size, gradient_accumulation_steps))

    return report


def validate_dpo_training(
    max_samples: int = 5,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    beta: float = 0.1,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> ValidationReport:
    """Validate DPO training configuration and data pipeline.

    Checks preference dataset, chosen/rejected distinctness, collator output.

    Args:
        max_samples: Number of samples to validate
        max_length: Maximum total sequence length
        max_prompt_length: Maximum prompt length
        beta: DPO beta parameter
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport(trainer_type="DPO")

    # 1. Preference dataset loading
    try:
        dataset = load_preference_dataset(max_samples=max_samples)
        report.add(
            ValidationResult(
                name="dataset_loading",
                status=ValidationStatus.PASSED,
                message=f"Loaded {len(dataset)} preference pairs",
            )
        )
    except Exception as e:
        report.add(
            ValidationResult(
                name="dataset_loading",
                status=ValidationStatus.FAILED,
                message=f"Dataset loading failed: {e}",
            )
        )
        return report

    # 2. Dataset columns
    required_cols = {"prompt", "chosen", "rejected"}
    actual_cols = set(dataset.column_names)
    if not required_cols.issubset(actual_cols):
        report.add(
            ValidationResult(
                name="dataset_columns",
                status=ValidationStatus.FAILED,
                message=f"Missing required columns: {required_cols - actual_cols}",
            )
        )
    else:
        report.add(
            ValidationResult(
                name="dataset_columns",
                status=ValidationStatus.PASSED,
                message=f"Has required columns: {required_cols}",
            )
        )

    # 3. Preference pair validation (CRITICAL)
    identical_pairs = 0
    empty_chosen = 0
    empty_rejected = 0

    for i in range(len(dataset)):
        example = dataset[i]
        chosen = example.get("chosen", "").strip()
        rejected = example.get("rejected", "").strip()

        if chosen == rejected:
            identical_pairs += 1
        if not chosen:
            empty_chosen += 1
        if not rejected:
            empty_rejected += 1

    if identical_pairs > 0:
        report.add(
            ValidationResult(
                name="preference_distinctness",
                status=ValidationStatus.FAILED,
                message=f"{identical_pairs}/{len(dataset)} pairs have identical chosen/rejected!",
                details={"identical_pairs": identical_pairs},
            )
        )
    else:
        report.add(
            ValidationResult(
                name="preference_distinctness",
                status=ValidationStatus.PASSED,
                message="All preference pairs are distinct",
            )
        )

    if empty_chosen > 0 or empty_rejected > 0:
        report.add(
            ValidationResult(
                name="preference_completeness",
                status=ValidationStatus.FAILED,
                message=f"Empty responses: {empty_chosen} chosen, {empty_rejected} rejected",
            )
        )
    else:
        report.add(
            ValidationResult(
                name="preference_completeness",
                status=ValidationStatus.PASSED,
                message="All responses non-empty",
            )
        )

    # 4. Sequence length validation
    over_max_length = 0
    over_prompt_length = 0

    for i in range(len(dataset)):
        example = dataset[i]
        prompt_words = len(example.get("prompt", "").split())
        chosen_words = len(example.get("chosen", "").split())
        rejected_words = len(example.get("rejected", "").split())

        # Rough token estimates
        prompt_tokens = int(prompt_words * 1.3)
        chosen_tokens = int(chosen_words * 1.3)
        rejected_tokens = int(rejected_words * 1.3)

        if prompt_tokens > max_prompt_length:
            over_prompt_length += 1
        if prompt_tokens + max(chosen_tokens, rejected_tokens) > max_length:
            over_max_length += 1

    if over_max_length > len(dataset) * 0.3:
        report.add(
            ValidationResult(
                name="sequence_lengths",
                status=ValidationStatus.WARNING,
                message=f"{over_max_length}/{len(dataset)} samples may exceed max_length={max_length}",
            )
        )
    else:
        report.add(
            ValidationResult(
                name="sequence_lengths",
                status=ValidationStatus.PASSED,
                message=f"Sequence lengths within bounds (max_length={max_length})",
            )
        )

    # 5. DPO collator validation
    try:
        from transformers import AutoProcessor

        from ..data.collators import DPODataCollator

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            trust_remote_code=True,
        )
        collator = DPODataCollator(processor=processor, max_length=max_length)

        test_features = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collator(test_features)

        required_keys = {"prompt", "chosen", "rejected"}
        if required_keys.issubset(set(batch.keys())):
            report.add(
                ValidationResult(
                    name="collator_output",
                    status=ValidationStatus.PASSED,
                    message=f"Collator produces {list(batch.keys())}",
                )
            )
        else:
            report.add(
                ValidationResult(
                    name="collator_output",
                    status=ValidationStatus.FAILED,
                    message=f"Missing keys: {required_keys - set(batch.keys())}",
                )
            )
    except Exception as e:
        report.add(
            ValidationResult(
                name="collator_output",
                status=ValidationStatus.FAILED,
                message=f"Collator test failed: {e}",
            )
        )

    # 6. Beta parameter check
    if beta < 0.01:
        report.add(
            ValidationResult(
                name="dpo_beta",
                status=ValidationStatus.WARNING,
                message=f"Beta={beta} is very small (weak KL constraint)",
            )
        )
    elif beta > 1.0:
        report.add(
            ValidationResult(
                name="dpo_beta",
                status=ValidationStatus.WARNING,
                message=f"Beta={beta} is large (strong KL constraint may limit learning)",
            )
        )
    else:
        report.add(
            ValidationResult(
                name="dpo_beta",
                status=ValidationStatus.PASSED,
                message=f"Beta={beta} is in typical range [0.01, 1.0]",
            )
        )

    # 7. LoRA configuration
    report.add(validate_lora_config(lora_r, lora_alpha, lora_dropout))

    return report


def validate_grpo_training(
    datasets_to_load: list[str] = ["rlhfv"],
    reward_type: str = "combined",
    max_samples: int = 5,
    num_generations: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> ValidationReport:
    """Validate GRPO training configuration and data pipeline.

    Checks dataset, reward function, generation diversity, and more.

    Args:
        datasets_to_load: Datasets to test loading
        reward_type: Reward function type
        max_samples: Number of samples to validate
        num_generations: Generations per prompt
        temperature: Generation temperature
        max_new_tokens: Max tokens per generation
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport(trainer_type="GRPO")

    # 1. Dataset loading
    try:
        dataset = load_sft_dataset(
            datasets_to_load=datasets_to_load,
            max_samples_per_dataset=max_samples,
        )
        report.add(
            ValidationResult(
                name="dataset_loading",
                status=ValidationStatus.PASSED,
                message=f"Loaded {len(dataset)} samples from {datasets_to_load}",
            )
        )
    except Exception as e:
        report.add(
            ValidationResult(
                name="dataset_loading",
                status=ValidationStatus.FAILED,
                message=f"Dataset loading failed: {e}",
            )
        )
        return report

    # 2. Dataset columns
    required_cols = {"question", "answer"}
    actual_cols = set(dataset.column_names)
    if not required_cols.issubset(actual_cols):
        report.add(
            ValidationResult(
                name="dataset_columns",
                status=ValidationStatus.FAILED,
                message=f"Missing required columns: {required_cols - actual_cols}",
            )
        )
    else:
        report.add(
            ValidationResult(
                name="dataset_columns",
                status=ValidationStatus.PASSED,
                message=f"Has required columns: {required_cols}",
            )
        )

    # 3. GRPO format conversion
    try:

        def prepare_for_grpo(example):
            question = example.get("question", "")
            prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
            return {"prompt": prompt, "ground_truth": example.get("answer", "")}

        grpo_dataset = dataset.map(prepare_for_grpo)

        if "prompt" in grpo_dataset.column_names:
            report.add(
                ValidationResult(
                    name="grpo_format_conversion",
                    status=ValidationStatus.PASSED,
                    message="Dataset converted to GRPO format (has 'prompt' column)",
                )
            )
        else:
            report.add(
                ValidationResult(
                    name="grpo_format_conversion",
                    status=ValidationStatus.FAILED,
                    message="Conversion failed: no 'prompt' column",
                )
            )
    except Exception as e:
        report.add(
            ValidationResult(
                name="grpo_format_conversion",
                status=ValidationStatus.FAILED,
                message=f"Conversion failed: {e}",
            )
        )

    # 4. GRPO collator validation
    try:
        from transformers import AutoProcessor

        from ..data.collators import GRPODataCollator

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            trust_remote_code=True,
        )
        collator = GRPODataCollator(processor=processor)

        test_features = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collator(test_features)

        required_keys = {"input_ids", "attention_mask", "ground_truths"}
        actual_keys = set(batch.keys())
        if required_keys.issubset(actual_keys):
            report.add(
                ValidationResult(
                    name="collator_output",
                    status=ValidationStatus.PASSED,
                    message=f"Collator produces {list(batch.keys())}",
                )
            )
        else:
            report.add(
                ValidationResult(
                    name="collator_output",
                    status=ValidationStatus.FAILED,
                    message=f"Missing keys: {required_keys - actual_keys}",
                )
            )
    except Exception as e:
        report.add(
            ValidationResult(
                name="collator_output",
                status=ValidationStatus.FAILED,
                message=f"Collator test failed: {e}",
            )
        )

    # 5. Reward function validation
    try:
        reward_fn = create_reward_function(reward_type)

        # Test with sample completions
        test_completions = ["The cat is on the left", "There are 3 objects", ""]
        test_ground_truths = ["The cat is on the left", "3", "something"]

        rewards = reward_fn(test_completions, test_ground_truths)

        # Check rewards are in valid range
        all_valid = all(-1.0 <= r <= 1.0 for r in rewards)
        if all_valid:
            reward_strs = [f"{r:.2f}" for r in rewards]
            report.add(
                ValidationResult(
                    name="reward_function",
                    status=ValidationStatus.PASSED,
                    message=f"Reward function '{reward_type}' returns valid scores: {reward_strs}",
                )
            )
        else:
            report.add(
                ValidationResult(
                    name="reward_function",
                    status=ValidationStatus.FAILED,
                    message=f"Rewards out of range [-1, 1]: {rewards}",
                )
            )
    except Exception as e:
        report.add(
            ValidationResult(
                name="reward_function",
                status=ValidationStatus.FAILED,
                message=f"Reward function failed: {e}",
            )
        )

    # 6. GRPO hyperparameters check
    issues = []
    if num_generations < 2:
        issues.append("num_generations must be >= 2")
    if temperature <= 0:
        issues.append("temperature must be > 0 for diversity")
    if max_new_tokens < 64:
        issues.append(f"max_new_tokens={max_new_tokens} is very small")
    if max_new_tokens > 512:
        issues.append(f"max_new_tokens={max_new_tokens} is large (slow)")

    if issues:
        report.add(
            ValidationResult(
                name="grpo_hyperparameters",
                status=ValidationStatus.FAILED if num_generations < 2 else ValidationStatus.WARNING,
                message="; ".join(issues),
            )
        )
    else:
        report.add(
            ValidationResult(
                name="grpo_hyperparameters",
                status=ValidationStatus.PASSED,
                message=f"num_generations={num_generations}, temperature={temperature}",
            )
        )

    # 7. LoRA configuration
    report.add(validate_lora_config(lora_r, lora_alpha, lora_dropout))

    return report


def debug_sft_with_mlx(
    model_name: str = "3b-4bit",
    datasets_to_load: list[str] = ["rlhfv"],
    max_samples: int = 5,
) -> dict:
    """Validate SFT pipeline using MLX inference.

    Runs comprehensive training validation first, then tests:
    - Dataset loading works
    - Images and questions are properly formatted
    - Model can generate reasonable responses

    Args:
        model_name: MLX model to use (3b-4bit, 7b-8bit, etc.)
        datasets_to_load: Datasets to test
        max_samples: Number of samples to test

    Returns:
        Dict with test results including validation_report
    """
    # Run training validation first
    print("\n" + "=" * 60)
    print("Running Training Validation...")
    print("=" * 60)

    validation_report = validate_sft_training(
        datasets_to_load=datasets_to_load,
        max_samples=max_samples,
    )
    validation_report.print_report()

    results = {
        "status": "success",
        "validation_passed": validation_report.passed,
        "validation_report": validation_report,
        "samples_tested": 0,
        "samples_with_responses": 0,
        "errors": [],
        "examples": [],
    }

    # Stop early if critical validation failed
    if not validation_report.passed:
        results["status"] = "failed"
        results["errors"].append("Training validation failed - fix issues before proceeding")
        return results

    # Check MLX availability
    if not is_mlx_available():
        raise ImportError("MLX not available. Install with: pip install mlx-vlm")

    print("\n" + "=" * 60)
    print("SFT Debug Mode (MLX Inference)")
    print("=" * 60)

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
    print(f"Validation: {'PASSED' if results['validation_passed'] else 'FAILED'}")
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

    Runs comprehensive training validation first, then tests:
    - Preference dataset loading works
    - Chosen/rejected responses are properly formatted
    - Model can generate responses for comparison

    Args:
        model_name: MLX model to use
        max_samples: Number of samples to test

    Returns:
        Dict with test results including validation_report
    """
    # Run training validation first
    print("\n" + "=" * 60)
    print("Running Training Validation...")
    print("=" * 60)

    validation_report = validate_dpo_training(max_samples=max_samples)
    validation_report.print_report()

    results = {
        "status": "success",
        "validation_passed": validation_report.passed,
        "validation_report": validation_report,
        "samples_tested": 0,
        "preference_pairs_valid": 0,
        "errors": [],
        "examples": [],
    }

    # Stop early if critical validation failed
    if not validation_report.passed:
        results["status"] = "failed"
        results["errors"].append("Training validation failed - fix issues before proceeding")
        return results

    # Check MLX availability
    if not is_mlx_available():
        raise ImportError("MLX not available. Install with: pip install mlx-vlm")

    print("\n" + "=" * 60)
    print("DPO Debug Mode (MLX Inference)")
    print("=" * 60)

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
    print(f"Validation: {'PASSED' if results['validation_passed'] else 'FAILED'}")
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

    Runs comprehensive training validation first, then tests:
    - Dataset loading works
    - Model can generate multiple responses per prompt
    - Reward function computes valid scores
    - Generation diversity and reward variance

    Args:
        model_name: MLX model to use
        datasets_to_load: Datasets to test
        reward_type: Reward function type (spatial, counting, combined)
        max_samples: Number of samples to test
        num_generations: Responses per prompt to generate

    Returns:
        Dict with test results including validation_report
    """
    # Run training validation first
    print("\n" + "=" * 60)
    print("Running Training Validation...")
    print("=" * 60)

    validation_report = validate_grpo_training(
        datasets_to_load=datasets_to_load,
        reward_type=reward_type,
        max_samples=max_samples,
        num_generations=num_generations,
    )
    validation_report.print_report()

    results = {
        "status": "success",
        "validation_passed": validation_report.passed,
        "validation_report": validation_report,
        "samples_tested": 0,
        "total_generations": 0,
        "reward_computations": 0,
        "avg_reward": 0.0,
        "errors": [],
        "examples": [],
    }

    # Stop early if critical validation failed
    if not validation_report.passed:
        results["status"] = "failed"
        results["errors"].append("Training validation failed - fix issues before proceeding")
        return results

    # Check MLX availability
    if not is_mlx_available():
        raise ImportError("MLX not available. Install with: pip install mlx-vlm")

    print("\n" + "=" * 60)
    print("GRPO Debug Mode (MLX Inference)")
    print("=" * 60)

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

        # Track diversity
        unique_gens = len(set(generations))
        results["examples"].append({
            "question": question[:50],
            "ground_truth": ground_truth[:50],
            "generations": [g[:40] for g in generations],
            "rewards": rewards,
            "unique_generations": unique_gens,
        })

    # Compute statistics
    if all_rewards:
        results["avg_reward"] = sum(all_rewards) / len(all_rewards)

    # Compute diversity and variance stats
    total_unique = sum(ex.get("unique_generations", 0) for ex in results["examples"])
    total_possible = results["samples_tested"] * num_generations
    diversity_ratio = total_unique / total_possible if total_possible > 0 else 0

    import statistics
    reward_std = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0

    results["diversity_ratio"] = diversity_ratio
    results["reward_std"] = reward_std

    # Summary
    print("\n" + "=" * 60)
    print("GRPO Debug Summary")
    print("=" * 60)
    print(f"Validation: {'PASSED' if results['validation_passed'] else 'FAILED'}")
    print(f"Status: {results['status']}")
    print(f"Samples tested: {results['samples_tested']}")
    print(f"Total generations: {results['total_generations']}")
    print(f"Reward computations: {results['reward_computations']}")
    print(f"Average reward: {results['avg_reward']:.3f}")
    print(f"Reward std: {reward_std:.3f}")
    print(f"Diversity ratio: {diversity_ratio:.1%}")
    print(f"Errors: {len(results['errors'])}")

    return results
