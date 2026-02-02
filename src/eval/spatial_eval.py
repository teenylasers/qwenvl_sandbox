"""Evaluation module for spatial reasoning benchmarks."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset
from tqdm import tqdm

from ..data.datasets import _download_image, load_cvbench, load_vsr_benchmark


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    model_path: str
    adapter_path: Optional[str] = None
    use_4bit: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.1
    batch_size: int = 1
    output_dir: str = "./eval_results"


@dataclass
class EvalResult:
    """Container for evaluation results."""

    benchmark: str
    accuracy: float
    total_samples: int
    correct: int
    predictions: list[dict]


class _BaseSpatialEvaluator:
    """Base evaluator with shared benchmark logic.

    Subclasses must implement __init__ and generate.
    """

    config: Any  # EvalConfig or MLXEvalConfig — must have output_dir

    def generate(self, image, question: str) -> str:
        """Generate response for image-question pair.

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Generated response
        """
        raise NotImplementedError

    def evaluate_vsr(self, max_samples: Optional[int] = None) -> EvalResult:
        """Evaluate on Visual Spatial Reasoning benchmark.

        VSR tests if model understands spatial relationships between objects.

        Args:
            max_samples: Maximum samples to evaluate

        Returns:
            Evaluation results
        """
        print("Evaluating on VSR benchmark...")
        dataset = load_vsr_benchmark(max_samples=max_samples)

        if len(dataset) == 0:
            print("Warning: VSR dataset is empty")
            return EvalResult(
                benchmark="VSR",
                accuracy=0.0,
                total_samples=0,
                correct=0,
                predictions=[],
            )

        correct = 0
        predictions = []

        for example in tqdm(dataset, desc="VSR"):
            image_url = example.get("image_link", "")
            image = _download_image(image_url) if image_url else None
            if image is None:
                continue
            caption = example["caption"]
            label = example["label"]  # True/False

            # Ask model to verify the caption
            question = (
                f"Is this statement about the image TRUE or FALSE?\n"
                f'Statement: "{caption}"\n'
                f"Answer with only TRUE or FALSE."
            )

            response = self.generate(image, question)

            # Parse response
            response_lower = response.lower()
            pred_true = "true" in response_lower and "false" not in response_lower
            pred_false = "false" in response_lower

            if pred_true:
                predicted = True
            elif pred_false:
                predicted = False
            else:
                predicted = None  # Can't determine

            is_correct = predicted == label

            if is_correct:
                correct += 1

            predictions.append(
                {
                    "caption": caption,
                    "label": label,
                    "predicted": predicted,
                    "response": response,
                    "correct": is_correct,
                }
            )

        total = len(predictions)
        accuracy = correct / total if total else 0

        print(f"VSR Accuracy: {accuracy:.2%} ({correct}/{total})")

        return EvalResult(
            benchmark="VSR",
            accuracy=accuracy,
            total_samples=total,
            correct=correct,
            predictions=predictions,
        )

    def evaluate_cvbench(self, max_samples: Optional[int] = None) -> EvalResult:
        """Evaluate on CV-Bench.

        Args:
            max_samples: Maximum samples

        Returns:
            Evaluation results
        """
        print("Evaluating on CV-Bench...")
        dataset = load_cvbench(max_samples=max_samples)

        if len(dataset) == 0:
            print("Warning: CV-Bench dataset is empty")
            return EvalResult(
                benchmark="CV-Bench",
                accuracy=0.0,
                total_samples=0,
                correct=0,
                predictions=[],
            )

        correct = 0
        predictions = []

        for example in tqdm(dataset, desc="CV-Bench"):
            image = example["image"]
            question = example["question"]
            answer = example["answer"]

            response = self.generate(image, question)

            # Check if answer matches (case-insensitive)
            is_correct = answer.lower().strip() in response.lower()

            if is_correct:
                correct += 1

            predictions.append(
                {
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "correct": is_correct,
                }
            )

        accuracy = correct / len(dataset) if dataset else 0

        print(f"CV-Bench Accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")

        return EvalResult(
            benchmark="CV-Bench",
            accuracy=accuracy,
            total_samples=len(dataset),
            correct=correct,
            predictions=predictions,
        )

    def evaluate_spatial_reasoning(
        self,
        dataset: Dataset,
        benchmark_name: str = "custom",
    ) -> EvalResult:
        """Evaluate on custom spatial reasoning dataset.

        Args:
            dataset: Dataset with 'image', 'question', 'answer' columns
            benchmark_name: Name for the benchmark

        Returns:
            Evaluation results
        """
        print(f"Evaluating on {benchmark_name}...")

        correct = 0
        predictions = []

        for example in tqdm(dataset, desc=benchmark_name):
            image = example["image"]
            question = example["question"]
            answer = example.get("answer", "")

            response = self.generate(image, question)

            # Simple match check
            is_correct = answer.lower().strip() in response.lower()

            if is_correct:
                correct += 1

            predictions.append(
                {
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "correct": is_correct,
                }
            )

        accuracy = correct / len(dataset) if dataset else 0

        print(f"{benchmark_name} Accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")

        return EvalResult(
            benchmark=benchmark_name,
            accuracy=accuracy,
            total_samples=len(dataset),
            correct=correct,
            predictions=predictions,
        )

    def run_all_benchmarks(
        self,
        max_samples_per_benchmark: Optional[int] = None,
    ) -> dict[str, EvalResult]:
        """Run evaluation on all available benchmarks.

        Args:
            max_samples_per_benchmark: Max samples per benchmark

        Returns:
            Dict mapping benchmark names to results
        """
        results = {}

        # VSR
        try:
            results["VSR"] = self.evaluate_vsr(max_samples_per_benchmark)
        except Exception as e:
            print(f"VSR evaluation failed: {e}")

        # CV-Bench
        try:
            results["CV-Bench"] = self.evaluate_cvbench(max_samples_per_benchmark)
        except Exception as e:
            print(f"CV-Bench evaluation failed: {e}")

        return results

    def save_results(
        self,
        results: dict[str, EvalResult],
        output_path: Optional[str] = None,
    ):
        """Save evaluation results to JSON.

        Args:
            results: Evaluation results
            output_path: Output file path
        """
        if output_path is None:
            output_path = Path(self.config.output_dir) / "eval_results.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        json_results = {}
        for name, result in results.items():
            json_results[name] = {
                "benchmark": result.benchmark,
                "accuracy": result.accuracy,
                "total_samples": result.total_samples,
                "correct": result.correct,
                "predictions": result.predictions,
            }

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {output_path}")


class SpatialEvaluator(_BaseSpatialEvaluator):
    """PyTorch-based evaluator for spatial reasoning benchmarks."""

    def __init__(self, config: EvalConfig):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        from ..models.qwen_vl import get_device, load_qwen_vl_for_inference

        self.config = config
        self.device = get_device()

        # Load model
        print(f"Loading model from {config.model_path}...")
        self.model, self.processor = load_qwen_vl_for_inference(
            model_name=config.model_path,
            adapter_path=config.adapter_path,
            use_4bit=config.use_4bit,
        )
        print(f"Model loaded on {self.device}")

    def generate(self, image, question: str) -> str:
        """Generate response for image-question pair.

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Generated response
        """
        import torch

        # Format input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Process input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        return response.strip()


def evaluate_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    benchmarks: list[str] = ["VSR", "CV-Bench"],
    max_samples: Optional[int] = None,
    output_dir: str = "./eval_results",
) -> dict[str, EvalResult]:
    """Convenience function to evaluate a model.

    Args:
        model_path: Path to model or HuggingFace model ID
        adapter_path: Optional path to LoRA adapter
        benchmarks: List of benchmarks to run
        max_samples: Max samples per benchmark
        output_dir: Output directory for results

    Returns:
        Evaluation results
    """
    config = EvalConfig(
        model_path=model_path,
        adapter_path=adapter_path,
        output_dir=output_dir,
    )

    evaluator = SpatialEvaluator(config)
    results = evaluator.run_all_benchmarks(max_samples)
    evaluator.save_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for name, result in results.items():
        print(
            f"{name}: {result.accuracy:.2%} ({result.correct}/{result.total_samples})"
        )
    print("=" * 60)

    return results
