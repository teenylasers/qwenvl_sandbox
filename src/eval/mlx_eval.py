"""MLX-based evaluation for spatial reasoning benchmarks on Apple Silicon."""

from typing import Optional
from dataclasses import dataclass

from .spatial_eval import _BaseSpatialEvaluator, EvalResult


@dataclass
class MLXEvalConfig:
    """Configuration for MLX evaluation."""

    mlx_model: str = "4b-4bit"
    max_new_tokens: int = 256
    temperature: float = 0.1
    output_dir: str = "./eval_results"


class MLXSpatialEvaluator(_BaseSpatialEvaluator):
    """MLX-based evaluator for spatial reasoning benchmarks.

    Uses MLX for fast inference on Apple Silicon (~200-400 tok/s).
    """

    def __init__(self, config: MLXEvalConfig):
        """Initialize MLX evaluator.

        Args:
            config: MLX evaluation configuration
        """
        from ..models.mlx_inference import MLXInference, MLXModelConfig

        self.config = config

        mlx_config = MLXModelConfig(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

        print(f"Loading MLX model: {config.mlx_model}")
        self.mlx_model = MLXInference(config.mlx_model, config=mlx_config)

    def generate(self, image, question: str) -> str:
        """Generate response for image-question pair using MLX.

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Generated response
        """
        response = self.mlx_model.generate(
            question,
            image=image,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return response.strip()


def evaluate_model_mlx(
    mlx_model: str = "4b-4bit",
    benchmarks: list[str] = ["VSR", "CV-Bench"],
    max_samples: Optional[int] = None,
    output_dir: str = "./eval_results",
) -> dict[str, EvalResult]:
    """Convenience function to evaluate using MLX.

    Args:
        mlx_model: MLX model shorthand or HuggingFace path
        benchmarks: List of benchmarks to run
        max_samples: Max samples per benchmark
        output_dir: Output directory for results

    Returns:
        Evaluation results
    """
    config = MLXEvalConfig(
        mlx_model=mlx_model,
        output_dir=output_dir,
    )

    evaluator = MLXSpatialEvaluator(config)
    results = evaluator.run_all_benchmarks(max_samples)
    evaluator.save_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary (MLX)")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name}: {result.accuracy:.2%} ({result.correct}/{result.total_samples})")
    print("=" * 60)

    return results
