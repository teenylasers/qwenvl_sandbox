"""Evaluation modules for spatial reasoning benchmarks."""

from .spatial_eval import SpatialEvaluator, EvalConfig, EvalResult, evaluate_model
from .mlx_eval import MLXSpatialEvaluator, MLXEvalConfig, evaluate_model_mlx

__all__ = [
    "SpatialEvaluator",
    "EvalConfig",
    "EvalResult",
    "evaluate_model",
    "MLXSpatialEvaluator",
    "MLXEvalConfig",
    "evaluate_model_mlx",
]
