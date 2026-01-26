"""MLX-based inference for Qwen2.5-VL on Apple Silicon.

This module provides fast local inference using MLX, which is optimized
for Apple Silicon and achieves ~200-400 tok/s compared to ~7-9 tok/s with MPS.

Use this for:
- Local debugging and testing
- Quick inference experiments
- Validating data pipelines

Note: Training still requires CUDA GPUs. MLX is inference-only for this pipeline.
"""

from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

# Lazy imports to avoid errors when MLX is not installed
_mlx_available = None


def is_mlx_available() -> bool:
    """Check if MLX and mlx-vlm are available."""
    global _mlx_available
    if _mlx_available is None:
        try:
            import mlx.core
            import mlx_vlm
            _mlx_available = True
        except ImportError:
            _mlx_available = False
    return _mlx_available


@dataclass
class MLXModelConfig:
    """Configuration for MLX model loading."""

    model_path: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    verbose: bool = False


# Available MLX models for Qwen2.5-VL
MLX_MODELS = {
    "3b-4bit": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "3b-8bit": "mlx-community/Qwen2.5-VL-3B-Instruct-8bit",
    "7b-4bit": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "7b-8bit": "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    "32b-4bit": "mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
}


class MLXInference:
    """MLX-based inference for Qwen2.5-VL.

    Example:
        >>> from src.models.mlx_inference import MLXInference
        >>> mlx_model = MLXInference("3b-4bit")
        >>> response = mlx_model.generate("Describe this image", image_path="test.jpg")
        >>> print(response)
    """

    def __init__(
        self,
        model_name: str = "3b-4bit",
        config: Optional[MLXModelConfig] = None,
    ):
        """Initialize MLX inference.

        Args:
            model_name: Model shorthand (3b-4bit, 7b-8bit, etc.) or full HF path
            config: Optional configuration override
        """
        if not is_mlx_available():
            raise ImportError(
                "MLX is not available. Install with: pip install mlx-vlm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        # Resolve model path
        if model_name in MLX_MODELS:
            model_path = MLX_MODELS[model_name]
        else:
            model_path = model_name

        self.config = config or MLXModelConfig(model_path=model_path)
        self.model_path = model_path

        print(f"Loading MLX model: {model_path}")
        self.model, self.processor = load(model_path)
        self.model_config = load_config(model_path)
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        image: Optional[Union[str, Path, Image.Image, list]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response for a prompt with optional image.

        Args:
            prompt: Text prompt
            image: Image path, URL, PIL Image, or list of images
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated text response
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Handle image input
        images = None
        num_images = 0
        if image is not None:
            if isinstance(image, list):
                images = image
                num_images = len(image)
            else:
                images = [image]
                num_images = 1

        # Apply chat template
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model_config,
            prompt,
            num_images=num_images,
        )

        # Generate
        output = generate(
            self.model,
            self.processor,
            formatted_prompt,
            images,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            verbose=self.config.verbose,
        )

        # Extract text from GenerationResult if needed
        if hasattr(output, 'text'):
            return output.text
        return output

    def batch_generate(
        self,
        prompts: list[str],
        images: Optional[list] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Generate responses for multiple prompts.

        Args:
            prompts: List of text prompts
            images: List of images (one per prompt, or None)
            max_tokens: Override max tokens

        Returns:
            List of generated responses
        """
        if images is None:
            images = [None] * len(prompts)

        responses = []
        for prompt, image in zip(prompts, images):
            response = self.generate(prompt, image, max_tokens)
            responses.append(response)

        return responses

    def test_spatial_reasoning(
        self,
        image: Union[str, Path, Image.Image],
        questions: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """Test spatial reasoning capabilities on an image.

        Args:
            image: Image to analyze
            questions: Optional custom questions, or use defaults

        Returns:
            Dict mapping questions to responses
        """
        if questions is None:
            questions = [
                "What objects are in this image and where are they located?",
                "Describe the spatial relationships between objects in this image.",
                "What is on the left side of the image? What is on the right?",
                "Count the number of distinct objects in this image.",
                "Is there anything above or below other objects? Describe.",
            ]

        results = {}
        for question in questions:
            response = self.generate(question, image)
            results[question] = response

        return results


def quick_test(model_name: str = "3b-4bit", image_url: Optional[str] = None):
    """Quick test of MLX inference.

    Args:
        model_name: Model to test
        image_url: Optional image URL to test with
    """
    if not is_mlx_available():
        print("MLX is not available. Install with: pip install mlx-vlm")
        return

    print(f"Testing MLX inference with {model_name}...")

    mlx_model = MLXInference(model_name)

    # Test with image
    if image_url is None:
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    print(f"\nTesting with image: {image_url}")
    response = mlx_model.generate(
        "Describe the spatial arrangement of objects in this image.",
        image=image_url,
    )
    print(f"Response: {response}")

    print("\nMLX inference test complete!")


if __name__ == "__main__":
    quick_test()
