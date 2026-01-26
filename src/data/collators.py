"""Data collators for VLM training with Qwen2.5-VL."""

from typing import Any
from dataclasses import dataclass
import torch
from PIL import Image


@dataclass
class VLMDataCollator:
    """Data collator for Qwen2.5-VL training.

    Handles image preprocessing and text tokenization for VLM batches.
    """

    processor: Any
    max_length: int = 2048
    padding: str = "max_length"
    truncation: bool = True

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate batch of examples.

        Args:
            features: List of examples with 'image', 'question', 'answer'

        Returns:
            Batch dict with input_ids, attention_mask, pixel_values, labels
        """
        images = []
        texts = []

        for feature in features:
            image = feature.get("image")
            if image is not None:
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                images.append(image)

            # Format as chat for Qwen2.5-VL
            messages = feature.get("messages", [])
            if not messages:
                question = feature.get("question", "")
                answer = feature.get("answer", "")
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]

            # Build text with image placeholder
            text = self._format_messages(messages)
            texts.append(text)

        # Process with Qwen processor
        batch = self.processor(
            text=texts,
            images=images if images else None,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (same as input_ids for causal LM, -100 for padding)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for Qwen2.5-VL chat template."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                # Add image placeholder for user messages
                formatted += f"<|im_start|>user\n<image>\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        return formatted


@dataclass
class DPODataCollator:
    """Data collator for DPO training with VLM.

    Handles chosen/rejected response pairs with images.
    """

    processor: Any
    max_length: int = 2048
    padding: str = "max_length"

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate batch for DPO training.

        Args:
            features: List with 'image', 'prompt', 'chosen', 'rejected'

        Returns:
            Batch dict for DPO trainer
        """
        batch = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "images": [],
        }

        for feature in features:
            prompt = feature.get("prompt", "")
            chosen = feature.get("chosen", "")
            rejected = feature.get("rejected", "")
            image = feature.get("image")

            # Format prompt with image placeholder
            formatted_prompt = f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            batch["prompt"].append(formatted_prompt)
            batch["chosen"].append(chosen)
            batch["rejected"].append(rejected)

            if image is not None:
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                batch["images"].append(image)

        return batch


@dataclass
class GRPODataCollator:
    """Data collator for GRPO training.

    Prepares prompts for generation and reward computation.
    """

    processor: Any
    max_length: int = 1024

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        """Collate batch for GRPO training.

        Args:
            features: List with 'image', 'question', and optionally 'ground_truth'

        Returns:
            Batch dict for GRPO trainer
        """
        prompts = []
        images = []
        ground_truths = []

        for feature in features:
            question = feature.get("question", feature.get("prompt", ""))
            image = feature.get("image")
            ground_truth = feature.get("ground_truth", feature.get("answer", ""))

            # Format as chat prompt (without assistant response)
            prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(prompt)
            ground_truths.append(ground_truth)

            if image is not None:
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                images.append(image)

        # Process prompts
        inputs = self.processor(
            text=prompts,
            images=images if images else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs["ground_truths"] = ground_truths

        return inputs
