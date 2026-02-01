"""Tests for the VLM training pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRewardFunctions:
    """Test reward functions for GRPO training."""

    def test_spatial_reward_exact_match(self):
        """Test spatial reward with exact match."""
        from src.training.reward_functions import SpatialRewardFunction

        reward_fn = SpatialRewardFunction()
        rewards = reward_fn(
            completions=["The cat is on the left of the dog"],
            ground_truths=["The cat is on the left of the dog"],
        )
        assert len(rewards) == 1
        assert rewards[0] > 0.5  # Should be high for exact match

    def test_spatial_reward_partial_match(self):
        """Test spatial reward with partial match."""
        from src.training.reward_functions import SpatialRewardFunction

        reward_fn = SpatialRewardFunction()
        rewards = reward_fn(
            completions=["The cat is left of the dog"],
            ground_truths=["The cat is on the left side of the dog"],
        )
        assert len(rewards) == 1
        # Should have some reward for partial match

    def test_spatial_reward_wrong_relation(self):
        """Test spatial reward with wrong relation."""
        from src.training.reward_functions import SpatialRewardFunction

        reward_fn = SpatialRewardFunction()
        rewards = reward_fn(
            completions=["The cat is on the right of the dog"],
            ground_truths=["The cat is on the left of the dog"],
        )
        assert len(rewards) == 1
        # Should be lower than exact match

    def test_counting_reward_exact(self):
        """Test counting reward with exact match."""
        from src.training.reward_functions import CountingRewardFunction

        reward_fn = CountingRewardFunction()
        rewards = reward_fn(
            completions=["There are 5 apples in the image."],
            ground_truths=["5"],
        )
        assert len(rewards) == 1
        assert rewards[0] == 1.0  # Exact match

    def test_counting_reward_close(self):
        """Test counting reward with close answer."""
        from src.training.reward_functions import CountingRewardFunction

        reward_fn = CountingRewardFunction()
        rewards = reward_fn(
            completions=["There are 4 apples."],
            ground_truths=["5"],
        )
        assert len(rewards) == 1
        assert rewards[0] == 0.5  # Off by 1

    def test_counting_reward_word_numbers(self):
        """Test counting reward with word numbers."""
        from src.training.reward_functions import CountingRewardFunction

        reward_fn = CountingRewardFunction()
        rewards = reward_fn(
            completions=["There are three cats."],
            ground_truths=["3"],
        )
        assert len(rewards) == 1
        assert rewards[0] == 1.0


class TestModelLoading:
    """Test model loading utilities."""

    def test_get_device(self):
        """Test device detection."""
        from src.models.qwen_vl import get_device

        device = get_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_get_peft_config(self):
        """Test LoRA config creation."""
        from src.models.qwen_vl import get_peft_config

        config = get_peft_config(r=32, lora_alpha=16)
        assert config.r == 32
        assert config.lora_alpha == 16
        assert "q_proj" in config.target_modules


class TestConfigs:
    """Test training configuration classes."""

    def test_sft_config_defaults(self):
        """Test SFT config has sensible defaults."""
        from src.training.sft_trainer import SFTTrainingConfig

        config = SFTTrainingConfig()
        assert config.model_name == "Qwen/Qwen3-VL-4B-Instruct"
        assert config.use_lora is True
        assert config.learning_rate == 2e-5

    def test_dpo_config_defaults(self):
        """Test DPO config has sensible defaults."""
        from src.training.dpo_trainer import DPOTrainingConfig

        config = DPOTrainingConfig()
        assert config.beta == 0.1
        assert config.loss_type == "sigmoid"

    def test_grpo_config_defaults(self):
        """Test GRPO config has sensible defaults."""
        from src.training.grpo_trainer import GRPOTrainingConfig

        config = GRPOTrainingConfig()
        assert config.num_generations == 4
        assert config.reward_type == "combined"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
