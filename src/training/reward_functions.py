"""Reward functions for GRPO training on spatial reasoning tasks."""

from typing import Optional, Callable
import re
import torch


class SpatialRewardFunction:
    """Reward function for spatial reasoning tasks.

    Computes rewards based on:
    - Exact match with ground truth
    - Spatial relation accuracy
    - Counting accuracy
    - Format compliance
    """

    # Common spatial relations
    SPATIAL_RELATIONS = {
        "left", "right", "above", "below", "in front of", "behind",
        "inside", "outside", "on", "under", "next to", "near",
        "far from", "between", "among", "around", "through",
    }

    def __init__(
        self,
        exact_match_weight: float = 0.5,
        spatial_weight: float = 0.3,
        format_weight: float = 0.2,
    ):
        """Initialize reward function.

        Args:
            exact_match_weight: Weight for exact match reward
            spatial_weight: Weight for spatial relation accuracy
            format_weight: Weight for format compliance
        """
        self.exact_match_weight = exact_match_weight
        self.spatial_weight = spatial_weight
        self.format_weight = format_weight

    def __call__(
        self,
        completions: list[str],
        ground_truths: list[str],
        **kwargs,
    ) -> list[float]:
        """Compute rewards for completions.

        Args:
            completions: List of model completions
            ground_truths: List of ground truth answers

        Returns:
            List of reward scores
        """
        rewards = []
        for completion, ground_truth in zip(completions, ground_truths):
            reward = self.compute_single_reward(completion, ground_truth)
            rewards.append(reward)
        return rewards

    def compute_single_reward(
        self,
        completion: str,
        ground_truth: str,
    ) -> float:
        """Compute reward for a single completion.

        Args:
            completion: Model completion
            ground_truth: Ground truth answer

        Returns:
            Reward score between -1 and 1
        """
        completion = completion.strip().lower()
        ground_truth = ground_truth.strip().lower()

        # Exact match component
        exact_match = 1.0 if completion == ground_truth else 0.0

        # Partial match / semantic similarity
        partial_match = self._compute_partial_match(completion, ground_truth)

        # Spatial relation accuracy
        spatial_score = self._compute_spatial_score(completion, ground_truth)

        # Format compliance
        format_score = self._compute_format_score(completion)

        # Combine scores
        reward = (
            self.exact_match_weight * exact_match +
            self.spatial_weight * spatial_score +
            self.format_weight * format_score
        )

        # Add partial match bonus if not exact
        if exact_match == 0:
            reward += 0.3 * partial_match

        # Scale to [-1, 1]
        return max(-1.0, min(1.0, reward * 2 - 1))

    def _compute_partial_match(self, completion: str, ground_truth: str) -> float:
        """Compute partial match score based on word overlap."""
        comp_words = set(completion.split())
        gt_words = set(ground_truth.split())

        if not gt_words:
            return 0.0

        overlap = len(comp_words & gt_words)
        precision = overlap / len(comp_words) if comp_words else 0
        recall = overlap / len(gt_words)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _compute_spatial_score(self, completion: str, ground_truth: str) -> float:
        """Check if spatial relations match."""
        comp_relations = self._extract_spatial_relations(completion)
        gt_relations = self._extract_spatial_relations(ground_truth)

        if not gt_relations:
            return 1.0 if not comp_relations else 0.5

        if not comp_relations:
            return 0.0

        # Check overlap
        overlap = len(comp_relations & gt_relations)
        return overlap / len(gt_relations)

    def _extract_spatial_relations(self, text: str) -> set:
        """Extract spatial relations from text."""
        found = set()
        for relation in self.SPATIAL_RELATIONS:
            if relation in text:
                found.add(relation)
        return found

    def _compute_format_score(self, completion: str) -> float:
        """Check format compliance (not too short, not garbage)."""
        # Penalize very short responses
        if len(completion) < 5:
            return 0.3

        # Penalize very long responses
        if len(completion) > 500:
            return 0.7

        # Penalize repetitive text
        words = completion.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return 0.3

        return 1.0


class CountingRewardFunction:
    """Reward function specifically for counting tasks."""

    def __init__(self, exact_weight: float = 0.7, close_weight: float = 0.3):
        self.exact_weight = exact_weight
        self.close_weight = close_weight

    def __call__(
        self,
        completions: list[str],
        ground_truths: list[str],
        **kwargs,
    ) -> list[float]:
        """Compute counting rewards."""
        rewards = []
        for completion, ground_truth in zip(completions, ground_truths):
            reward = self._compute_counting_reward(completion, ground_truth)
            rewards.append(reward)
        return rewards

    def _compute_counting_reward(self, completion: str, ground_truth: str) -> float:
        """Compute reward based on counting accuracy."""
        pred_num = self._extract_number(completion)
        gt_num = self._extract_number(ground_truth)

        if pred_num is None or gt_num is None:
            return -0.5  # Penalize if can't extract number

        if pred_num == gt_num:
            return 1.0  # Exact match

        # Partial credit for close answers
        diff = abs(pred_num - gt_num)
        if diff == 1:
            return 0.5
        elif diff <= 3:
            return 0.2
        else:
            return -0.5

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract first number from text."""
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        # Try word numbers
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12,
        }
        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                return num
        return None


class CombinedRewardFunction:
    """Combines multiple reward functions."""

    def __init__(
        self,
        spatial_reward: Optional[SpatialRewardFunction] = None,
        counting_reward: Optional[CountingRewardFunction] = None,
        spatial_weight: float = 0.6,
        counting_weight: float = 0.4,
    ):
        self.spatial_reward = spatial_reward or SpatialRewardFunction()
        self.counting_reward = counting_reward or CountingRewardFunction()
        self.spatial_weight = spatial_weight
        self.counting_weight = counting_weight

    def __call__(
        self,
        completions: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
        **kwargs,
    ) -> list[float]:
        """Compute combined rewards based on task type."""
        if task_types is None:
            task_types = ["spatial"] * len(completions)

        rewards = []
        for completion, ground_truth, task_type in zip(
            completions, ground_truths, task_types
        ):
            if task_type == "counting":
                reward = self.counting_reward([completion], [ground_truth])[0]
            else:
                reward = self.spatial_reward([completion], [ground_truth])[0]
            rewards.append(reward)

        return rewards


def create_reward_function(reward_type: str = "combined") -> Callable:
    """Factory function to create reward functions.

    Args:
        reward_type: Type of reward function (spatial, counting, combined)

    Returns:
        Reward function callable
    """
    if reward_type == "spatial":
        return SpatialRewardFunction()
    elif reward_type == "counting":
        return CountingRewardFunction()
    else:
        return CombinedRewardFunction()
