"""Validation utilities for training pipeline debugging.

Provides infrastructure for comprehensive training validation checks
that can be run locally on Apple Silicon before cloud training.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationStatus(Enum):
    """Status of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    status: ValidationStatus
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        icons = {
            "passed": "[OK]",
            "failed": "[FAIL]",
            "warning": "[WARN]",
            "skipped": "[SKIP]",
        }
        return f"{icons[self.status.value]} {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregated validation report for a training pipeline."""

    trainer_type: str
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)

    @property
    def passed(self) -> bool:
        """Check if all critical validations passed."""
        return all(
            r.status in (ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.SKIPPED)
            for r in self.results
        )

    @property
    def summary(self) -> dict[str, int]:
        """Get summary counts of validation results."""
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == ValidationStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == ValidationStatus.FAILED),
            "warnings": sum(1 for r in self.results if r.status == ValidationStatus.WARNING),
            "skipped": sum(1 for r in self.results if r.status == ValidationStatus.SKIPPED),
        }

    def print_report(self) -> None:
        """Print formatted validation report."""
        print("\n" + "=" * 70)
        print(f"VALIDATION REPORT: {self.trainer_type}")
        print("=" * 70)

        for result in self.results:
            print(f"  {result}")
            if result.details and result.status == ValidationStatus.FAILED:
                for key, value in result.details.items():
                    print(f"      {key}: {value}")

        print("-" * 70)
        s = self.summary
        print(f"SUMMARY: {s['passed']}/{s['total']} passed, {s['failed']} failed, {s['warnings']} warnings")
        status = "PASSED" if self.passed else "FAILED"
        print(f"OVERALL: {status}")
        print("=" * 70)


def validate_batch_shapes(
    batch: dict[str, Any],
    expected_keys: list[str],
    batch_size: int,
    max_seq_length: int,
    has_images: bool = True,
) -> list[ValidationResult]:
    """Validate training batch tensor shapes.

    Args:
        batch: The batch dictionary from a data collator
        expected_keys: Keys that should be present in the batch
        batch_size: Expected batch size
        max_seq_length: Maximum sequence length for validation
        has_images: Whether to validate pixel_values

    Returns:
        List of ValidationResult objects
    """
    results = []

    # Check all expected keys present
    missing_keys = set(expected_keys) - set(batch.keys())
    if missing_keys:
        results.append(
            ValidationResult(
                name="batch_keys",
                status=ValidationStatus.FAILED,
                message=f"Missing batch keys: {missing_keys}",
                details={"expected": expected_keys, "actual": list(batch.keys())},
            )
        )
    else:
        results.append(
            ValidationResult(
                name="batch_keys",
                status=ValidationStatus.PASSED,
                message=f"All {len(expected_keys)} expected keys present",
            )
        )

    # Validate input_ids shape
    if "input_ids" in batch:
        input_ids = batch["input_ids"]
        shape = tuple(input_ids.shape)

        if len(shape) != 2:
            results.append(
                ValidationResult(
                    name="input_ids_shape",
                    status=ValidationStatus.FAILED,
                    message=f"input_ids should be 2D, got {len(shape)}D",
                )
            )
        elif shape[0] != batch_size:
            results.append(
                ValidationResult(
                    name="input_ids_batch_size",
                    status=ValidationStatus.FAILED,
                    message=f"Batch size mismatch: expected {batch_size}, got {shape[0]}",
                )
            )
        elif shape[1] > max_seq_length:
            results.append(
                ValidationResult(
                    name="input_ids_seq_length",
                    status=ValidationStatus.WARNING,
                    message=f"Sequence length {shape[1]} exceeds max {max_seq_length}",
                )
            )
        else:
            results.append(
                ValidationResult(
                    name="input_ids_shape",
                    status=ValidationStatus.PASSED,
                    message=f"Shape: {shape}",
                )
            )

    # Validate labels alignment
    if "labels" in batch and "input_ids" in batch:
        labels_shape = tuple(batch["labels"].shape)
        input_shape = tuple(batch["input_ids"].shape)

        if labels_shape != input_shape:
            results.append(
                ValidationResult(
                    name="labels_alignment",
                    status=ValidationStatus.FAILED,
                    message="Labels shape must match input_ids",
                    details={"input_ids": input_shape, "labels": labels_shape},
                )
            )
        else:
            # Check labels have -100 for padding
            padding_count = (batch["labels"] == -100).sum().item()
            results.append(
                ValidationResult(
                    name="labels_alignment",
                    status=ValidationStatus.PASSED,
                    message=f"Labels aligned, {padding_count} tokens masked with -100",
                )
            )

    # Validate pixel_values for VLM
    if has_images and "pixel_values" in batch:
        pv = batch["pixel_values"]
        pv_shape = tuple(pv.shape)

        if len(pv_shape) < 4:
            results.append(
                ValidationResult(
                    name="pixel_values_shape",
                    status=ValidationStatus.FAILED,
                    message=f"pixel_values should be at least 4D, got {len(pv_shape)}D",
                )
            )
        else:
            results.append(
                ValidationResult(
                    name="pixel_values_shape",
                    status=ValidationStatus.PASSED,
                    message=f"Shape: {pv_shape}",
                )
            )

    return results


def validate_lora_config(
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> ValidationResult:
    """Validate LoRA configuration parameters.

    Args:
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate

    Returns:
        ValidationResult for LoRA config
    """
    expected_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    issues = []
    if lora_r < 8:
        issues.append(f"r={lora_r} is very small")
    if lora_r > 256:
        issues.append(f"r={lora_r} is very large")
    if lora_alpha < lora_r / 4:
        issues.append(f"alpha={lora_alpha} may be too small relative to r={lora_r}")
    if lora_dropout > 0.5:
        issues.append(f"dropout={lora_dropout} is high")

    if issues:
        return ValidationResult(
            name="lora_config",
            status=ValidationStatus.WARNING,
            message=f"LoRA config issues: {', '.join(issues)}",
            details={"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout},
        )

    return ValidationResult(
        name="lora_config",
        status=ValidationStatus.PASSED,
        message=f"LoRA r={lora_r}, alpha={lora_alpha}, targets {len(expected_modules)} modules",
    )


def validate_gradient_accumulation(
    batch_size: int,
    gradient_accumulation_steps: int,
    min_effective_batch: int = 8,
) -> ValidationResult:
    """Validate gradient accumulation settings.

    Args:
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of accumulation steps
        min_effective_batch: Minimum recommended effective batch size

    Returns:
        ValidationResult for gradient accumulation
    """
    effective_batch = batch_size * gradient_accumulation_steps

    if effective_batch < min_effective_batch:
        return ValidationResult(
            name="gradient_accumulation",
            status=ValidationStatus.WARNING,
            message=f"Effective batch size {effective_batch} is small (recommend >= {min_effective_batch})",
        )

    return ValidationResult(
        name="gradient_accumulation",
        status=ValidationStatus.PASSED,
        message=f"Effective batch size: {batch_size} x {gradient_accumulation_steps} = {effective_batch}",
    )
