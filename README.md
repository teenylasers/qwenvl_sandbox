# VLM Spatial Reasoning Training Pipeline

Post-training pipeline for Qwen2.5-VL focused on spatial reasoning capabilities, with SFT, DPO, and GRPO training stages.

## Features

- **SFT (Supervised Fine-Tuning)** with TRL's SFTTrainer
- **DPO (Direct Preference Optimization)** for alignment
- **GRPO (Group Relative Policy Optimization)** with custom spatial reasoning rewards
- Support for **QLoRA** (4-bit quantization + LoRA)
- **Apple Silicon (MLX or MPS)** support for local development
- **Cloud GPU** ready (CUDA, multi-GPU with DeepSpeed)

## Installation

```bash
# Clone the repository
cd qwenvl_sandbox

# Install dependencies
pip install -e .

# For cloud training with vLLM and DeepSpeed
pip install -e ".[cloud]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### 1. Download Datasets

```bash
python scripts/download_datasets.py --datasets all
```

### 2. Run SFT Training

```bash
# Full training (requires GPU)
python scripts/train_sft.py --config configs/sft_config.yaml

# Debug with MLX (Faster local testing on Apple Silicon in inference mode only)
python scripts/train_sft.py --debug --mlx --max_samples 5

# Debug with PyTorch (Slower local testing that runs training for a small number of steps)
python scripts/train_sft.py --debug --max_samples 100 --no_wandb
```

### 3. Run DPO Training

```bash
# Using SFT checkpoint
python scripts/train_dpo.py --config configs/dpo_config.yaml --sft_checkpoint ./outputs/sft

# Debug with MLX
python scripts/train_dpo.py --debug --mlx --max_samples 5

# Debug with PyTorch
python scripts/train_dpo.py --debug --max_samples 50 --no_wandb
```

### 4. Run GRPO Training

```bash
# Using SFT/DPO checkpoint
python scripts/train_grpo.py --config configs/grpo_config.yaml --sft_checkpoint ./outputs/dpo

# Debug with MLX (tests multi-generation and reward computation)
python scripts/train_grpo.py --debug --mlx --max_samples 5

# With specific reward type
python scripts/train_grpo.py --reward_type spatial --debug --mlx
```

### 5. Evaluate

```bash
# Evaluate trained model
python scripts/evaluate.py --model_path ./outputs/dpo --benchmarks VSR CV-Bench

# Evaluate base model
python scripts/evaluate.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --max_samples 100
```

## Project Structure

```
qwenvl_sandbox/
├── configs/
│   ├── sft_config.yaml      # SFT training config
│   ├── dpo_config.yaml      # DPO training config
│   └── grpo_config.yaml     # GRPO training config
├── src/
│   ├── data/
│   │   ├── datasets.py      # Dataset loading (RLHF-V, PixMo, SpatialVLM)
│   │   └── collators.py     # Data collators for VLM
│   ├── models/
│   │   ├── qwen_vl.py       # Model loading utilities
│   │   └── mlx_inference.py # MLX inference for Apple Silicon
│   ├── training/
│   │   ├── sft_trainer.py   # SFT trainer
│   │   ├── dpo_trainer.py   # DPO trainer
│   │   ├── grpo_trainer.py  # GRPO trainer
│   │   ├── reward_functions.py  # Spatial reasoning rewards
│   │   └── mlx_debug.py     # MLX-based debug validation
│   └── eval/
│       └── spatial_eval.py  # Evaluation on benchmarks
├── scripts/
│   ├── download_datasets.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   └── evaluate.py
└── tests/
    └── test_pipeline.py
```

## Datasets

### Training Data

| Dataset | Samples | Purpose |
|---------|---------|---------|
| [RLHF-V](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset) | 5,733 | Preference pairs for DPO |
| [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap) | ~1M | Dense captions for SFT |
| [SpatialVLM](https://spatial-vlm.github.io/) | 2B synthetic | Spatial VQA |

### Evaluation Benchmarks

| Benchmark | Samples | Focus |
|-----------|---------|-------|
| [VSR](https://huggingface.co/datasets/cambridgeltl/vsr_random) | 10k | Visual spatial relations |
| [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | 2,638 | Comprehensive VLM evaluation |

## Configuration

### Model Settings

```yaml
model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  use_lora: true
  use_4bit: true  # QLoRA (CUDA only)
  lora:
    r: 64
    alpha: 16
    dropout: 0.05
```

### Training Settings

```yaml
training:
  num_epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  gradient_checkpointing: true
```

## Local Development (Apple Silicon)

### Option 1: MLX (Recommended for Testing)

MLX provides fast local inference on Apple Silicon (~200-400 tok/s):

```bash
# Install MLX dependencies
pip install -e ".[mlx]"

# Quick test
python scripts/test_mlx.py

# Test with specific model
python scripts/test_mlx.py --model 7b-8bit

# Test spatial reasoning
python scripts/test_mlx.py --spatial --image /path/to/image.jpg

# Run all tests
python scripts/test_mlx.py --all
```

**Available MLX Models:**
| Model | HuggingFace Path | RAM Required |
|-------|------------------|--------------|
| 3b-4bit | mlx-community/Qwen2.5-VL-3B-Instruct-4bit | ~4GB |
| 3b-8bit | mlx-community/Qwen2.5-VL-3B-Instruct-8bit | ~6GB |
| 7b-4bit | mlx-community/Qwen2.5-VL-7B-Instruct-4bit | ~8GB |
| 7b-8bit | mlx-community/Qwen2.5-VL-7B-Instruct-8bit | ~14GB |
| 32b-4bit | mlx-community/Qwen2.5-VL-32B-Instruct-4bit | ~20GB |

**Python API:**
```python
from src.models import MLXInference

model = MLXInference("3b-4bit")
response = model.generate(
    "Describe the spatial relationships in this image.",
    image="path/to/image.jpg"
)
print(response)
```

### MLX Debug Mode for Training Scripts

Use `--debug --mlx` to validate training pipelines locally without running actual training:

```bash
# Validate SFT pipeline (data loading, model generation)
python scripts/train_sft.py --debug --mlx --max_samples 5

# Validate DPO pipeline (preference pairs, chosen/rejected format)
python scripts/train_dpo.py --debug --mlx --max_samples 5

# Validate GRPO pipeline (multi-generation, reward computation)
python scripts/train_grpo.py --debug --mlx --max_samples 5

# Use a different MLX model
python scripts/train_sft.py --debug --mlx --mlx_model 7b-8bit --max_samples 3
```

**What MLX debug mode validates:**
- Dataset loading and preprocessing works correctly
- Images and prompts are properly formatted
- Model generates coherent responses
- Reward functions compute valid scores (GRPO)
- Preference pairs have distinct chosen/rejected responses (DPO)

### Option 2: PyTorch MPS (Limited)

MPS has compatibility issues with Qwen2.5-VL and is not recommended:
- Very slow (~7-9 tok/s)
- Backward pass issues prevent training
- Use only if MLX is not available

## Cloud Training

For full training, use cloud GPUs. The pipeline supports:

- **RunPod** / **Lambda Labs** / **Modal**
- Multi-GPU with DeepSpeed or FSDP
- vLLM integration for GRPO sampling

Example cloud config:
```yaml
hardware:
  bf16: true
  use_vllm: true  # For GRPO
  vllm_gpu_memory_utilization: 0.8
```

## Reward Functions

Custom reward functions for spatial reasoning:

### SpatialRewardFunction
- Exact match with ground truth
- Spatial relation extraction and comparison
- Format compliance checking

### CountingRewardFunction
- Exact number matching
- Partial credit for close answers
- Word-to-number conversion

### CombinedRewardFunction
- Combines spatial and counting rewards
- Task-type aware weighting

## Training Pipeline

```
Base Model (Qwen2.5-VL-3B)
         │
         ▼
    ┌─────────┐
    │   SFT   │  ← PixMo-Cap + SpatialVLM + RLHF-V (chosen)
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │   DPO   │  ← RLHF-V preference pairs
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  GRPO   │  ← Spatial reasoning rewards
    └────┬────┘
         │
         ▼
  Spatial-Enhanced VLM
```

## License

MIT
