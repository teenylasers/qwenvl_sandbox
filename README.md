# VLM Spatial Reasoning Training Pipeline

Post-training pipeline for Qwen3-VL focused on spatial reasoning capabilities, with SFT, DPO, and GRPO training stages.

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
# Evaluate trained model (PyTorch, requires GPU)
python scripts/evaluate.py --model_path ./outputs/dpo --benchmarks VSR CV-Bench

# Evaluate base model (PyTorch)
python scripts/evaluate.py --model_path Qwen/Qwen3-VL-2B-Instruct --max_samples 100

# Evaluate with MLX (fast, Apple Silicon)
python scripts/evaluate.py --mlx --benchmarks VSR CV-Bench

# Evaluate with a specific MLX model
python scripts/evaluate.py --mlx --mlx-model 8b-4bit --max_samples 100
```

## Project Structure

```
qwenvl_sandbox/
├── configs/
│   ├── sft_config.yaml      # SFT training config (2B default)
│   ├── dpo_config.yaml      # DPO training config (2B default)
│   ├── grpo_config.yaml     # GRPO training config (2B default)
│   ├── *_4b.yaml            # 4B model variants
│   └── colab_*.yaml         # Colab-optimized configs
├── src/
│   ├── data/
│   │   ├── datasets.py      # Dataset loading (RLHF-V, PixMo)
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
│       ├── spatial_eval.py  # Evaluation on benchmarks (PyTorch)
│       └── mlx_eval.py      # Evaluation on benchmarks (MLX)
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
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | 150K | Visual instruction following |
| [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points) | 2.38M | Counting and pointing |
| [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) | 102K | GPT4V-powered captions |

### Evaluation Benchmarks

| Benchmark | Samples | Focus |
|-----------|---------|-------|
| [VSR](https://huggingface.co/datasets/cambridgeltl/vsr_random) | 10k | Visual spatial relations |
| [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | 2,638 | Comprehensive VLM evaluation |

## Testing                                                                                                                            
                                                                                                                                      
```bash                                                                                                                               
# Run all unit tests (fast, no network)                                                                                               
pytest tests/ -v -m "not data_download"

# Run dataset loading tests only
pytest tests/test_datasets.py -v -m "not data_download"

# Run pipeline tests (reward functions, configs, model loading)
pytest tests/test_pipeline.py -v

# Run a specific test
pytest tests/test_datasets.py::TestLlavaInstruct::test_conversation_extraction -v

# Run integration tests (downloads real data from HuggingFace)
pytest tests/ -v -m data_download                                                                                                              
```  

## Configuration

### Model Settings

The default model is Qwen3-VL-2B-Instruct. For 4B, use the `*_4b.yaml` config files.

```yaml
model:
  name: "Qwen/Qwen3-VL-2B-Instruct"  # or "Qwen/Qwen3-VL-4B-Instruct"
  use_lora: true
  use_4bit: true  # QLoRA (CUDA only)
  lora:
    r: 32           # 32 for 2B, 64 for 4B
    alpha: 16
    dropout: 0.05
```

### Training Settings

```yaml
training:
  num_epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 4  # 4 for 2B, 8 for 4B
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
python scripts/test_mlx.py --model 8b-4bit

# Test spatial reasoning
python scripts/test_mlx.py --spatial --image /path/to/image.jpg

# Run all tests
python scripts/test_mlx.py --all
```

**Available MLX Models:**
| Model | HuggingFace Path | RAM Required |
|-------|------------------|--------------|
| 2b-4bit | mlx-community/Qwen3-VL-2B-Instruct-4bit | ~2GB |
| 4b-4bit | mlx-community/Qwen3-VL-4B-Instruct-4bit | ~4GB |
| 8b-4bit | mlx-community/Qwen3-VL-8B-Instruct-4bit | ~8GB |
| 8b-8bit | mlx-community/Qwen3-VL-8B-Instruct-8bit | ~14GB |
| 30b-3bit | mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit | ~12GB |

**Python API:**
```python
from src.models import MLXInference

model = MLXInference("2b-4bit")
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
python scripts/train_sft.py --debug --mlx --mlx_model 8b-4bit --max_samples 3
```

**What MLX debug mode validates:**
- Dataset loading and preprocessing works correctly
- Images and prompts are properly formatted
- Model generates coherent responses
- Reward functions compute valid scores (GRPO)
- Preference pairs have distinct chosen/rejected responses (DPO)

### MLX Benchmark Evaluation

Run the full VSR and CV-Bench evaluation benchmarks locally on Apple Silicon using MLX:

```bash
# Evaluate with default model (2b-4bit, ~2GB RAM)
python scripts/evaluate.py --mlx --benchmarks VSR CV-Bench

# Quick check with limited samples
python scripts/evaluate.py --mlx --max_samples 100

# Use a larger model
python scripts/evaluate.py --mlx --mlx-model 8b-4bit

# Save results to a specific directory
python scripts/evaluate.py --mlx --output_dir ./eval_results/mlx_base
```

### Option 2: PyTorch MPS (Limited)

MPS has compatibility issues with Qwen3-VL and is not recommended:
- Very slow (~7-9 tok/s)
- Backward pass issues prevent training
- Use only if MLX is not available

## Cloud Training

For full training, use cloud GPUs. The pipeline supports:

- **Google Colab** (free tier with T4, Pro with V100/A100)
- **RunPod** / **Lambda Labs** / **Modal**
- Multi-GPU with DeepSpeed or FSDP
- vLLM integration for GRPO sampling

### Google Colab (Recommended for Getting Started)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/qwenvl_sandbox/blob/main/notebooks/qwen3_vl_training.ipynb)

The included Colab notebook handles:
- Automatic GPU detection and config optimization
- Google Drive persistence for checkpoints and datasets
- Session timeout recovery with checkpoint resume
- WandB integration for experiment tracking

See [docs/COLAB_TRAINING.md](docs/COLAB_TRAINING.md) for detailed instructions.

Quick start:
```bash
# Use pre-configured Colab settings
python scripts/train_sft.py --config configs/colab_sft_config.yaml

# Resume from checkpoint after session timeout
python scripts/train_sft.py --config configs/colab_sft_config.yaml \
    --resume_from_checkpoint /path/to/checkpoint-200
```

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
Base Model (Qwen3-VL-2B)
         │
         ▼
    ┌─────────┐
    │   SFT   │  ← PixMo-Cap + RLHF-V (chosen) + optional: LLaVA-Instruct, PixMo-Points, ShareGPT4V
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
