# Training Qwen3-VL on Google Colab

This guide explains how to train the Qwen3-VL spatial reasoning model on Google Colab.

## Prerequisites

1. **Google Account** with access to Colab
2. **Google Drive** with at least 20GB free space
3. **(Optional)** Colab Pro/Pro+ for longer runtimes and better GPUs
4. **(Optional)** Weights & Biases account for experiment tracking

## Quick Start

1. Open the notebook in Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/qwenvl_sandbox/blob/main/notebooks/qwen3_vl_training.ipynb)

2. Select GPU runtime: `Runtime` > `Change runtime type` > `T4 GPU`

3. Run all cells in order

## GPU Tiers and Capabilities

| Tier | GPU | VRAM | Max Seq Length (2B) | LoRA Rank (2B) | Recommended Use |
|------|-----|------|---------------------|----------------|-----------------|
| Free | T4 | 16GB | 2048 | 16 | SFT with 1k samples |
| Pro | V100 | 16GB | 1536 | 32 | SFT/DPO full training |
| Pro+ | A100 | 40GB | 2048 | 32 | All stages, larger batches |

## Storage and Persistence

All data is stored on Google Drive to persist across sessions:

```
/content/drive/MyDrive/qwen3_vl_training/
├── checkpoints/           # Model checkpoints
│   ├── sft/
│   ├── dpo/
│   └── grpo/
├── datasets_cache/        # Downloaded datasets
└── hf_cache/              # HuggingFace model cache
```

First-time downloads:
- Model: ~4GB (Qwen3-VL-2B, or ~8GB for 4B)
- RLHF-V dataset: ~2GB

Subsequent runs use the cached files.

## Session Timeout Handling

Colab sessions can disconnect (free tier: ~90 mins idle, 12h max runtime).

### Before Training
1. Mount Google Drive for checkpoint persistence
2. Set `save_steps` to a low value (100-200)
3. Enable WandB for progress tracking

### After Timeout
1. Reconnect to runtime
2. Run the "Session Resume Helper" cell (Cell 10)
3. Copy the checkpoint path to `RESUME_FROM_CHECKPOINT`
4. Re-run training cells

### Example Resume
```python
# In configuration cell, set:
RESUME_FROM_CHECKPOINT = "/content/drive/MyDrive/qwen3_vl_training/checkpoints/sft/checkpoint-400"
```

## Configuration Options

### For T4 (Free Tier) with 2B model (default)
```yaml
model:
  name: "Qwen/Qwen3-VL-2B-Instruct"
  lora:
    r: 16
training:
  max_seq_length: 2048
  gradient_accumulation_steps: 8
grpo:
  num_generations: 4
```

### For T4 (Free Tier) with 4B model
Use the `*_4b.yaml` configs, e.g., `configs/colab_sft_config_4b.yaml`.

### For A100 (Pro+)
```yaml
model:
  lora:
    r: 32
training:
  max_seq_length: 2048
  gradient_accumulation_steps: 2
grpo:
  num_generations: 4
```

## WandB Setup

### Method 1: Colab Secrets (Recommended)
1. Go to Colab Settings (gear icon)
2. Click "Secrets" tab
3. Add secret: Name=`WANDB_API_KEY`, Value=your API key

### Method 2: Interactive Login
Run `wandb.login()` and paste your API key when prompted.

### Method 3: Disable WandB
```python
USE_WANDB = False
```

## Training Pipeline

### Stage 1: SFT (Supervised Fine-Tuning)
```python
TRAINING_STAGE = "sft"
MAX_SAMPLES = 1000
NUM_EPOCHS = 1
```

### Stage 2: DPO (Direct Preference Optimization)
```python
TRAINING_STAGE = "dpo"
SFT_CHECKPOINT = "/content/drive/MyDrive/qwen3_vl_training/checkpoints/sft"
MAX_SAMPLES = 500
```

### Stage 3: GRPO (Group Relative Policy Optimization)
```python
TRAINING_STAGE = "grpo"
SFT_CHECKPOINT = "/content/drive/MyDrive/qwen3_vl_training/checkpoints/dpo"
MAX_SAMPLES = 300
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `max_seq_length` to 512
- Reduce LoRA rank to 16
- Increase `gradient_accumulation_steps` to 32
- Reduce `max_samples`

### Session Disconnected
- Use the resume helper cell (Cell 10)
- Ensure checkpoints are on Drive (not `/content`)
- Consider Colab Pro for longer sessions

### Slow Training
- Check GPU is active: `!nvidia-smi`
- Reduce `dataloader_num_workers` if CPU bottleneck
- Use fewer datasets

### Model Download Fails
- Check Google Drive has space
- Try smaller model variant
- Clear HF cache and retry:
  ```python
  !rm -rf /content/drive/MyDrive/qwen3_vl_training/hf_cache
  ```

## Estimated Training Times

| Stage | Samples | GPU | Est. Time |
|-------|---------|-----|-----------|
| SFT | 1000 | T4 | ~4 hours |
| DPO | 500 | T4 | ~2 hours |
| GRPO | 300 | T4 | ~3 hours |

Note: Times are approximate and depend on sequence length and other factors.

## Using Pre-configured Configs

Instead of dynamic configuration, you can use the pre-made Colab configs:

```bash
# SFT
python scripts/train_sft.py --config configs/colab_sft_config.yaml

# DPO
python scripts/train_dpo.py --config configs/colab_dpo_config.yaml \
    --sft_checkpoint /path/to/sft/checkpoint

# GRPO
python scripts/train_grpo.py --config configs/colab_grpo_config.yaml \
    --sft_checkpoint /path/to/dpo/checkpoint
```

## Command Line Reference

```bash
# SFT with resume
python scripts/train_sft.py \
    --config configs/colab_sft_config.yaml \
    --resume_from_checkpoint /path/to/checkpoint

# DPO with resume
python scripts/train_dpo.py \
    --config configs/colab_dpo_config.yaml \
    --sft_checkpoint /path/to/sft \
    --resume_from_checkpoint /path/to/checkpoint

# GRPO with resume
python scripts/train_grpo.py \
    --config configs/colab_grpo_config.yaml \
    --sft_checkpoint /path/to/dpo \
    --resume_from_checkpoint /path/to/checkpoint
```
