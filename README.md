# Aha Moment

Reproduce the DeepSeek R1 "aha moment" on a single GPU using GRPO (Group Relative Policy Optimization).

A base pretrained model (Qwen2.5-3B) learns chain-of-thought reasoning through reinforcement learning alone — no supervised fine-tuning, no human-written reasoning traces. The model discovers `<think>...</think><answer>...</answer>` formatting and step-by-step math reasoning purely from reward signals.

## Hardware

- Single NVIDIA GPU with >= 16 GB VRAM (tested on RTX 5090 32 GB)
- ~4 hours for 1000 training steps

## Quick Start

```bash
# Install system dependencies (CUDA toolkit, gcc-12)
chmod +x install_deps.sh
./install_deps.sh

# Create venv and install Python packages
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_pretrained.txt

# Train
python train_pretrained.py
```

## What It Does

1. Loads Qwen2.5-3B (base pretrained, no instruction tuning)
2. Applies QLoRA (rank 64) for parameter-efficient training
3. Trains with GRPO on GSM8K math problems using five reward signals:
   - **correctness_reward** (2.0) — extracted answer matches ground truth
   - **int_reward** (0.5) — answer is a valid integer
   - **strict_format_reward** (0.5) — exact `<think>\n...\n</think>\n<answer>\n...\n</answer>` format
   - **soft_format_reward** (0.5) — loose tag matching
   - **xmlcount_reward** (up to 0.5) — partial credit for individual tags
4. Uses vLLM for fast generation, PyTorch for gradient computation

## Training Timeline

| Steps | What Happens |
|-------|-------------|
| 0-50 | Gibberish, random tokens, no format |
| 50-100 | Model discovers `<think>/<answer>` tags, starts attempting math |
| 100-200 | Format solidifies, correctness starts improving |
| 200-400 | **"Aha moment"** — correctness jumps, coherent chain-of-thought |
| 400-1000 | Refinement, accuracy stabilizes |

## NVCC / CUDA Issues on WSL2

Building packages like `vllm`, `bitsandbytes`, or `flash-attn` from source requires a working CUDA toolkit. Here are common issues and fixes:

### "Could not find nvcc"

The default `cuda-toolkit` from Ubuntu repos is often too old. Install directly from NVIDIA:

```bash
# Add NVIDIA repo (Ubuntu 24.04 example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

Then set the environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/nvvm/bin:$PATH
```

### "Unsupported gpu architecture 'compute_120a'"

RTX 5090 (Blackwell, sm_120a) requires CUDA 12.8+. Older toolkit versions don't recognize this architecture. Solution: install `cuda-toolkit-12-8` or newer.

### "Unsupported GNU version! gcc versions later than 12 are not supported"

CUDA 12.x requires gcc <= 12. If your system has gcc 13+:

```bash
sudo apt install -y gcc-12 g++-12
export NVCC_PREPEND_FLAGS="--compiler-bindir=/usr/bin/gcc-12"
```

### "cicc: not found"

The CUDA intermediate compiler lives in a non-standard path. Add it:

```bash
export PATH=/usr/local/cuda-12.8/nvvm/bin:$PATH
```

### Unsloth coef_1 Shape Mismatch Bug

Unsloth's compiled `UnslothGRPOTrainer` has a bug where `coef_1` (from `grpo_accumulated_loss`) includes extra `max_left_pad` columns but `completion_mask` does not. This causes `RuntimeError: The size of tensor a (N) must match the size of tensor b (M)` during metrics logging.

The training script includes an automatic patch that fixes this after `PatchFastRL` runs. The key insight: `PatchFastRL` stores the module as `"UnslothGRPOTrainer"` in `sys.modules` (not `"unsloth_compiled_cache.UnslothGRPOTrainer"`), so the reload must use the correct name.

## TensorBoard

```bash
tensorboard --logdir outputs_pretrained/logs
```

## Files

- `train_pretrained.py` — main training script
- `requirements_pretrained.txt` — Python dependencies
- `install_deps.sh` — system-level CUDA toolkit and gcc setup
