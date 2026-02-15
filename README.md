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

# Train (default: RTX 5090)
python train_pretrained.py

# Or specify a config for your GPU
python train_pretrained.py --config configs/4090.yaml

# Resume from latest checkpoint
python train_pretrained.py --resume

# Resume from a specific checkpoint
python train_pretrained.py --resume outputs_pretrained/checkpoint-200

# Interactive Q&A with the trained model
python train_pretrained.py --play

# Use a specific LoRA adapter
python train_pretrained.py --play path/to/lora
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

## Example: Interactive Play (after ~400 steps)

```
You: Victor is competing in the CrossFit Open workout 24.1, which has a 15-minute time cap.
He completes burpees over the bar at a rate of 8 per minute for the first 5 minutes, then
slows to 5 per minute for the next 6 minutes, and finishes the last 4 minutes at 3 per
minute. How many total burpees over the bar did Victor complete?

Model:
<think>
Victor completes burpees over the bar at a rate of 8 per minute for the first 5 minutes.
So, the number of burpees he completes in the first 5 minutes is:
8 * 5 = 40 burpees
Then, he slows down to 5 per minute for the next 6 minutes:
5 * 6 = 30 burpees
Finally, he finishes the last 4 minutes at 3 per minute:
3 * 4 = 12 burpees
Therefore, the total number of burpees Victor completes is:
40 + 30 + 12 = 82 burpees
</think>
<answer>82</answer>
```

This question is **not** in the GSM8K training set — the model learned to reason step-by-step purely from reward signals.

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

Unsloth's compiled `UnslothGRPOTrainer` has a bug that crashes training with an error like:

```
RuntimeError: The size of tensor a (793) must match the size of tensor b (768)
  at non-singleton dimension 1
```

or similar with numbers like `(537) vs (512)`. The difference between the two numbers equals the left-padding size (typically ~25 tokens, roughly the system prompt length).

**Root cause:** In `compute_loss`, the function `grpo_accumulated_loss` returns `coef_1` with shape `[batch, logits_to_keep + max_left_pad]`, but the metrics section uses `completion_mask` with shape `[batch, logits_to_keep]`. The `max_left_pad` extra columns come from left-padding alignment for variable-length prompts in the batch.

The crash happens at:
```python
# Inside UnslothGRPOTrainer.compute_loss (unsloth_compiled_cache/UnslothGRPOTrainer.py)
def masked_batch_mean(x):
    return (x * completion_mask).sum() / completion_token_count
    #       ^ coef_1 has 793 cols    ^ completion_mask has 768 cols → RuntimeError
```

**Fix:** The training script includes an automatic patch that inserts `coef_1 = coef_1[:, -completion_mask.shape[1]:]` before both `masked_batch_mean` definitions. The patch runs after `PatchFastRL` writes the compiled file, then reloads the module and re-binds the class in `trl`.

**Key detail for anyone trying to patch this manually:** `PatchFastRL` stores the module as `"UnslothGRPOTrainer"` in `sys.modules` (not `"unsloth_compiled_cache.UnslothGRPOTrainer"`). If you use the wrong name, `importlib.reload` silently does nothing and the fix never takes effect.

## TensorBoard

```bash
tensorboard --logdir outputs_pretrained/logs
```

## Files

- `train_pretrained.py` — main training script
- `configs/5090.yaml` — config for RTX 5090 (32 GB) — lora_rank=64, num_gen=8
- `configs/4090.yaml` — config for RTX 4090 (24 GB) — lora_rank=32, num_gen=6 *(not tested yet)*
- `requirements_pretrained.txt` — Python dependencies
- `install_deps.sh` — system-level CUDA toolkit and gcc setup
