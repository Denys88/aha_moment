"""
train_pretrained.py — Reproduce the DeepSeek R1 "aha moment" on a single GPU.

Uses Unsloth + TRL to train Qwen2.5-3B with GRPO on GSM8K math problems.
The model learns chain-of-thought reasoning through reinforcement learning alone.

Hardware: Single NVIDIA GPU with >= 16GB VRAM (tested on RTX 5090 32GB).

Usage:
    pip install unsloth vllm
    python train_pretrained.py                    # default: configs/5090.yaml
    python train_pretrained.py --config configs/4090.yaml
"""

import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # Extra context length optimization

import argparse
import re
import yaml
import pathlib, importlib, sys
from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset, Dataset
from transformers import TrainerCallback
PatchFastRL("GRPO", FastLanguageModel)

# Fix Unsloth bug: coef_1 shape includes max_left_pad but completion_mask doesn't
# Patch the compiled file after PatchFastRL writes it, then reload module
_cache = pathlib.Path("unsloth_compiled_cache/UnslothGRPOTrainer.py")
if _cache.exists():
    _src = _cache.read_text()
    _patched = False
    for _old, _new in [
        (
            "        completion_token_count = completion_mask.sum().clamp(min = 1.0)",
            "        # Fix: trim coef_1 to match completion_mask (left-pad mismatch)\n"
            "        if coef_1.shape[1] != completion_mask.shape[1]:\n"
            "            coef_1 = coef_1[:, -completion_mask.shape[1]:]\n"
            "        completion_token_count = completion_mask.sum().clamp(min = 1.0)",
        ),
        (
            "        completion_token_count = completion_mask.sum().clamp(min=1.0)",
            "        # Fix: trim coef_1 to match completion_mask (left-pad mismatch)\n"
            "        if coef_1.shape[1] != completion_mask.shape[1]:\n"
            "            coef_1 = coef_1[:, -completion_mask.shape[1]:]\n"
            "        completion_token_count = completion_mask.sum().clamp(min=1.0)",
        ),
    ]:
        if _old in _src and _src.count("Fix: trim coef_1") < 2:
            _src = _src.replace(_old, _new, 1)
            _patched = True
    if _patched:
        _cache.write_text(_src)
        # PatchFastRL stores the module as "UnslothGRPOTrainer" in sys.modules
        _mod_name = "UnslothGRPOTrainer"
        if _mod_name in sys.modules:
            _cache_dir = str(_cache.parent.resolve())
            _restore_path = _cache_dir not in sys.path
            if _restore_path:
                sys.path.insert(0, _cache_dir)
            importlib.reload(sys.modules[_mod_name])
            if _restore_path:
                sys.path.remove(_cache_dir)
            # Re-bind patched class in trl so GRPOTrainer uses the fixed code
            import trl, trl.trainer, trl.trainer.grpo_trainer
            _cls = sys.modules[_mod_name].UnslothGRPOTrainer
            _cfg = sys.modules[_mod_name].UnslothGRPOConfig
            trl.GRPOTrainer = _cls
            trl.trainer.GRPOTrainer = _cls
            trl.trainer.grpo_trainer.GRPOTrainer = _cls
            trl.GRPOConfig = _cfg
            trl.trainer.GRPOConfig = _cfg
            trl.trainer.grpo_trainer.GRPOConfig = _cfg
            # Re-apply generate_and_score wrapper lost during reload
            from unsloth.models.rl import _wrap_grpo_generate_and_score
            _wrap_grpo_generate_and_score(_cls)
            print("Applied coef_1/completion_mask shape fix")

# Import AFTER patching so we get the fixed classes
from trl import GRPOConfig, GRPOTrainer

# =============================================================================
# Configuration — load from YAML
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/5090.yaml",
                    help="Path to YAML config file")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

# Model
MODEL_NAME   = cfg["model"]["name"]
MAX_SEQ_LEN  = cfg["model"]["max_seq_len"]
LORA_RANK    = cfg["model"]["lora_rank"]
LOAD_IN_4BIT = cfg["model"]["load_in_4bit"]
USE_VLLM     = cfg["model"]["use_vllm"]
GPU_MEM_UTIL = cfg["model"]["gpu_memory_utilization"]

# Training
MAX_STEPS    = cfg["training"]["max_steps"]
LR           = cfg["training"]["learning_rate"]
NUM_GEN      = cfg["training"]["num_generations"]
GRAD_ACCUM   = cfg["training"]["gradient_accumulation_steps"]
MAX_COMP_LEN = cfg["training"]["max_completion_length"]
MAX_PROMPT   = cfg["training"]["max_prompt_length"]
SAVE_STEPS   = cfg["training"]["save_steps"]
OUTPUT_DIR   = cfg["training"]["output_dir"]

# =============================================================================
# System prompt & answer format
# =============================================================================
SYSTEM_PROMPT = """\
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""


def extract_answer(text: str) -> str:
    """Pull content from <answer>...</answer> tags."""
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# =============================================================================
# Dataset — GSM8K (grade-school math, ~7.5k train problems)
# =============================================================================
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    return data


# =============================================================================
# Reward functions (all receive batched inputs from TRL)
# =============================================================================
def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """2.0 if extracted answer matches ground truth, else 0.0."""
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_answer(r) for r in responses]
    # Log first example each batch for monitoring
    q = prompts[0][-1]["content"]
    print(
        f"{'─'*50}\n"
        f"Q: {q}\n"
        f"Gold: {answer[0]}\n"
        f"Pred: {extracted[0]}\n"
        f"Raw:  {responses[0][:300]}"
    )
    return [2.0 if e == a else 0.0 for e, a in zip(extracted, answer)]


def int_reward(completions, **kwargs) -> list[float]:
    """0.5 if the extracted answer is a valid integer (GSM8K answers are ints)."""
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_answer(r) for r in responses]
    return [0.5 if e.lstrip("-").replace(",", "").isdigit() else 0.0 for e in extracted]


def strict_format_reward(completions, **kwargs) -> list[float]:
    """0.5 for strict newline-delimited tag format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]


def soft_format_reward(completions, **kwargs) -> list[float]:
    """0.5 for loose tag format (allows whitespace variation)."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]


def xmlcount_reward(completions, **kwargs) -> list[float]:
    """Partial credit for individual XML tags being present."""
    def _score(text: str) -> float:
        s = 0.0
        if text.count("<think>\n") == 1:
            s += 0.125
        if text.count("\n</think>\n") == 1:
            s += 0.125
        if text.count("\n<answer>\n") == 1:
            s += 0.125
            # penalise trailing junk after </answer>
            s -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            s += 0.125
            s -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return s
    return [_score(c[0]["content"]) for c in completions]


# =============================================================================
# Model — Unsloth QLoRA
# =============================================================================
print(f"Loading {MODEL_NAME} (4-bit={LOAD_IN_4BIT}, vLLM={USE_VLLM}) ...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=LOAD_IN_4BIT,
    fast_inference=USE_VLLM,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_MEM_UTIL,
)

# Base model has no chat template — use Qwen2.5's template
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# =============================================================================
# GRPO Training
# =============================================================================
dataset = get_gsm8k()

training_args = GRPOConfig(
    # Generation
    use_vllm=USE_VLLM,
    num_generations=NUM_GEN,
    max_prompt_length=MAX_PROMPT,
    max_completion_length=MAX_SEQ_LEN-MAX_PROMPT,

    # Optimizer
    learning_rate=LR,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    max_grad_norm=0.1,

    # Batching
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Schedule
    max_steps=MAX_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=1,

    # Output
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/logs",
    mask_truncated_completions=False, 
    # ── Algorithm ──
    # Default loss_type="dapo" (recommended, no length bias).
    # For GSPO (Qwen-3 style), uncomment the two lines below:
    # loss_type="grpo",
    # importance_sampling_level="sequence",
)

# =============================================================================
# Callback — print sample completions every N steps
# =============================================================================
PRINT_EVERY = cfg["sampling"]["print_every"]
NUM_SAMPLES = cfg["sampling"]["num_samples"]

SAMPLE_QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "A train travels 60 miles in 1.5 hours. What is its average speed in miles per hour?",
    "If you have 3 apples and buy 5 more, then give away 2, how many apples do you have?",
]


class PrintSamplesCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % PRINT_EVERY != 0:
            return
        model_obj = kwargs.get("model", None)
        if model_obj is None:
            return

        print(f"\n{'='*70}")
        print(f"  SAMPLE COMPLETIONS — step {state.global_step}")
        print(f"{'='*70}")

        if USE_VLLM:
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_COMP_LEN)
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )
                for q in SAMPLE_QUESTIONS[:NUM_SAMPLES]
            ]
            try:
                lora_req = model.load_lora("grpo_saved_lora") if hasattr(model, "load_lora") else None
            except Exception:
                lora_req = None
            try:
                outputs = model_obj.fast_generate(prompts, sampling_params=sp, lora_request=lora_req)
                for i, (q, out) in enumerate(zip(SAMPLE_QUESTIONS, outputs)):
                    text = out.outputs[0].text
                    print(f"\n  [{i+1}] Q: {q}")
                    print(f"      A: {text[:600]}")
            except Exception as e:
                print(f"  (generation failed: {e})")
        else:
            # Non-vLLM fallback: use HF generate
            import torch
            for i, q in enumerate(SAMPLE_QUESTIONS[:NUM_SAMPLES]):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_obj.device)
                with torch.no_grad():
                    out = model_obj.generate(ids, max_new_tokens=MAX_COMP_LEN,
                                             temperature=0.8, top_p=0.95, do_sample=True)
                text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                print(f"\n  [{i+1}] Q: {q}")
                print(f"      A: {text[:600]}")

        print(f"{'='*70}\n")


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward,
        soft_format_reward,
        strict_format_reward,
        int_reward,
        correctness_reward,
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[PrintSamplesCallback()],
)

print(f"\n{'='*60}")
print(f"Starting GRPO training: {MAX_STEPS} steps")
print(f"  model      = {MODEL_NAME}")
print(f"  lora_rank  = {LORA_RANK}")
print(f"  group_size = {NUM_GEN}")
print(f"  grad_accum = {GRAD_ACCUM}")
print(f"  lr         = {LR}")
print(f"  max_comp   = {MAX_COMP_LEN}")
print(f"{'='*60}\n")

trainer.train()

# =============================================================================
# Save LoRA adapter
# =============================================================================
os.makedirs("checkpoints", exist_ok=True)
model.save_lora("checkpoints/grpo_pretrained_lora")
print(f"\nLoRA adapter saved to checkpoints/grpo_pretrained_lora")

# =============================================================================
# Quick test inference
# =============================================================================
print(f"\n{'='*60}")
print("Test inference (without LoRA):")
text = tokenizer.apply_chat_template([
    {"role": "user", "content": "What is 25 * 4 + 10?"},
], tokenize=False, add_generation_prompt=True)

if USE_VLLM:
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_COMP_LEN)
    output = model.fast_generate([text], sampling_params=sampling_params, lora_request=None)
    print(output[0].outputs[0].text[:500])

    print(f"\n{'─'*60}")
    print("Test inference (with GRPO LoRA):")
    output = model.fast_generate(
        [tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is 25 * 4 + 10?"},
        ], tokenize=False, add_generation_prompt=True)],
        sampling_params=sampling_params,
        lora_request=model.load_lora("checkpoints/grpo_pretrained_lora"),
    )
    print(output[0].outputs[0].text[:500])
