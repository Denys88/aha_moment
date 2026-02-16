"""
train_pretrained.py — Reproduce the DeepSeek R1 "aha moment" on a single GPU.

Uses Unsloth + TRL to train Qwen2.5-3B with GRPO on GSM8K math problems.
The model learns chain-of-thought reasoning through reinforcement learning alone.

Hardware: Single NVIDIA GPU with >= 16GB VRAM (tested on RTX 5090 32GB).

Usage:
    pip install unsloth vllm
    python train_pretrained.py                              # default: configs/5090.yaml
    python train_pretrained.py --config configs/4090.yaml   # RTX 4090
    python train_pretrained.py --resume                     # resume from latest checkpoint
    python train_pretrained.py --resume outputs_pretrained/checkpoint-200
    python train_pretrained.py --play                       # interactive Q&A with trained model
    python train_pretrained.py --play path/to/lora          # use specific LoRA adapter
    python train_pretrained.py --eval                       # evaluate on GSM8K test set
    python train_pretrained.py --eval path/to/lora          # evaluate specific LoRA adapter
"""

import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import argparse
import importlib
import pathlib
import re
import sys

import yaml
from datasets import load_dataset, Dataset
from transformers import TrainerCallback
from unsloth import FastLanguageModel, PatchFastRL

# PatchFastRL must run before importing GRPOTrainer — it replaces the class in trl.
PatchFastRL("GRPO", FastLanguageModel)

SYSTEM_PROMPT = """\
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

SAMPLE_QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "A train travels 60 miles in 1.5 hours. What is its average speed in miles per hour?",
    "If you have 3 apples and buy 5 more, then give away 2, how many apples do you have?",
]


# =============================================================================
# Unsloth bug fix
# =============================================================================

def apply_unsloth_patches():
    """Fix coef_1/completion_mask shape mismatch in UnslothGRPOTrainer.

    PatchFastRL compiles UnslothGRPOTrainer.py and imports it as module
    "UnslothGRPOTrainer" in sys.modules.  The compiled code has a bug:
    coef_1 returned from grpo_accumulated_loss includes max_left_pad extra
    columns, but completion_mask does not.  We patch the file on disk, reload
    the module, and re-bind the class in trl.
    """
    cache = pathlib.Path("unsloth_compiled_cache/UnslothGRPOTrainer.py")
    if not cache.exists():
        return

    src = cache.read_text()
    patched = False
    for old, new in [
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
        if old in src and src.count("Fix: trim coef_1") < 2:
            src = src.replace(old, new, 1)
            patched = True

    if not patched:
        return

    cache.write_text(src)

    mod_name = "UnslothGRPOTrainer"
    if mod_name not in sys.modules:
        return

    cache_dir = str(cache.parent.resolve())
    restore_path = cache_dir not in sys.path
    if restore_path:
        sys.path.insert(0, cache_dir)
    importlib.reload(sys.modules[mod_name])
    if restore_path:
        sys.path.remove(cache_dir)

    import trl, trl.trainer, trl.trainer.grpo_trainer
    cls = sys.modules[mod_name].UnslothGRPOTrainer
    cfg = sys.modules[mod_name].UnslothGRPOConfig
    trl.GRPOTrainer = cls
    trl.trainer.GRPOTrainer = cls
    trl.trainer.grpo_trainer.GRPOTrainer = cls
    trl.GRPOConfig = cfg
    trl.trainer.GRPOConfig = cfg
    trl.trainer.grpo_trainer.GRPOConfig = cfg

    from unsloth.models.rl import _wrap_grpo_generate_and_score
    _wrap_grpo_generate_and_score(cls)
    print("Applied coef_1/completion_mask shape fix")


apply_unsloth_patches()

# Import AFTER patching so we get the fixed classes.
from trl import GRPOConfig, GRPOTrainer


# =============================================================================
# Config
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/5090.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", nargs="?", const=True, default=False,
                        help="Resume from checkpoint. No value = latest, or pass a path.")
    parser.add_argument("--play", nargs="?", const="checkpoints/grpo_pretrained_lora",
                        default=None, metavar="LORA_PATH",
                        help="Interactive mode: ask math questions. "
                             "No value = default LoRA path, or pass a path.")
    parser.add_argument("--eval", nargs="?", const="checkpoints/grpo_pretrained_lora",
                        default=None, metavar="LORA_PATH",
                        help="Evaluate accuracy on GSM8K test set. "
                             "No value = default LoRA path, or pass a path.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "train"],
                        help="Dataset split for --eval (default: test)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Dataset
# =============================================================================

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })


# =============================================================================
# Reward functions
# =============================================================================

def extract_answer(text: str) -> str:
    """Pull content from <answer>...</answer> tags."""
    if "<answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """2.0 if extracted answer matches ground truth, else 0.0."""
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_answer(r) for r in responses]
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
    """0.5 if the extracted answer is a valid integer."""
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
            s -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            s += 0.125
            s -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return s
    return [_score(c[0]["content"]) for c in completions]


REWARD_FUNCS = [
    xmlcount_reward,
    soft_format_reward,
    strict_format_reward,
    int_reward,
    correctness_reward,
]


# =============================================================================
# Model
# =============================================================================

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def load_model(cfg: dict):
    """Load base model with QLoRA adapters."""
    m = cfg["model"]
    print(f"Loading {m['name']} (4-bit={m['load_in_4bit']}, vLLM={m['use_vllm']}) ...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["name"],
        max_seq_length=m["max_seq_len"],
        load_in_4bit=m["load_in_4bit"],
        fast_inference=m["use_vllm"],
        max_lora_rank=m["lora_rank"],
        gpu_memory_utilization=m["gpu_memory_utilization"],
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    model = FastLanguageModel.get_peft_model(
        model,
        r=m["lora_rank"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=m["lora_rank"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def build_training_args(cfg: dict) -> GRPOConfig:
    m = cfg["model"]
    t = cfg["training"]
    return GRPOConfig(
        use_vllm=m["use_vllm"],
        num_generations=t["num_generations"],
        max_prompt_length=t["max_prompt_length"],
        max_completion_length=m["max_seq_len"] - t["max_prompt_length"],

        learning_rate=float(t["learning_rate"]),
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=0.1,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"],

        max_steps=t["max_steps"],
        save_steps=t["save_steps"],
        logging_steps=1,

        output_dir=t["output_dir"],
        report_to="tensorboard",
        logging_dir=f"{t['output_dir']}/logs",
        mask_truncated_completions=False,
    )


def make_samples_callback(cfg: dict, model, tokenizer):
    """Create a callback that prints sample completions every N steps."""
    use_vllm = cfg["model"]["use_vllm"]
    max_tokens = cfg["training"]["max_completion_length"]
    print_every = cfg["sampling"]["print_every"]
    num_samples = cfg["sampling"]["num_samples"]

    class PrintSamplesCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % print_every != 0:
                return
            model_obj = kwargs.get("model")
            if model_obj is None:
                return

            print(f"\n{'='*70}")
            print(f"  SAMPLE COMPLETIONS — step {state.global_step}")
            print(f"{'='*70}")

            if use_vllm:
                self._generate_vllm(model_obj, max_tokens, num_samples)
            else:
                self._generate_hf(model_obj, max_tokens, num_samples)

            print(f"{'='*70}\n")

        def _generate_vllm(self, model_obj, max_tokens, num_samples):
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )
                for q in SAMPLE_QUESTIONS[:num_samples]
            ]
            try:
                lora_req = model.load_lora("grpo_saved_lora") if hasattr(model, "load_lora") else None
            except Exception:
                lora_req = None
            try:
                outputs = model_obj.fast_generate(prompts, sampling_params=sp, lora_request=lora_req)
                for i, (q, out) in enumerate(zip(SAMPLE_QUESTIONS, outputs)):
                    print(f"\n  [{i+1}] Q: {q}")
                    print(f"      A: {out.outputs[0].text[:600]}")
            except Exception as e:
                print(f"  (generation failed: {e})")

        def _generate_hf(self, model_obj, max_tokens, num_samples):
            import torch
            for i, q in enumerate(SAMPLE_QUESTIONS[:num_samples]):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_obj.device)
                with torch.no_grad():
                    out = model_obj.generate(ids, max_new_tokens=max_tokens,
                                             temperature=0.8, top_p=0.95, do_sample=True)
                text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                print(f"\n  [{i+1}] Q: {q}")
                print(f"      A: {text[:600]}")

    return PrintSamplesCallback()


# =============================================================================
# Test inference
# =============================================================================

def test_inference(model, tokenizer, cfg: dict):
    if not cfg["model"]["use_vllm"]:
        return

    from vllm import SamplingParams
    max_tokens = cfg["training"]["max_completion_length"]
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)

    print(f"\n{'='*60}")
    print("Test inference (without LoRA):")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 25 * 4 + 10?"}],
        tokenize=False, add_generation_prompt=True,
    )
    output = model.fast_generate([text], sampling_params=sp, lora_request=None)
    print(output[0].outputs[0].text[:500])

    print(f"\n{'─'*60}")
    print("Test inference (with GRPO LoRA):")
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "What is 25 * 4 + 10?"}],
        tokenize=False, add_generation_prompt=True,
    )
    output = model.fast_generate(
        [text], sampling_params=sp,
        lora_request=model.load_lora("checkpoints/grpo_pretrained_lora"),
    )
    print(output[0].outputs[0].text[:500])


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, tokenizer, cfg: dict, lora_path: str, split: str = "test"):
    """Evaluate accuracy on GSM8K dataset."""
    from vllm import SamplingParams

    max_tokens = cfg["model"]["max_seq_len"] - cfg["training"]["max_prompt_length"]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    try:
        lora_req = model.load_lora(lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    except Exception as e:
        print(f"Warning: could not load LoRA from {lora_path}: {e}")
        print("Evaluating base model (no LoRA)")
        lora_req = None

    dataset = get_gsm8k(split=split)
    questions = [ex["question"] for ex in dataset]
    gold_answers = [ex["answer"] for ex in dataset]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    print(f"\nEvaluating on {len(prompts)} GSM8K {split} examples...")
    outputs = model.fast_generate(prompts, sampling_params=sp, lora_request=lora_req)

    correct = 0
    total = len(outputs)
    for i, (out, gold) in enumerate(zip(outputs, gold_answers)):
        pred = extract_answer(out.outputs[0].text)
        if pred == gold:
            correct += 1
        elif (i + 1) % 200 == 0 or i < 3:
            print(f"  [{i+1}] Gold={gold}  Pred={pred}  Q={questions[i][:80]}...")

    accuracy = correct / total * 100
    print(f"\n{'='*60}")
    print(f"  GSM8K {split} Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*60}\n")
    return accuracy


# =============================================================================
# Interactive play
# =============================================================================

def play_interactive(model, tokenizer, cfg: dict, lora_path: str):
    """Interactive loop: ask math questions, get chain-of-thought answers."""
    from vllm import SamplingParams

    max_tokens = cfg["model"]["max_seq_len"] - cfg["training"]["max_prompt_length"]
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tokens)

    try:
        lora_req = model.load_lora(lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    except Exception as e:
        print(f"Warning: could not load LoRA from {lora_path}: {e}")
        print("Running without LoRA (base model only)")
        lora_req = None

    print(f"\nInteractive mode — type a math question, 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        output = model.fast_generate([text], sampling_params=sp, lora_request=lora_req)
        response = output[0].outputs[0].text
        print(f"\nModel:\n{response}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)

    model, tokenizer = load_model(cfg)

    if args.play is not None:
        play_interactive(model, tokenizer, cfg, args.play)
        return

    if args.eval is not None:
        evaluate(model, tokenizer, cfg, args.eval, args.split)
        return

    dataset = get_gsm8k()
    training_args = build_training_args(cfg)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=dataset,
        callbacks=[make_samples_callback(cfg, model, tokenizer)],
    )

    t = cfg["training"]
    m = cfg["model"]
    print(f"\n{'='*60}")
    print(f"Starting GRPO training: {t['max_steps']} steps")
    print(f"  model      = {m['name']}")
    print(f"  lora_rank  = {m['lora_rank']}")
    print(f"  group_size = {t['num_generations']}")
    print(f"  grad_accum = {t['gradient_accumulation_steps']}")
    print(f"  lr         = {t['learning_rate']}")
    print(f"  max_comp   = {t['max_completion_length']}")
    print(f"  resume     = {args.resume}")
    print(f"{'='*60}\n")

    resume_ckpt = args.resume if isinstance(args.resume, str) else args.resume
    trainer.train(resume_from_checkpoint=resume_ckpt)

    os.makedirs("checkpoints", exist_ok=True)
    model.save_lora("checkpoints/grpo_pretrained_lora")
    print(f"\nLoRA adapter saved to checkpoints/grpo_pretrained_lora")

    test_inference(model, tokenizer, cfg)


if __name__ == "__main__":
    main()
