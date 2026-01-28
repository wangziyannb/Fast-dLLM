import os
import re
import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from v2.Fast_dLLM_v2_7B.modeling import Fast_dLLM_QwenForCausalLM  # type: ignore

MASK_ID = 151665
STOP_TOKEN = 151645

# GSM8K 标准答案在 "#### <number>"，但模型可能不会严格按这个格式输出，所以做鲁棒抽取
RE_ANSWER = re.compile(r"####\s*([-+]?\d[\d,\.]*)")
RE_NUMBER = re.compile(r"[-+]?\d[\d,\.]*")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    # 你可以在这里调整 prompt，但要保证 fixed vs oracle 用同一套 prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Solve the following problem. Show your reasoning, "
                "and end with the final answer formatted exactly as: #### <number>.\n\n"
                f"{question.strip()}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_final_int(text: str) -> Optional[int]:
    # 先找 ####
    m = RE_ANSWER.findall(text)
    if m:
        s = m[-1]
    else:
        nums = RE_NUMBER.findall(text)
        if not nums:
            return None
        s = nums[-1]

    s = s.replace(",", "").strip()
    # GSM8K 通常是整数；如果出现小数就尽量转成 int
    try:
        if "." in s:
            return int(round(float(s)))
        return int(s)
    except Exception:
        return None


@torch.inference_mode()
def oracle_score_mask_all(
    model: Fast_dLLM_QwenForCausalLM,
    full_ids: torch.LongTensor,   # (1, L) = prompt + gold_answer
    prompt_len: int,
    block_size: int,
    mask_id: int = MASK_ID,
) -> float:
    """
    Oracle-A：把整个 gold answer 区间全 mask，一次 forward，算 answer 区间的 per-token mean CE
    """
    x = full_ids.clone()
    x[:, prompt_len:] = mask_id

    out = model(x, use_cache=False, output_hidden_states=False, block_size=block_size)
    logits = out.logits  # (1, L, V)

    if logits.dim() != 3:
        return float("inf")

    V = logits.size(-1)
    # Fast-dLLM generate 里用同款 token shift：logits_shift[t] 预测 token[t]
    logits_shift = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

    sel_logits = logits_shift[:, prompt_len:, :]   # answer positions
    target = full_ids[:, prompt_len:]

    loss = F.cross_entropy(
        sel_logits.reshape(-1, V),
        target.reshape(-1),
        reduction="mean",
    )
    return float(loss.item())


@torch.inference_mode()
def oracle_score_chunked(
    model: Fast_dLLM_QwenForCausalLM,
    full_ids: torch.LongTensor,   # (1, L)
    prompt_len: int,
    block_size: int,
    mask_id: int = MASK_ID,
) -> float:
    """
    Oracle-B：按 block_size 把 gold answer 分段，每段单独 mask（prefix 使用 gold），token-weighted mean CE
    更贴近“跨 block”生成，但更慢。
    """
    L = full_ids.size(1)
    i = prompt_len
    total_loss = 0.0
    total_cnt = 0

    while i < L:
        chunk_len = min(block_size, L - i)
        x = full_ids[:, : i + chunk_len].clone()
        x[:, i : i + chunk_len] = mask_id

        out = model(x, use_cache=False, output_hidden_states=False, block_size=block_size)
        logits = out.logits

        if logits.dim() != 3:
            return float("inf")

        V = logits.size(-1)
        logits_shift = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        sel_logits = logits_shift[:, i : i + chunk_len, :]
        target = full_ids[:, i : i + chunk_len]

        loss_vec = F.cross_entropy(
            sel_logits.reshape(-1, V),
            target.reshape(-1),
            reduction="none",
        )
        total_loss += float(loss_vec.sum().item())
        total_cnt += int(loss_vec.numel())
        i += chunk_len

    return total_loss / max(total_cnt, 1)


def choose_block_size_oracle(
    model: Fast_dLLM_QwenForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    gold_answer_text: str,
    candidate_block_sizes: List[int],
    device: torch.device,
    oracle_mode: str = "mask_all",  # "mask_all" or "chunked"
) -> Tuple[int, float]:
    # 用 concat 的方式确保 prompt_len 精确
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ans_ids = tokenizer(gold_answer_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = torch.cat([prompt_ids, ans_ids], dim=1)

    prompt_len = prompt_ids.size(1)

    best_B = candidate_block_sizes[0]
    best_score = float("inf")

    for B in candidate_block_sizes:
        if oracle_mode == "chunked":
            score = oracle_score_chunked(model, full_ids, prompt_len, B)
        else:
            score = oracle_score_mask_all(model, full_ids, prompt_len, B)

        if score < best_score:
            best_score = score
            best_B = B

    return best_B, best_score


@torch.inference_mode()
def run_generate(
    model: Fast_dLLM_QwenForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    block_size: int,
    max_new_tokens: int,
    threshold: float,
    device: torch.device,
) -> str:
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # 你要求 mini==block，所以 small_block_size=block_size
    gen_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        small_block_size=block_size,
        threshold=threshold,
        mask_id=MASK_ID,
        stop_token=STOP_TOKEN,
        temperature=0,
        top_p=1.0,
    )

    out_text = tokenizer.decode(gen_ids[0][input_ids.size(1):], skip_special_tokens=True)
    return out_text


def eval_gsm8k_oracle(
    model_path: str = "Efficient-Large-Model/Fast_dLLM_v2_7B",
    device: str = "cuda",
    seed: int = 0,
    max_new_tokens: int = 512,
    threshold: float = 1.0,
    # 你可以先用 [16, 32] 跑通，再扩展到 [8,16,32] 或 [4,8,16,32]
    candidate_block_sizes: List[int] = [1,2],
    fixed_block_sizes: List[int] = [1,2],
    oracle_mode: str = "chunked",  # "mask_all" or "chunked"
    num_samples: Optional[int] = 100,  # None=全量 test
    print_examples: int = 3,
) -> None:
    set_seed(seed)

    device_obj = torch.device(device)
    model = Fast_dLLM_QwenForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device_obj).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    # 统计：fixed baselines
    fixed_correct = {B: 0 for B in fixed_block_sizes}
    fixed_total = 0

    # 统计：oracle
    oracle_correct = 0
    oracle_total = 0
    chosen_counter = Counter()

    for idx, ex in enumerate(tqdm(ds, total=len(ds), desc="Evaluating GSM8K")):
        q = ex["question"]
        gold = ex["answer"]
        gold_num = extract_final_int(gold)

        prompt = build_prompt(tokenizer, q)

        # ===== fixed baselines =====
        for B in fixed_block_sizes:
            out = run_generate(model, tokenizer, prompt, B, max_new_tokens, threshold, device_obj)
            pred = extract_final_int(out)
            if gold_num is not None and pred == gold_num:
                fixed_correct[B] += 1

        fixed_total += 1

        # ===== oracle choose B =====
        best_B, best_score = choose_block_size_oracle(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            gold_answer_text=gold,
            candidate_block_sizes=candidate_block_sizes,
            device=device_obj,
            oracle_mode=oracle_mode,
        )
        chosen_counter[best_B] += 1

        out = run_generate(model, tokenizer, prompt, best_B, max_new_tokens, threshold, device_obj)
        pred = extract_final_int(out)

        if gold_num is not None and pred == gold_num:
            oracle_correct += 1
        oracle_total += 1

        if idx < print_examples:
            print(f"\n===== Example {idx} =====")
            print(f"Chosen B = {best_B}, oracle_score = {best_score:.4f}")
            print("Question:", q)
            print("Gold:", gold_num)
            print("Pred:", pred)
            print("Output:\n", out[:1000])

    print("\n=== Fixed block_size baselines ===")
    for B in fixed_block_sizes:
        acc = fixed_correct[B] / max(fixed_total, 1)
        print(f"fixed B={B:>2}: acc = {acc:.4f} ({fixed_correct[B]}/{fixed_total})")

    print("\n=== Oracle-selected block_size ===")
    oracle_acc = oracle_correct / max(oracle_total, 1)
    print(f"oracle_mode={oracle_mode}, candidates={candidate_block_sizes}")
    print(f"oracle acc = {oracle_acc:.4f} ({oracle_correct}/{oracle_total})")

    print("\nChosen block_size distribution:", dict(sorted(chosen_counter.items())))


if __name__ == "__main__":
    fire.Fire(eval_gsm8k_oracle)
