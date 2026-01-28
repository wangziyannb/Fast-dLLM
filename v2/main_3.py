import os
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from v2.Fast_dLLM_v2_7B.modeling import Fast_dLLM_QwenForCausalLM  # type: ignore


MASK_ID = 151665
STOP_TOKEN_ID = 151645  # <|im_end|>  (tokenizer_config.json shows 151645 -> "<|im_end|>")  :contentReference[oaicite:2]{index=2}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_prompt(tokenizer, question: str) -> str:
    """
    用 chat template（更贴近 Qwen2.5-Instruct / Fast-dLLM v2 的训练格式）
    让模型尽量用 '#### <number>' 结尾，方便抽取答案。
    """
    user = (
        "Solve the following math problem. Show your reasoning, and end with a line "
        "in the exact format:\n#### <answer>\n\n"
        f"Question:\n{question}\n"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # fallback
    return user + "\nAnswer:\n"


def extract_gold_answer_number(gold_answer: str) -> Optional[str]:
    """
    GSM8K answer 字段通常含 '#### 42'，取其中数字。
    """
    m = re.search(r"####\s*(-?\d+)", gold_answer)
    return m.group(1) if m else None


def extract_pred_answer_number(text: str) -> Optional[str]:
    """
    优先找 '#### <num>'；否则回退到最后一个整数。
    """
    m = re.findall(r"####\s*(-?\d+)", text)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


@torch.no_grad()
def generate_answer(
    model: Fast_dLLM_QwenForCausalLM,
    tokenizer,
    prompt_text: str,
    block_size: int,
    max_new_tokens: int,
    threshold: float,
    temperature: float,
    top_p: float,
    use_block_cache: bool,
) -> str:
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        small_block_size=block_size,  # 你当前设定：mini block = block
        threshold=threshold,
        temperature=temperature,
        top_p=top_p,
        stop_token=STOP_TOKEN_ID,
        use_block_cache=use_block_cache,
        tokenizer=tokenizer,  # 允许传入但 generate() 实际会忽略；保持和 model card 一致 :contentReference[oaicite:3]{index=3}
    )
    out = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    return out


@torch.no_grad()
def oracle_mean_nll_for_block_size(
    model: Fast_dLLM_QwenForCausalLM,
    prompt_ids: torch.LongTensor,   # (1, P)
    answer_ids: torch.LongTensor,   # (1, A) 仅包含 answer 内容 + <|im_end|>（可选）
    block_size: int,
    mask_id: int = MASK_ID,
) -> float:
    """
    计算给定 block_size=B 时，gold answer tokens 的 mean NLL（masked-block 语义 + cache 递推）。
    尽量贴合 modeling.py 的 generate()：
      - block 内 masked diffusion 用 token-shift logits 对齐
      - 若 prompt_len > B 且 prompt_len % B == 0：先算一个 next-token CE（对应 generate 的“先 argmax 出一个 token”）
    """
    device = model.device
    prompt_ids = prompt_ids.to(device)
    answer_ids = answer_ids.to(device)

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)  # (1, L)
    P = prompt_ids.shape[1]
    L = full_ids.shape[1]

    total_nll = 0.0
    total_cnt = 0

    past = None

    # 1) prefill: 缓存 prompt 的整 block 部分
    prefix_full = (P // block_size) * block_size
    out_pref = None
    if prefix_full > 0:
        out_pref = model(
            full_ids[:, :prefix_full],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )
        past = out_pref.past_key_values

    # 2) generate() 的边界特判：prompt_len > B 且 prompt_len % B == 0
    #    它会先用 raw logits[-1] 预测一个 next token。
    #    oracle 里我们把“第一个 answer token”当成 gold next token，并把它的 CE 计入分数。
    start_pos = P  # 第一个 answer token 的全局位置
    if (P > block_size) and (P % block_size == 0) and (P < L):
        # out_pref 应该刚好是 full_ids[:P] 的输出；如果不是就补一次
        if out_pref is None or prefix_full != P:
            tmp = model(full_ids[:, :P], use_cache=True, past_key_values=past, update_past_key_values=False, block_size=block_size)
            logits_next = tmp.logits[:, -1, :]  # (1, V)
        else:
            logits_next = out_pref.logits[:, -1, :]  # (1, V)

        gold_first = full_ids[:, P]  # (1,)
        total_nll += F.cross_entropy(logits_next, gold_first, reduction="sum").item()
        total_cnt += 1
        start_pos = P + 1  # 后续 token 走 block masked

    # 3) 从包含 start_pos 的 block 开始，逐 block 计算 masked-only CE，再用 gold block 更新 cache
    block_start = (start_pos // block_size) * block_size

    while block_start < L:
        seg_len = min(block_size, L - block_start)
        gold_seg = full_ids[:, block_start : block_start + seg_len]  # (1, seg_len)

        # pad 到 block_size（尾块不足时）
        if seg_len < block_size:
            pad = torch.full((1, block_size - seg_len), mask_id, device=device, dtype=torch.long)
            gold_block = torch.cat([gold_seg, pad], dim=1)  # (1, B)
        else:
            gold_block = gold_seg  # (1, B)

        # 本 block 内需要评分的范围：只算 answer token（从 start_pos 起），且只算真实 token（<= L）
        global_s = max(start_pos, block_start)
        global_e = min(L, block_start + seg_len)
        if global_s < global_e:
            local_s = global_s - block_start
            local_e = global_e - block_start

            x = gold_block.clone()
            x[:, local_s:local_e] = mask_id

            out = model(
                x,
                use_cache=True,
                past_key_values=past,
                update_past_key_values=False,
                block_size=block_size,
            )
            logits = out.logits  # (1, B, V)

            # token shift：logits_shift[t] 预测 token[t]
            logits_shift = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
            sel_logits = logits_shift[:, local_s:local_e, :]
            sel_target = gold_block[:, local_s:local_e]

            total_nll += F.cross_entropy(
                sel_logits.reshape(-1, sel_logits.size(-1)),
                sel_target.reshape(-1),
                reduction="sum",
            ).item()
            total_cnt += (local_e - local_s)

        # 用 gold_seg 更新 cache（对应 generate 里“block 解完后 update_past_key_values=True”）
        out_up = model(
            gold_seg,
            use_cache=True,
            past_key_values=past,
            update_past_key_values=True,
            block_size=block_size,
        )
        past = out_up.past_key_values

        block_start += block_size

    return total_nll / max(total_cnt, 1)


@torch.no_grad()
def choose_oracle_block_size(
    model: Fast_dLLM_QwenForCausalLM,
    tokenizer,
    prompt_text: str,
    gold_answer_text: str,
    block_sizes: List[int],
    append_im_end: bool = True,
) -> Tuple[int, Dict[int, float]]:
    """
    返回 (best_B, {B: score(B)})，score 越小越好。
    """
    prompt_enc = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    prompt_ids = prompt_enc["input_ids"]  # (1, P)

    # gold answer tokens：用 add_special_tokens=False，避免加 BOS；这是“可生成”的 continuation token 序列
    suffix = gold_answer_text.strip()
    if append_im_end:
        suffix = suffix + "\n<|im_end|>"  # 151645 -> <|im_end|>  :contentReference[oaicite:4]{index=4}
    ans_ids = tokenizer([suffix], return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"]

    scores: Dict[int, float] = {}
    for B in block_sizes:
        scores[B] = oracle_mean_nll_for_block_size(
            model=model,
            prompt_ids=prompt_ids,
            answer_ids=ans_ids,
            block_size=B,
        )
    best_B = min(scores, key=scores.get)
    return best_B, scores


def main(
    model_path: str = "Efficient-Large-Model/Fast_dLLM_v2_7B",
    device: str = "cuda",
    split: str = "test",              # 建议先用 test 看 accuracy
    num_samples: int = 200,           # 先小跑；全量 test=1319
    block_sizes: str = "4,8,16,32",   # 你也可以 "1,2,4,8,16,32"
    max_new_tokens: int = 256,
    threshold: float = 0.9,
    temperature: float = 0.0,
    top_p: float = 0.95,
    use_block_cache: bool = False,
    seed: int = 0,
) -> None:
    set_seed(seed)

    bs_list = [int(x) for x in block_sizes.split(",") if x.strip()]
    bs_list = sorted(list(dict.fromkeys(bs_list)))

    device_obj = torch.device(device)
    model = Fast_dLLM_QwenForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device_obj)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.select(range(min(len(ds), num_samples)))

    # 统计
    fixed_correct = {B: 0 for B in bs_list}
    oracle_correct = 0

    oracle_choice_hist = {B: 0 for B in bs_list}

    for ex in tqdm(ds, desc=f"gsm8k-{split}"):
        q = ex["question"]
        gold = ex["answer"]
        gold_num = extract_gold_answer_number(gold)
        if gold_num is None:
            continue

        prompt_text = build_prompt(tokenizer, q)

        # ===== fixed baselines =====
        for B in bs_list:
            out = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                block_size=B,
                max_new_tokens=max_new_tokens,
                threshold=threshold,
                temperature=temperature,
                top_p=top_p,
                use_block_cache=use_block_cache,
            )
            pred_num = extract_pred_answer_number(out)
            if pred_num == gold_num:
                fixed_correct[B] += 1

        # ===== oracle pick B =====
        best_B, scores = choose_oracle_block_size(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            gold_answer_text=gold,   # 你也可以改成只用 "#### <num>" 试试
            block_sizes=bs_list,
            append_im_end=True,
        )
        oracle_choice_hist[best_B] += 1

        out = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            block_size=best_B,
            max_new_tokens=max_new_tokens,
            threshold=threshold,
            temperature=temperature,
            top_p=top_p,
            use_block_cache=use_block_cache,
        )
        pred_num = extract_pred_answer_number(out)
        if pred_num == gold_num:
            oracle_correct += 1

    n = len(ds)
    print("\n=== Fixed block_size accuracy ===")
    for B in bs_list:
        acc = fixed_correct[B] / n
        print(f"fixed_B={B:>2d}  acc={acc:.4f}  ({fixed_correct[B]}/{n})")

    print("\n=== Oracle-selected block_size accuracy ===")
    print(f"oracle_acc={oracle_correct/n:.4f}  ({oracle_correct}/{n})")

    print("\n=== Oracle choice histogram ===")
    for B in bs_list:
        print(f"B={B:>2d}: {oracle_choice_hist[B]}")


if __name__ == "__main__":
    fire.Fire(main)
