import random
import os
import types
from typing import Dict, List

import fire
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise RuntimeError('PyTorch must be installed to run this script.') from e

try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise RuntimeError('The transformers library must be installed.') from e

try:
    from datasets import load_dataset
except ImportError as e:
    raise RuntimeError('The datasets library must be installed.') from e

from v2 import generation_functions  # type: ignore
from v2.Fast_dLLM_v2_7B.modeling import Fast_dLLM_QwenForCausalLM  # type: ignore


def set_random_seed(seed: int) -> None:
    """Helper to set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_gsm8k_dataset(num_samples: int) -> List[str]:
    """
    Load the GSM8K training split and return a list of concatenated
    `question` + `answer` strings. Only the first `num_samples` examples are
    returned to keep evaluation lightweight.

    Parameters
    ----------
    num_samples : int
        The number of training examples to load.

    Returns
    -------
    List[str]
        A list of strings where each element is `question\nanswer`.
    """
    dataset = load_dataset('openai/gsm8k', 'main', split='train')
    dataset = dataset.select(range(min(len(dataset), num_samples)))
    combined = []
    for example in dataset:
        # Combine question and answer with a newline. Adjust formatting as needed.
        combined.append(example['question'].strip() + '\n' + example['answer'].strip())
    return combined


def compute_block_losses_per_token(
        model: Fast_dLLM_QwenForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        block_sizes: List[int],
        use_mdm_sample: bool,  # 这里我们按“masked-block diffusion loss”算；mdm_sample 本身不返回loss
        device: torch.device,
        mask_id: int = 151665,
) -> Dict[int, Dict[str, List[float]]]:
    """
    对每个 token 起点 i，构造“next block 全 MASK”的输入，计算该 block 的 masked-only loss。
    若失败则 fallback 到 AR loss（teacher forcing）在区间 [i, i+B) 的 loss。
    返回：losses[bs]["sum"/"avg"/"min"/"max"] 是长度为 k 的 list（不足B就NaN）。
    """
    # ========== tokenize (batch_size=1) ==========
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    full_ids = encoded["input_ids"].to(device)  # (1, k)
    k = full_ids.shape[1]

    # ========== 先把 AR per-token CE 准备好，作为 fallback ==========
    # AR ce_loss[t] 对应 token position t+1 (标准 next-token LM)
    with torch.no_grad():
        ar_out = model(full_ids[:, :-1], use_cache=False, output_hidden_states=False)
    ar_logits = ar_out.logits  # (1, k-1, V)
    ar_targets = full_ids[:, 1:]  # (1, k-1)
    V = ar_logits.size(-1)
    ar_ce = F.cross_entropy(ar_logits.reshape(-1, V), ar_targets.reshape(-1), reduction="none").view(1, -1)[0]

    # ar_ce 长度 = k-1，ar_ce[pos] 是预测 full_ids[pos+1] 的loss

    def _ar_block_stats(i: int, B: int):
        """
        AR fallback：用 ar_ce[i : i+B] 的统计。
        注意：ar_ce[i] 对应 token i+1 的loss，所以这其实是“从 i+1 开始的B个token”。
        作为 fallback 足够，但不是严格 masked-block 语义。
        """
        start = i
        end = i + B
        if end <= ar_ce.numel():
            seg = ar_ce[start:end]
            return seg.sum().item(), seg.mean().item(), seg.min().item(), seg.max().item()
        return float("nan"), float("nan"), float("nan"), float("nan")

    # ========== 输出容器 ==========
    losses: Dict[int, Dict[str, List[float]]] = {
        bs: {"sum": [], "avg": [], "min": [], "max": []} for bs in block_sizes
    }

    # ========== masked-block diffusion-style loss ==========
    # mdm_sample 不返回loss，所以 use_mdm_sample 这里只当作“尝试用 masked-block 语义”
    use_masked_block = bool(use_mdm_sample)

    for B in block_sizes:
        for i in range(k):
            # 不足一个完整block就NaN（或你也可以选择只到 k-B）
            if i + B > k:
                for stat in ["sum", "avg", "min", "max"]:
                    losses[B][stat].append(float("nan"))
                continue

            if not use_masked_block:
                s, a, mi, ma = _ar_block_stats(i, B)
                losses[B]["sum"].append(s)
                losses[B]["avg"].append(a)
                losses[B]["min"].append(mi)
                losses[B]["max"].append(ma)
                continue

            # 构造输入：prefix(0..i-1) + [MASK]*B
            x = full_ids[:, : i + B].clone()  # (1, i+B)
            x[:, i: i + B] = mask_id

            # forward：尽量按 Fast-dLLM 的接口传 block_size/bd_size
            try:
                # 某些实现会读 model.bd_size
                if hasattr(model, "bd_size"):
                    model.bd_size = B
                out = model(
                    x,
                    use_cache=False,
                    output_hidden_states=False,
                    block_size=B,  # 若模型forward支持
                )
                logits = out.logits  # (1, L, V) 或 (1, L, V) 具体看实现
            except TypeError:
                # forward不吃 block_size 参数
                try:
                    out = model(x, use_cache=False, output_hidden_states=False)
                    logits = out.logits
                except Exception:
                    # 完全失败 -> fallback AR
                    s, a, mi, ma = _ar_block_stats(i, B)
                    losses[B]["sum"].append(s)
                    losses[B]["avg"].append(a)
                    losses[B]["min"].append(mi)
                    losses[B]["max"].append(ma)
                    continue
            except Exception:
                # 其它错误 -> fallback AR
                s, a, mi, ma = _ar_block_stats(i, B)
                losses[B]["sum"].append(s)
                losses[B]["avg"].append(a)
                losses[B]["min"].append(mi)
                losses[B]["max"].append(ma)
                continue

            # logits shape 可能是 (1, L, V)，其中 L = i+B 或 i+B-1（依模型）
            # 我们需要对齐到 token positions：使用 Fast-dLLM 的 token-shift 技巧：
            # logits_shift[:, t] 用来预测 token t（而不是 t+1）
            if logits.dim() != 3:
                # 异常 -> fallback
                s, a, mi, ma = _ar_block_stats(i, B)
                losses[B]["sum"].append(s)
                losses[B]["avg"].append(a)
                losses[B]["min"].append(mi)
                losses[B]["max"].append(ma)
                continue

            L = logits.size(1)
            # token-shift: [logits[:,0], logits[:,:-1]]
            logits_shift = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

            # 目标 token：用真实 full_ids 的同位置 token
            # 我们只算 masked positions：t ∈ [i, i+B-1]
            # 但要确保 logits_shift 有这些位置
            if i + B > L:
                # 模型输出长度不足 -> fallback
                s, a, mi, ma = _ar_block_stats(i, B)
                losses[B]["sum"].append(s)
                losses[B]["avg"].append(a)
                losses[B]["min"].append(mi)
                losses[B]["max"].append(ma)
                continue

            target = full_ids[:, :L]  # (1, L)
            masked_pos = torch.arange(i, i + B, device=device)

            # 取出 masked positions 的 logits 和 target
            sel_logits = logits_shift[:, masked_pos, :]  # (1, B, V)
            sel_target = target[:, masked_pos]  # (1, B)

            # masked-only CE
            ce = F.cross_entropy(
                sel_logits.reshape(-1, V),
                sel_target.reshape(-1),
                reduction="none",
            ).view(-1)  # (B,)

            losses[B]["sum"].append(ce.sum().item())
            losses[B]["avg"].append(ce.mean().item())
            losses[B]["min"].append(ce.min().item())
            losses[B]["max"].append(ce.max().item())

    return losses


import math


def is_valid(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def dp_route_min_avg(results, k=None, block_sizes=(4, 8, 16, 32), stat="avg"):
    """
    用 DP 找 route，使得 sum over steps of cost(i,B) 最小，其中 cost=results[B][stat][i].
    注意：这是“每块 avg 的加和最小”，不是按 token 数加权的全局 token-avg。
    """
    # 推断长度 k
    if k is None:
        # 取任意一个 block 的 stat 长度
        b0 = next(iter(results.keys()))
        k = len(results[b0][stat])

    dp = [float("inf")] * (k + 1)
    nxt = [None] * (k + 1)
    dp[k] = 0.0

    for i in range(k - 1, -1, -1):
        best = float("inf")
        bestB = None
        for B in block_sizes:
            if i + B <= k:
                c = results[B][stat][i]
                if not is_valid(c):
                    continue
                v = c + dp[i + B]
                if v < best:
                    best = v
                    bestB = B
        dp[i] = best
        nxt[i] = bestB

    # 还原路径
    route = []
    i = 0
    total_cost = 0.0
    while i < k and nxt[i] is not None:
        B = nxt[i]
        c = results[B][stat][i]
        route.append((i, B, c))
        total_cost += c
        i += B

    return route, total_cost, dp[0]


def greedy_route_min_avg(results, k=None, block_sizes=(4, 8, 16, 32), stat="avg"):
    """
    你描述的贪心：在当前位置 i 选 avg 最小的 B，然后跳到 i+B。
    """
    if k is None:
        b0 = next(iter(results.keys()))
        k = len(results[b0][stat])

    route = []
    i = 0
    total_cost = 0.0
    while i < k:
        best = float("inf")
        bestB = None
        for B in block_sizes:
            if i + B <= k:
                c = results[B][stat][i]
                if not is_valid(c):
                    continue
                if c < best:
                    best = c
                    bestB = B
        if bestB is None:
            break
        route.append((i, bestB, best))
        total_cost += best
        i += bestB

    return route, total_cost


def pretty_print_route(route, title="route"):
    print(f"\n{title} (len={len(route)} steps):")
    for (i, B, c) in route[:50]:
        print(f"  i={i:4d}  B={B:2d}  avg_loss={c:.6f}")
    if len(route) > 50:
        print(f"  ... ({len(route) - 50} more steps)")
    # 也可以打印 block size 分布
    from collections import Counter
    cnt = Counter(B for _, B, _ in route)
    print("  block_size counts:", dict(sorted(cnt.items())))


def fixed_route(results, k=None, B=32, allowed=(1, 2, 4, 8, 16, 32), stat="avg"):
    if k is None:
        b0 = next(iter(results.keys()))
        k = len(results[b0][stat])

    route = []
    i = 0
    total_cost = 0.0

    allowed = sorted(set(allowed))
    while i < k:
        # 尽量用主块 B
        if i + B <= k and is_valid(results[B][stat][i]):
            useB = B
        else:
            # 尾巴：找一个能刚好放得下的最大块（或最合适块）
            useB = None
            for b in reversed(allowed):
                if i + b <= k and is_valid(results[b][stat][i]):
                    useB = b
                    break
            if useB is None:
                break

        c = results[useB][stat][i]
        route.append((i, useB, c))
        total_cost += c
        i += useB

    return route, total_cost


def main(
        model_path: str = 'Efficient-Large-Model/Fast_dLLM_v2_7B',
        device: str = 'cuda',
        use_mdm_sample: bool = False,
) -> None:
    block_sizes = [1, 2, 4, 8, 16, 32]
    # Load model and tokenizer
    device_obj = torch.device(device)
    model = Fast_dLLM_QwenForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device_obj)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Bind diffusion sampling method (unused for now)
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
    )
    # Load the first sample from GSM8K
    prompt = prepare_gsm8k_dataset(1)[0]

    results = compute_block_losses_per_token(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        block_sizes=block_sizes,
        use_mdm_sample=use_mdm_sample,
        device=device_obj,
        mask_id=151665,
    )
    print('\nPer‑token block losses for block sizes 4/8/16/32:')
    for bs in block_sizes:
        print(f'Block size {bs:>2}: {results[bs]}')

    import math
    import pandas as pd

    # results: 你的那个 dict
    # results[bs] = {"sum":[...], "avg":[...], "min":[...], "max":[...]}

    def to_long_df(results, stats=("sum", "avg", "min", "max")):
        rows = []
        for bs, d in results.items():
            for stat in stats:
                arr = d[stat]
                for token_idx, v in enumerate(arr):
                    # 统一把 nan 处理掉（可选）
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    rows.append({
                        "block_size": int(bs),
                        "stat": stat,
                        "token_idx": int(token_idx),
                        "value": float(v),
                    })
        return pd.DataFrame(rows)

    df = to_long_df(results)
    df.head()

    import matplotlib.pyplot as plt

    stats = ["sum", "avg", "min", "max"]

    for stat in stats:
        plt.figure()
        sub = df[df["stat"] == stat]

        for bs in block_sizes:
            if bs == 1 or bs == 2:
                continue
            s = sub[sub["block_size"] == bs].sort_values("token_idx")
            if len(s) == 0:
                continue
            plt.plot(s["token_idx"], s["value"], label=f"bs={bs}")

        plt.title(f"{stat} vs token_idx")
        plt.xlabel("token_idx")
        plt.ylabel(stat)
        plt.legend()
        plt.savefig(f'{stat} vs token_idx.png')
        plt.show()

    # ======= 在你拿到 results 后调用 =======
    # k 取 token 长度（和 results list 一致）
    k = len(results[4]["avg"])

    dp_route, dp_sum, dp0 = dp_route_min_avg(results, k=k, block_sizes=block_sizes, stat="avg")
    greedy_route, greedy_sum = greedy_route_min_avg(results, k=k, block_sizes=block_sizes, stat="avg")

    fixed_sums = {}
    fixed_routes = {}
    for B in block_sizes:
        r, s = fixed_route(results, k=k, B=B, stat="avg")
        fixed_routes[B] = r
        fixed_sums[B] = s

    import pandas as pd
    rows = []
    for B in block_sizes:
        rows.append({"method": f"fixed_B={B}", "sum_avg_cost": fixed_sums[B], "steps": len(fixed_routes[B])})
    rows.append({"method": "greedy", "sum_avg_cost": greedy_sum, "steps": len(greedy_route)})
    rows.append({"method": "dp", "sum_avg_cost": dp_sum, "steps": len(dp_route)})

    df_cmp = pd.DataFrame(rows).sort_values("sum_avg_cost")
    print("\n=== Route comparison (objective: minimize sum of per-block avg loss) ===")
    print(df_cmp.to_string(index=False))

    # 需要的话也打印路线细节
    pretty_print_route(dp_route, title="DP route")
    pretty_print_route(greedy_route, title="Greedy route")
    for B in block_sizes:
        pretty_print_route(fixed_routes[B], title=f"Fixed B={B} route")


if __name__ == '__main__':
    fire.Fire(main)
