# v2/dp_prefill_planner.py
import hashlib
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _round_down_to(x: int, m: int) -> int:
    x = int(x)
    m = int(m)
    if m <= 0:
        return x
    return (x // m) * m


def _gcd_of_sizes(sizes: Sequence[int]) -> int:
    g = 0
    for s in sizes:
        g = math.gcd(g, int(s))
    return int(g)


def _plan_for_len_allowed(length: int, allowed_sizes: Sequence[int], prefer_size: Optional[int] = None) -> List[int]:
    """
    Return a plan such that sum(plan) == length and each element is in allowed_sizes.
    DP objective: minimize number of blocks; tie-break towards prefer_size (if provided).
    """
    length = int(length)
    if length <= 0:
        return []

    sizes = sorted({int(s) for s in allowed_sizes if int(s) > 0}, reverse=True)
    if not sizes:
        raise ValueError("allowed_sizes must contain positive ints")

    g = _gcd_of_sizes(sizes)
    if length % g != 0:
        raise ValueError(f"length={length} not divisible by gcd={g} of {sizes}")

    # tie-break order
    if prefer_size is not None and int(prefer_size) in sizes:
        ps = int(prefer_size)
        sizes = [ps] + [s for s in sizes if s != ps]

    rank = {s: i for i, s in enumerate(sizes)}

    INF = 10**9
    dp = [INF] * (length + 1)
    prev: List[Optional[int]] = [None] * (length + 1)
    dp[0] = 0

    for t in range(length + 1):
        if dp[t] == INF:
            continue
        for s in sizes:
            nt = t + s
            if nt > length:
                continue
            cand = dp[t] + 1
            if cand < dp[nt]:
                dp[nt] = cand
                prev[nt] = s
            elif cand == dp[nt]:
                if prev[nt] is None or rank[s] < rank[prev[nt]]:
                    prev[nt] = s

    if dp[length] == INF:
        raise ValueError(f"Cannot build plan for length={length} with allowed_sizes={sizes}")

    plan: List[int] = []
    t = length
    while t > 0:
        s = prev[t]
        if s is None:
            raise RuntimeError("DP reconstruct failed")
        plan.append(int(s))
        t -= int(s)
    plan.reverse()
    return plan


class PrefixDPPlanner:
    """
    DP planner used in eval.py, adapted for prefill.

    It computes a block-size plan to minimize:
        sum over blocks of avg_loss(i, B)
    where avg_loss(i,B) is computed by masking tokens [i:i+B) and
    running a forward pass with fixed block_size=B to get average CE on that block.

    New capability:
      get_plan_with_split(..., split_len=...) enforces that no block crosses split_len,
      so you can use:
        - plan_prefix (sum == split_len) for prefill forward (cache_len=0, seqlen=split_len)
        - plan_total  (sum == total_len) for the *first full-block* decode forward
          (cache_len=split_len, seqlen=block_size).
    """

    def __init__(
        self,
        dp_block_sizes: Sequence[int] = (4, 8, 16, 32),
        dp_cache_size: int = 4096,
        dp_max_analyze_len: Optional[int] = None,
        dp_fixed_block_size: Optional[int] = None,
    ):
        self.dp_block_sizes = tuple(int(x) for x in dp_block_sizes)
        self.dp_cache_size = int(dp_cache_size)
        self.dp_max_analyze_len = dp_max_analyze_len
        self.dp_fixed_block_size = dp_fixed_block_size

        self._allowed_sizes = sorted({int(x) for x in self.dp_block_sizes if int(x) > 0}, reverse=True)
        if not self._allowed_sizes:
            raise ValueError(f"dp_block_sizes must contain positive ints, got {dp_block_sizes}")

        self._gcd = _gcd_of_sizes(self._allowed_sizes)
        if self._gcd <= 0:
            raise ValueError("Invalid gcd(dp_block_sizes)")

        self._cache: "OrderedDict[Tuple, List[int]]" = OrderedDict()

    def _make_key(self, ids_1d_cpu: torch.Tensor, split_len: Optional[int]) -> Tuple:
        arr = ids_1d_cpu.detach().contiguous().numpy()
        h = hashlib.sha1(arr.tobytes()).hexdigest()
        return (
            "ids",
            int(ids_1d_cpu.numel()),
            h,
            self.dp_block_sizes,
            self.dp_max_analyze_len,
            self.dp_fixed_block_size,
            int(split_len) if split_len is not None else None,
        )

    def _cache_get(self, key: Tuple) -> Optional[List[int]]:
        plan = self._cache.get(key, None)
        if plan is None:
            return None
        self._cache.move_to_end(key)
        return plan

    def _cache_put(self, key: Tuple, plan: List[int]) -> None:
        self._cache[key] = plan
        self._cache.move_to_end(key)
        if len(self._cache) > self.dp_cache_size:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def _masked_block_avg_loss_prefix(
        self,
        model,
        ids_device_1d: torch.Tensor,  # full 1D ids on device
        i: int,
        B: int,
        mask_id: int,
        eos_id: int,
    ) -> float:
        """
        x = ids[:i+B], mask x[i:i+B], pad to multiple of B (at least one block), set eos at first pad pos,
        forward with block_size=B, compute avg CE on masked block tokens.
        """
        x = ids_device_1d[: i + B].clone().unsqueeze(0)  # (1, i+B)
        x[:, i : i + B] = int(mask_id)

        L0 = int(x.shape[1])
        pad_len = B - (L0 % B)
        if pad_len == 0:
            pad_len = B
        pad = torch.full((1, pad_len), int(mask_id), dtype=torch.long, device=ids_device_1d.device)
        x = torch.cat([x, pad], dim=1)
        x[:, L0] = int(eos_id)

        out = model.forward(input_ids=x, use_cache=False, block_size=int(B))
        logits = out.logits
        if logits is None or logits.dim() != 3:
            return float("nan")

        logits_shift = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        V = logits_shift.size(-1)
        sel_logits = logits_shift[:, i : i + B, :].reshape(-1, V).float()
        sel_target = ids_device_1d[i : i + B].reshape(-1)
        ce = F.cross_entropy(sel_logits, sel_target, reduction="none")
        return float(ce.mean().item())

    @torch.no_grad()
    def _compute_cost_table_avg(
        self,
        model,
        ids_1d_cpu: torch.Tensor,  # 1D on CPU
        device: torch.device,
        mask_id: int,
        eos_id: int,
    ) -> Dict[int, List[float]]:
        k = int(ids_1d_cpu.numel())
        results: Dict[int, List[float]] = {int(B): [float("nan")] * k for B in self.dp_block_sizes}
        if k == 0:
            return results

        ids_dev = ids_1d_cpu.to(device)
        for B in self.dp_block_sizes:
            B = int(B)
            if B <= 0 or B > k:
                continue
            for i in range(0, k):
                if i + B > k:
                    continue
                results[B][i] = self._masked_block_avg_loss_prefix(model, ids_dev, i, B, mask_id, eos_id)
        return results

    def _dp_plan_min_sum_avg(
        self,
        results_avg: Dict[int, List[float]],
        k: int,
        split_len: Optional[int] = None,
    ) -> List[int]:
        """
        DP objective: minimize sum avg_cost(i,B).

        If split_len is provided, we enforce: no block crosses split_len.
        i.e., disallow transitions i -> i+B with i < split_len < i+B.
        """
        k = int(k)
        if split_len is not None:
            split_len = int(split_len)
            if split_len < 0 or split_len > k:
                raise ValueError(f"split_len={split_len} must be in [0, {k}]")

        dp = [float("inf")] * (k + 1)
        nxt: List[Optional[int]] = [None] * (k + 1)
        dp[k] = 0.0

        for i in range(k - 1, -1, -1):
            best = float("inf")
            bestB: Optional[int] = None
            for B in self.dp_block_sizes:
                B = int(B)
                j = i + B
                if j > k:
                    continue
                if split_len is not None and i < split_len < j:
                    continue
                c = results_avg[B][i]
                if c is None or (isinstance(c, float) and math.isnan(c)):
                    continue
                v = float(c) + dp[j]
                if v < best:
                    best = v
                    bestB = B
            dp[i] = best
            nxt[i] = bestB

        plan: List[int] = []
        i = 0
        while i < k and nxt[i] is not None:
            B = int(nxt[i])
            plan.append(B)
            i += B

        if i != k:
            raise RuntimeError(f"DP failed to cover k={k}, got sum={i}, plan={plan}")
        return plan

    @staticmethod
    def _extract_prefix_plan(plan_total: List[int], split_len: int) -> List[int]:
        split_len = int(split_len)
        if split_len <= 0:
            return []
        acc = 0
        out: List[int] = []
        for B in plan_total:
            if acc + B > split_len:
                raise RuntimeError(
                    f"Plan crosses split_len={split_len}: accumulated={acc}, next block={B}, plan={plan_total}"
                )
            out.append(int(B))
            acc += int(B)
            if acc == split_len:
                return out
        raise RuntimeError(f"Plan does not reach split_len={split_len}: sum(plan)={sum(plan_total)}")

    def get_plan(
        self,
        model,
        prefix_ids_1d: torch.Tensor,  # 1D ids (CPU or GPU)
        device: torch.device,
        mask_id: int,
        eos_id: int,
        prefer_block_size: int,
        force_recompute: bool = False,
    ) -> List[int]:
        prefix_cpu = prefix_ids_1d.detach().cpu()
        key = self._make_key(prefix_cpu, split_len=None)
        if not force_recompute:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        P = int(prefix_cpu.numel())
        if P == 0:
            self._cache_put(key, [])
            return []

        if P % self._gcd != 0:
            raise ValueError(
                f"prefix length P={P} not divisible by gcd(dp_block_sizes)={self._gcd}. "
                f"Use a gcd-aligned length (e.g., round up) for DP planning."
            )

        fixed_plan: List[int] = []
        ids_for_dp = prefix_cpu

        # Optional approximate truncation (same spirit as eval.py)
        if self.dp_max_analyze_len is not None and P > int(self.dp_max_analyze_len):
            analyze_len = _round_down_to(int(self.dp_max_analyze_len), self._gcd)
            if analyze_len <= 0:
                analyze_len = self._gcd
            fixed_len = P - analyze_len

            fixed_bs = int(self.dp_fixed_block_size) if self.dp_fixed_block_size is not None else int(prefer_block_size)
            fixed_plan = _plan_for_len_allowed(fixed_len, self._allowed_sizes, prefer_size=fixed_bs)
            ids_for_dp = prefix_cpu[fixed_len:]

        k = int(ids_for_dp.numel())
        results_avg = self._compute_cost_table_avg(model, ids_for_dp, device=device, mask_id=mask_id, eos_id=eos_id)
        suffix_plan = self._dp_plan_min_sum_avg(results_avg, k=k, split_len=None)

        plan = fixed_plan + suffix_plan
        if sum(plan) != P:
            raise RuntimeError(f"Internal error: sum(plan)={sum(plan)} != P={P}")

        self._cache_put(key, plan)
        return plan

    def get_plan_with_split(
        self,
        model,
        ids_1d: torch.Tensor,  # full ids (CPU or GPU) of length total_len (prefill_len + block_size)
        split_len: int,        # boundary at which cache ends (prefill_len)
        device: torch.device,
        mask_id: int,
        eos_id: int,
        prefer_block_size: int,
        force_recompute: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """
        Return (plan_prefix, plan_total) with:
          sum(plan_total) == len(ids_1d)
          sum(plan_prefix) == split_len
          plan_total does NOT contain any block crossing split_len.
        """
        ids_cpu = ids_1d.detach().cpu()
        P = int(ids_cpu.numel())
        split_len = int(split_len)

        if split_len < 0 or split_len > P:
            raise ValueError(f"split_len={split_len} must be in [0, {P}]")
        if P == 0:
            return ([], [])
        if P % self._gcd != 0:
            raise ValueError(
                f"total length P={P} not divisible by gcd(dp_block_sizes)={self._gcd}. "
                f"Round to a gcd-aligned length before calling get_plan_with_split."
            )
        if split_len % self._gcd != 0:
            raise ValueError(
                f"split_len={split_len} not divisible by gcd(dp_block_sizes)={self._gcd}. "
                f"Prefill length must align to gcd so DP blocks can end exactly at split."
            )

        key = self._make_key(ids_cpu, split_len=split_len)
        if not force_recompute:
            cached = self._cache_get(key)
            if cached is not None:
                plan_total = cached
                plan_prefix = self._extract_prefix_plan(plan_total, split_len)
                return plan_prefix, plan_total

        # Optional approximate truncation
        fixed_plan: List[int] = []
        ids_for_dp = ids_cpu
        split_in_slice: Optional[int] = split_len

        if self.dp_max_analyze_len is not None and P > int(self.dp_max_analyze_len):
            analyze_len = _round_down_to(int(self.dp_max_analyze_len), self._gcd)
            if analyze_len <= 0:
                analyze_len = self._gcd
            fixed_len = P - analyze_len

            fixed_bs = int(self.dp_fixed_block_size) if self.dp_fixed_block_size is not None else int(prefer_block_size)

            if split_len <= fixed_len:
                # boundary is inside fixed region -> enforce by splitting fixed_plan
                fixed_plan = (
                    _plan_for_len_allowed(split_len, self._allowed_sizes, prefer_size=fixed_bs)
                    + _plan_for_len_allowed(fixed_len - split_len, self._allowed_sizes, prefer_size=fixed_bs)
                )
                split_in_slice = None  # already satisfied
            else:
                fixed_plan = _plan_for_len_allowed(fixed_len, self._allowed_sizes, prefer_size=fixed_bs)
                split_in_slice = split_len - fixed_len  # boundary inside DP slice

            ids_for_dp = ids_cpu[fixed_len:]

        k = int(ids_for_dp.numel())
        results_avg = self._compute_cost_table_avg(model, ids_for_dp, device=device, mask_id=mask_id, eos_id=eos_id)
        dp_plan = self._dp_plan_min_sum_avg(results_avg, k=k, split_len=split_in_slice)

        plan_total = fixed_plan + dp_plan
        if sum(plan_total) != P:
            raise RuntimeError(f"Internal error: sum(plan_total)={sum(plan_total)} != P={P}")

        # validate boundary
        _ = self._extract_prefix_plan(plan_total, split_len)

        self._cache_put(key, plan_total)
        plan_prefix = self._extract_prefix_plan(plan_total, split_len)
        return plan_prefix, plan_total