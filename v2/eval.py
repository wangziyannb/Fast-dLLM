# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import hashlib
import math
from datetime import timedelta
from typing import OrderedDict, Tuple, List, Optional, Sequence, Dict, Any

import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from accelerate import InitProcessGroupKwargs
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import time
import types
from modeling import Fast_dLLM_QwenForCausalLM
from v2 import generation_functions_new


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("fast_dllm_v2")
class Fast_dLLM_v2EvalHarness(LM):

    def __init__(
            self,
            model_path='Efficient-Large-Model/Fast_dLLM_v2_7B',
            device="cuda",
            show_speed=False,
            max_new_tokens=2048,
            batch_size=32,
            mask_id=151665,
            use_block_cache=False,
            small_block_size=32,
            bd_size=32,
            threshold=0.9,
            speed_log_path=None,
            # ===== DP planning for prefix (loglikelihood) =====
            dp_block_sizes=(4, 8, 16, 32),
            dp_max_analyze_len=None,  # 可选：只对 prefix 末尾 N tokens 做 DP（默认 None=全量）
            dp_cache_size=4096,  # prefix plan LRU cache
            dp_fixed_block_size=None,  # 若 dp_max_analyze_len 截断，前面那段用固定块大小（默认用 self.bd_size）
            debug_prefix_dp=False,
            debug_prefix_dp_first_n=10,  # 只打印前 N 个样本，<=0 表示全打印
            debug_prefix_dp_force_recompute=False,  # True: 即使cache命中也重新算一次(方便你一直看)
            **kwargs,
    ):
        super().__init__()
        pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=120))
        accelerator = accelerate.Accelerator(kwargs_handlers=[pg_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **model_kwargs
        )
        self.model.eval()

        self.model.mdm_sample = types.MethodType(generation_functions_new.Fast_dLLM_QwenForCausalLM.batch_sample,
                                                 self.model)

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.show_speed = show_speed
        self.max_new_tokens = max_new_tokens
        self.batch_size = int(batch_size)
        self.mask_id = mask_id
        self.model_path = model_path
        self.use_block_cache = use_block_cache
        self.small_block_size = small_block_size
        self.threshold = threshold
        self.bd_size = bd_size
        self.speed_log_path = speed_log_path
        # LRU cache: key -> plan_prefix (List[int])
        self._prefix_plan_cache: "OrderedDict[Tuple, List[int]]" = OrderedDict()
        self.dp_max_analyze_len = dp_max_analyze_len
        self.dp_cache_size = dp_cache_size
        self.dp_fixed_block_size = dp_fixed_block_size
        self.dp_block_sizes = dp_block_sizes
        self.debug_prefix_dp = bool(debug_prefix_dp)
        self.debug_prefix_dp_first_n = int(debug_prefix_dp_first_n)
        self.debug_prefix_dp_force_recompute = bool(debug_prefix_dp_force_recompute)
        self._debug_prefix_dp_cnt = 0

        # ---- derived: allowed block sizes & gcd alignment ----
        self._dp_allowed_sizes = sorted({int(x) for x in self.dp_block_sizes if int(x) > 0}, reverse=True)
        if not self._dp_allowed_sizes:
            raise ValueError(f"dp_block_sizes must contain positive ints, got {self.dp_block_sizes}")

        g = 0
        for x in self._dp_allowed_sizes:
            g = math.gcd(g, x)
        self._dp_gcd = int(g)
        if self._dp_gcd <= 0:
            raise ValueError(f"Invalid dp_block_sizes={self.dp_block_sizes}")

        # 建议加一个一致性检查：total_len 是 bd_size 的倍数，最好也能被 dp_gcd 整除
        if int(self.bd_size) % self._dp_gcd != 0:
            raise ValueError(
                f"bd_size={self.bd_size} must be divisible by gcd(dp_block_sizes)={self._dp_gcd} "
                f"to build a valid block plan without illegal tail blocks."
            )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.model_path

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt,
                                                  tokenize=False)

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        batch[:, prompt_index.sum()] = self.mask_id

        batch = torch.cat([batch.to(self.device),
                           torch.full((b, self.bd_size - batch.shape[1] % self.bd_size), self.mask_id, dtype=torch.long,
                                      device=self.device)], dim=1)
        if batch.shape[1] > l:
            batch[:, l] = self.tokenizer.eos_token_id

        return batch

    # @torch.no_grad()
    # def get_logits(self, batch):
    #     logits = self.model(batch, block_size=self.bd_size).logits
    #     logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    #     return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_logits(self, batch, block_sizes: Optional[torch.Tensor] = None):
        if block_sizes is None:
            logits = self.model(batch, use_cache=False).logits
        else:
            logits = self.model(batch, use_cache=False, block_sizes=block_sizes).logits

        # align: logits_shift[:, t] predicts token t
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        return logits[:, :batch.shape[1]]

    # @torch.no_grad()
    # def get_loglikelihood(self, prefix, target):
    #     seq = torch.concatenate([prefix, target])[None, :]
    #
    #     prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
    #
    #     loss_acc = []
    #
    #     perturbed_seq = self._forward_process(seq.clone(), prompt_index)
    #
    #     mask_indices = perturbed_seq == self.mask_id
    #
    #     logits = self.get_logits(perturbed_seq)
    #     seq = torch.cat([seq.to(self.device), torch.full((seq.shape[0], self.bd_size-seq.shape[1]%self.bd_size), -100, dtype=torch.long, device=self.device)], dim=1)
    #     loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none')
    #     loss = loss.sum()
    #     loss_acc.append(loss.item())
    #
    #     return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target, prefix_text: Optional[str] = None):
        # seq: (1, L0)
        seq = torch.concatenate([prefix, target])[None, :]

        prefix_len = int(prefix.numel())

        # keep prompt_index on CPU (safer for indexing)
        prompt_index = (torch.arange(seq.shape[1]) < prefix_len)

        loss_acc = []

        perturbed_seq = self._forward_process(seq.clone(), prompt_index)  # (1, L1) on device
        total_len = int(perturbed_seq.shape[1])

        # # # ---- 1) DP analyze prefix -> plan_prefix ----
        # # plan_prefix = self._get_or_compute_prefix_plan(prefix, prefix_text=prefix_text)
        # #
        # # # ---- 2) build total plan for full forward (prefix plan + fixed tail) ----
        # # plan_total = self._build_total_block_plan(
        # #     plan_prefix=plan_prefix,
        # #     prefix_len=prefix_len,
        # #     total_len=total_len,
        # # )
        # # block_sizes_tensor = torch.tensor(plan_total, device=self.device, dtype=torch.long)
        #
        # # ---- 1) DP analyze full seq (prefix + target) -> plan_seq ----
        # seq_len = int(seq.shape[1])  # = prefix_len + target_len
        # seq_ids_1d = seq[0].detach().cpu()  # 1D ids on CPU for DP caching/hash
        #
        # # 用 id hash 做 key，避免你只传 prefix_text 导致 key 不包含 target
        # plan_seq = self._get_or_compute_prefix_plan(seq_ids_1d, prefix_text=None)
        #
        # # ---- 2) build total plan: DP for [0:seq_len), fixed for the rest (eos/pad) ----
        # plan_total = self._build_total_block_plan(
        #     plan_prefix=plan_seq,
        #     prefix_len=seq_len,  # 注意这里用 seq_len，不是原来的 prefix_len
        #     total_len=total_len,
        # )
        # block_sizes_tensor = torch.tensor(plan_total, device=self.device, dtype=torch.long)

        # ---- 1) DP analyze a gcd-aligned prefix length (seq + maybe EOS/pad) ----
        seq_len = int(seq.shape[1])  # prefix+target length (before _forward_process padding)
        seq_ids_1d = seq[0].detach().cpu()  # true (unmasked) ids for seq part

        align = int(self._dp_gcd)  # gcd(dp_block_sizes), e.g. 4
        dp_len = self._round_up_to(seq_len, align)  # make it representable by {4,8,16,32}
        if dp_len > total_len:
            # 理论上不该发生（因为 total_len 是 32 的倍数且 >= seq_len+1）
            dp_len = total_len

        # ids for DP:
        #   - [0:seq_len) 用真实 token（不要用 perturbed_seq，那里 prefix_len 位置被 mask 了）
        #   - [seq_len:dp_len) 用 _forward_process 产生的 EOS + padding masks（不会进 loss）
        ids_dp = torch.empty((dp_len,), dtype=torch.long)
        ids_dp[:seq_len] = seq_ids_1d
        if dp_len > seq_len:
            ids_dp[seq_len:dp_len] = perturbed_seq[0, seq_len:dp_len].detach().cpu()

        plan_dp = self._get_or_compute_prefix_plan(ids_dp, prefix_text=None)

        # ---- 2) build total plan for whole perturbed sequence ----
        plan_total = self._build_total_block_plan(
            plan_prefix=plan_dp,
            prefix_len=dp_len,
            total_len=total_len,
        )
        block_sizes_tensor = torch.tensor(plan_total, device=self.device, dtype=torch.long)

        # ---- 3) forward with variable block_sizes ----
        mask_indices = perturbed_seq == self.mask_id
        logits = self.get_logits(perturbed_seq, block_sizes=block_sizes_tensor)

        # pad labels to same length as perturbed_seq (keep your original behavior)
        pad_len = self.bd_size - seq.shape[1] % self.bd_size
        seq_padded = torch.cat(
            [
                seq.to(self.device),
                torch.full((seq.shape[0], pad_len), -100, dtype=torch.long, device=self.device),
            ],
            dim=1,
        )

        loss = F.cross_entropy(logits[mask_indices], seq_padded[mask_indices], reduction='none')
        loss = loss.sum()
        loss_acc.append(loss.item())

        # ===== debug: compare multiple fixed-block baselines vs dp =====
        if self.rank == 0:
            cand_Bs = [4, 8, 16, 32]

            # DP loss（你当前已算好的）
            loss_dp = float(loss.item())

            eff_tokens = int((seq_padded[mask_indices] != -100).sum().item())

            # 跑一组 fixed-B baselines
            fixed_losses = {}
            for B in cand_Bs:
                if total_len % B != 0:
                    # 理论上 total_len 是 32 的倍数时不会发生（除非 self.bd_size 不是 32 的倍数）
                    continue

                plan_fixed_B = [int(B)] * (total_len // int(B))
                block_sizes_fixed = torch.tensor(plan_fixed_B, device=self.device, dtype=torch.long)

                logits_fixed = self.get_logits(perturbed_seq, block_sizes=block_sizes_fixed)
                loss_fixed = F.cross_entropy(
                    logits_fixed[mask_indices],
                    seq_padded[mask_indices],
                    reduction='none'
                ).sum().item()

                fixed_losses[B] = loss_fixed

            # 打印对比
            items = " ".join([f"B={B}: {fixed_losses[B]:.6f}" for B in cand_Bs if B in fixed_losses])
            print(
                f"[DP DEBUG] eff_tokens={eff_tokens} "
                f"fixed_losses(sum CE) [{items}] | dp_loss(sum CE)={loss_dp:.6f} "
                f"seq_len={seq_len} dp_len={dp_len} total_len={total_len}"
            )
            print(f"[DP DEBUG] plan_dp(sum={sum(plan_dp)}, n={len(plan_dp)}): {plan_dp[:32]}")
        return - sum(loss_acc) / len(loss_acc)

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                # ll = self.get_loglikelihood(prefix, target)
                ll = self.get_loglikelihood(prefix, target, prefix_text=elem["prefix_text"])
                out.append((ll, 0.0))
        torch.cuda.empty_cache()
        return out

    def generate_until(self, requests):
        output = [None] * len(requests)  # pre-allocate output list
        num_tokens = 0

        start_time = time.time()

        requests_with_indices = [(i, req) for i, req in enumerate(requests)]
        requests_with_indices.sort(key=lambda x: len(x[1].args[0]))

        batched_requests = []
        current_batch = []
        for i, req in requests_with_indices:
            current_batch.append((i, req))
            if len(current_batch) == self.batch_size:
                batched_requests.append(current_batch)
                current_batch = []

        if current_batch:
            batched_requests.append(current_batch)

        for _, batch in enumerate(tqdm(batched_requests, desc="Generating...")):
            batched_input_ids = []
            max_len = 0
            min_len = 1e9
            seq_len = []

            for orig_idx, req in batch:
                question = req.args[0]

                if req.task_name.startswith('minerva_math'):
                    question = question.replace("Solution:",
                                                "Please reason step by step, and put your final answer within \\boxed{{}}.")
                elif req.task_name.startswith('gsm8k'):
                    question = question.replace("Answer:",
                                                "Please reason step by step, and put your final answer within \\boxed{{}}.")
                model_inputs = self.tokenizer([question], return_tensors="pt").to(self.device)
                batched_input_ids.append(model_inputs["input_ids"])
                max_len = max(max_len, model_inputs["input_ids"].shape[1])
                min_len = min(min_len, model_inputs["input_ids"].shape[1])
                seq_len.append(model_inputs["input_ids"].shape[1])

            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([input_ids, torch.full((1, max_len - input_ids.shape[1]), self.mask_id,
                                                                  dtype=torch.long, device=self.device)], dim=1) for
                                 input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            with torch.no_grad():
                if self.accelerator is not None:
                    generated_ids = self.accelerator.unwrap_model(self.model).mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                    )
                else:
                    generated_ids = self.model.mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                        use_dp_prefill=True,
                        dp_max_analyze_len=None
                    )

            # extract new generated tokens, and keep original index order
            for batch_pos, (orig_idx, req) in enumerate(batch):
                generated_answer = self.tokenizer.decode(
                    generated_ids[batch_pos][seq_len[batch_pos]:],
                    skip_special_tokens=True
                )

                # count token number
                if self.show_speed:
                    num_tokens += (generated_ids[batch_pos][seq_len[batch_pos]:] != self.mask_id).sum()

                # put result in the correct original index position
                output[orig_idx] = generated_answer

                print('=' * 20)
                print('question: ', req.args[0])
                print('answer: ', generated_answer)
                print('=' * 20, end='\n\n')

        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            if self.speed_log_path is not None:
                # save speed, self.small_block_size, bd_size, into this json file
                import json
                from datetime import datetime

                timestamp = datetime.now().isoformat().replace(':', '-')
                log_data = {
                    "timestamp": timestamp,
                    "num_tokens": num_tokens.item(),
                    "time_taken_second": end_time - start_time,
                    "tokens_per_second": (num_tokens / (end_time - start_time)).item(),
                    "small_block_size": self.small_block_size,
                    "bd_size": self.bd_size
                }

                with open(self.speed_log_path + "_" + timestamp + '.json', 'w') as f:
                    json.dump(log_data, f, indent=2)

                print(f"Speed log saved to: {self.speed_log_path}")

        return output

    # -------------------------
    # DP utilities (prefix plan)
    # -------------------------
    def _is_valid_cost(self, x) -> bool:
        return x is not None and not (isinstance(x, float) and math.isnan(x))

    def _make_prefix_cache_key(
            self,
            prefix_ids: torch.Tensor,
            prefix_text: Optional[str] = None,
    ) -> Tuple:
        """
        Prefer prefix_text as cache key (cheap & stable); fallback to sha1(ids).
        """
        if prefix_text is not None:
            return ("text", prefix_text)

        # fallback: hash ids bytes
        ids = prefix_ids.detach().cpu().numpy()
        h = hashlib.sha1(ids.tobytes()).hexdigest()
        return ("ids", int(prefix_ids.numel()), h)

    def _cache_get_prefix_plan(self, key: Tuple) -> Optional[List[int]]:
        plan = self._prefix_plan_cache.get(key, None)
        if plan is None:
            return None
        # refresh LRU
        self._prefix_plan_cache.move_to_end(key)
        return plan

    def _cache_put_prefix_plan(self, key: Tuple, plan: List[int]) -> None:
        self._prefix_plan_cache[key] = plan
        self._prefix_plan_cache.move_to_end(key)
        if len(self._prefix_plan_cache) > self.dp_cache_size:
            self._prefix_plan_cache.popitem(last=False)

    def _fixed_plan_for_len(self, length: int, block_size: int) -> List[int]:
        if length <= 0:
            return []
        plan = []
        remaining = length
        while remaining > 0:
            take = min(block_size, remaining)
            plan.append(int(take))
            remaining -= take
        return plan

    @torch.no_grad()
    def _masked_block_avg_loss_prefix(
            self,
            prefix_ids_device: torch.Tensor,  # 1D on device
            i: int,
            B: int,
    ) -> float:
        """
        Compute masked-block avg CE on prefix only:
          x = prefix[:i+B], mask x[i:i+B]
          forward with block_size=B (fixed) to get logits
          shift logits and compute CE only on masked positions.
        Returns avg loss over the B tokens.
        """
        # x: (1, i+B)
        # x: (1, i+B)
        x = prefix_ids_device[: i + B].clone().unsqueeze(0)
        x[:, i:i + B] = self.mask_id

        # pad to multiple of B, and always append at least one block (mimic _forward_process behavior)
        L0 = int(x.shape[1])
        pad_len = B - (L0 % B)
        if pad_len == 0:
            pad_len = B

        pad = torch.full((1, pad_len), self.mask_id, dtype=torch.long, device=self.device)
        x = torch.cat([x, pad], dim=1)

        # mark EOS at the first padding position
        x[:, L0] = self.tokenizer.eos_token_id

        out = self.model(
            x,
            use_cache=False,
            output_hidden_states=False,
            block_size=B,  # NOTE: analysis step uses scalar B (same spirit as your earlier DP script)
        )
        logits = out.logits  # (1, L, V)

        if logits.dim() != 3:
            return float("nan")

        # token-shift alignment: logits_shift[:, t] predicts token t
        logits_shift = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        V = logits_shift.size(-1)

        sel_logits = logits_shift[:, i:i + B, :].reshape(-1, V).float()
        sel_target = prefix_ids_device[i:i + B].reshape(-1)

        ce = F.cross_entropy(sel_logits, sel_target, reduction="none")  # (B,)
        return float(ce.mean().item())

    @torch.no_grad()
    def _compute_prefix_cost_table_avg(
            self,
            prefix_ids: torch.Tensor,  # 1D CPU ok
            block_sizes: Sequence[int],
    ) -> Dict[int, List[float]]:
        """
        results[B][i] = avg masked-block loss for block starting at i with size B (NaN if i+B>k)
        """
        k = int(prefix_ids.numel())
        results: Dict[int, List[float]] = {int(B): [float("nan")] * k for B in block_sizes}
        if k == 0:
            return results

        prefix_ids_device = prefix_ids.to(self.device)

        for B in block_sizes:
            B = int(B)
            if B <= 0:
                continue
            if B > k:
                continue
            for i in range(0, k):
                if i + B > k:
                    # keep NaN
                    continue
                results[B][i] = self._masked_block_avg_loss_prefix(prefix_ids_device, i, B)

        return results

    def _dp_plan_min_sum_avg(
            self,
            results_avg: Dict[int, List[float]],
            k: int,
            block_sizes: Sequence[int],
    ) -> List[int]:
        """
        DP objective: minimize sum over steps of avg_cost(i,B)
        Exactly the same objective as your earlier dp_route_min_avg(stat="avg").
        Returns: plan as list of block sizes whose sum == k (or shorter if infeasible).
        """
        dp = [float("inf")] * (k + 1)
        nxt: List[Optional[int]] = [None] * (k + 1)
        dp[k] = 0.0

        for i in range(k - 1, -1, -1):
            best = float("inf")
            bestB = None
            for B in block_sizes:
                B = int(B)
                if i + B <= k:
                    c = results_avg[B][i]
                    if not self._is_valid_cost(c):
                        continue
                    v = c + dp[i + B]
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

        return plan

    def _get_or_compute_prefix_plan(
            self,
            prefix_ids: torch.Tensor,  # 1D CPU tensor from dataset
            prefix_text: Optional[str] = None,
    ) -> List[int]:
        """
        Returns a block-size plan for the *entire prefix length*.
        If dp_max_analyze_len is set and prefix is longer, only DP the suffix;
        the earlier part uses fixed blocks (dp_fixed_block_size or self.bd_size).
        """
        key = self._make_prefix_cache_key(prefix_ids, prefix_text=prefix_text)
        cached = self._cache_get_prefix_plan(key)
        if cached is not None and not self.debug_prefix_dp_force_recompute:
            return cached

        P = int(prefix_ids.numel())
        # 必须可被 gcd(dp_block_sizes) 整除，否则无法用 allowed sizes 精确覆盖
        if P % int(self._dp_gcd) != 0:
            raise ValueError(
                f"prefix length P={P} is not divisible by gcd(dp_block_sizes)={self._dp_gcd}. "
                f"Caller should pass a gcd-aligned ids tensor (e.g. extend into EOS/pad)."
            )

        if P == 0:
            self._cache_put_prefix_plan(key, [])
            return []

        # Optional: analyze only suffix to control cost
        fixed_plan: List[int] = []
        ids_for_dp = prefix_ids
        if self.dp_max_analyze_len is not None and P > int(self.dp_max_analyze_len):
            analyze_len = int(self.dp_max_analyze_len)
            analyze_len = self._round_down_to(analyze_len, int(self._dp_gcd))  # gcd-aligned
            fixed_len = P - analyze_len

            fixed_bs = self.dp_fixed_block_size if self.dp_fixed_block_size is not None else int(self.bd_size)
            fixed_plan = self._plan_for_len_allowed(
                fixed_len,
                allowed_sizes=self._dp_allowed_sizes,
                prefer_size=int(fixed_bs),
            )
            ids_for_dp = prefix_ids[fixed_len:]  # suffix


        k = int(ids_for_dp.numel())
        results_avg = self._compute_prefix_cost_table_avg(ids_for_dp, self.dp_block_sizes)
        suffix_plan = self._dp_plan_min_sum_avg(results_avg, k=k, block_sizes=self.dp_block_sizes)

        # ---------- DEBUG: print DP objective (not final target loss) ----------
        if self.debug_prefix_dp and (self.rank == 0):
            if (self.debug_prefix_dp_first_n <= 0) or (self._debug_prefix_dp_cnt < self.debug_prefix_dp_first_n):
                # 1) DP route objective
                dp_obj = self._dp_obj_sum_avg_from_table(results_avg, suffix_plan, k)

                # 2) Fixed-B objectives for B in {1,2,4,8,16,32}
                fixed_objs = {}
                fixed_plans = {}
                for B in self.dp_block_sizes:
                    plan_fixed = self._fixed_B_plan_with_leftover(k, int(B), self.dp_block_sizes)
                    # plan_fixed = self._fixed_B_plan_with_leftover(k, int(B), [1,4,8,16,32])
                    obj_fixed = self._dp_obj_sum_avg_from_table(results_avg, plan_fixed, k)
                    fixed_objs[int(B)] = obj_fixed
                    fixed_plans[int(B)] = plan_fixed

                # 3) 打印（你可以按需要精简）
                P = int(prefix_ids.numel())
                fixed_len = P - k  # 如果 dp_max_analyze_len 生效，这里>0
                print("\n" + "=" * 80)
                print(
                    f"[PREFIX-DP DEBUG #{self._debug_prefix_dp_cnt}] prefix_total_len={P}, fixed_len={fixed_len}, dp_analyze_len={k}")
                print(f"  dp_block_sizes={list(map(int, self.dp_block_sizes))}")
                print(
                    f"  DP route: obj(sum(avg))={dp_obj:.6f}, n_blocks={len(suffix_plan)}, first_blocks={suffix_plan[:20]}")

                # 固定B：按 B 从小到大打印
                for B in sorted(fixed_objs.keys()):
                    obj = fixed_objs[B]
                    plan_preview = fixed_plans[B][:20]
                    print(
                        f"  FIXED B={B:>2}: obj(sum(avg))={obj:.6f}, n_blocks={len(fixed_plans[B])}, first_blocks={plan_preview}")

                bestB = min(fixed_objs, key=lambda x: fixed_objs[x])
                print(f"  best_fixed_B={bestB}, best_fixed_obj={fixed_objs[bestB]:.6f}")
                if math.isfinite(dp_obj) and math.isfinite(fixed_objs[bestB]):
                    print(f"  dp_vs_best_fixed: delta={dp_obj - fixed_objs[bestB]:.6f} (negative means DP better)")
                print("=" * 80 + "\n")

                self._debug_prefix_dp_cnt += 1
        # ---------- DEBUG END ----------

        plan = fixed_plan + suffix_plan

        # Sanity: sum(plan) should equal total prefix length P (unless infeasible)
        if sum(plan) != P:
            missing = P - sum(plan)
            if missing > 0:
                fixed_bs = self.dp_fixed_block_size if self.dp_fixed_block_size is not None else int(self.bd_size)
                plan += self._plan_for_len_allowed(
                    missing,
                    allowed_sizes=self._dp_allowed_sizes,
                    prefer_size=int(fixed_bs),
                )


        self._cache_put_prefix_plan(key, plan)
        return plan

    def _build_total_block_plan(
            self,
            plan_prefix: List[int],
            prefix_len: int,
            total_len: int,
    ) -> List[int]:
        """
        Build block_sizes plan for the whole perturbed sequence:
          - prefix part: plan_prefix (sum == prefix_len)
          - rest part (target + eos + pad): fill with fixed self.bd_size blocks
        """
        if prefix_len < 0 or total_len < 0 or prefix_len > total_len:
            raise ValueError(f"Invalid lengths: prefix_len={prefix_len}, total_len={total_len}")

        if sum(plan_prefix) != prefix_len:
            # Be strict: we rely on this for correctness of block_sizes sum.
            raise ValueError(f"plan_prefix sum {sum(plan_prefix)} != prefix_len {prefix_len}")

        rest_len = total_len - prefix_len
        plan_rest = self._plan_for_len_allowed(
            rest_len,
            allowed_sizes=self._dp_allowed_sizes,
            prefer_size=int(self.bd_size),
        )
        plan_total = list(plan_prefix) + plan_rest


        if sum(plan_total) != total_len:
            raise ValueError(f"plan_total sum {sum(plan_total)} != total_len {total_len}")

        return plan_total

    def _dp_obj_sum_avg_from_table(
            self,
            results_avg: Dict[int, List[float]],
            plan: List[int],
            k: int,
    ) -> float:
        """
        计算 DP 目标值：sum over blocks of avg_cost(i,B)
        plan 覆盖长度 k
        """
        s = 0.0
        i = 0
        for B in plan:
            B = int(B)
            c = results_avg[B][i]
            if not self._is_valid_cost(c):
                return float("inf")
            s += float(c)
            i += B
        if i != k:
            return float("inf")
        return s

    def _fixed_B_plan_with_leftover(
            self,
            k: int,
            B: int,
            allowed_sizes: Sequence[int],
    ) -> List[int]:
        """
        固定尽可能多用 B，剩余 rem 用 allowed_sizes 精确填满。
        不假设 allowed_sizes 里有 1，因此不会死循环。
        """
        k = int(k)
        B = int(B)
        if k <= 0:
            return []
        if B <= 0:
            raise ValueError(f"Invalid B={B}")

        plan = [B] * (k // B)
        rem = k % B
        if rem:
            plan += self._plan_for_len_allowed(rem, allowed_sizes=allowed_sizes)
        if sum(plan) != k:
            raise RuntimeError(f"fixed-B plan sum {sum(plan)} != k {k}, plan={plan}")
        return plan


    def _round_up_to(self, x: int, m: int) -> int:
        x = int(x); m = int(m)
        if m <= 0:
            return x
        return ((x + m - 1) // m) * m

    def _round_down_to(self, x: int, m: int) -> int:
        x = int(x); m = int(m)
        if m <= 0:
            return x
        return (x // m) * m

    def _plan_for_len_allowed(
        self,
        length: int,
        allowed_sizes: Sequence[int],
        prefer_size: Optional[int] = None,
    ) -> List[int]:
        """
        返回一个 plan，使得：
          - 每个 block size 都属于 allowed_sizes
          - sum(plan) == length
        用 DP 保证“不会出现尾块 19/3/2/1 这种非法尺寸”。
        """
        length = int(length)
        if length <= 0:
            return []

        sizes = sorted({int(s) for s in allowed_sizes if int(s) > 0}, reverse=True)
        if not sizes:
            raise ValueError("allowed_sizes must contain positive ints")

        # necessary divisibility check
        g = 0
        for s in sizes:
            g = math.gcd(g, s)
        if length % g != 0:
            raise ValueError(f"length={length} not divisible by gcd={g} of allowed_sizes={sizes}")

        if prefer_size is not None:
            ps = int(prefer_size)
            if ps in sizes:
                sizes = [ps] + [s for s in sizes if s != ps]

        # DP: minimize number of blocks; tie-break by sizes order (earlier is preferred)
        rank = {s: i for i, s in enumerate(sizes)}
        INF = 10**9
        dp = [INF] * (length + 1)
        prev: List[Optional[int]] = [None] * (length + 1)
        dp[0] = 0

        for t in range(0, length + 1):
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

        if sum(plan) != length:
            raise RuntimeError(f"plan sum {sum(plan)} != length {length} (plan={plan})")
        return plan

    def _safe_id_to_token(self, tid: int) -> str:
        # 不同 tokenizer 可能行为不同，做个兜底
        try:
            tok = self.tokenizer.convert_ids_to_tokens(tid)
            if tok is None:
                tok = self.tokenizer.decode([tid], skip_special_tokens=False)
            return tok
        except Exception:
            return self.tokenizer.decode([tid], skip_special_tokens=False)

    @torch.no_grad()
    def analyze_tokenwise_bigblock(
            self,
            prefix_ids: torch.Tensor,  # 1D (CPU or GPU ok)
            target_ids: torch.Tensor,  # 1D
            block_size_big: Optional[int] = None,
            return_prefix_tokens: bool = False,
            compute_top1_prob: bool = True,
            topk_for_print: int = 0,  # >0 则额外给出 topk token（会更慢）
    ) -> Dict[str, Any]:
        """
        用一个“很大的 block_size”做一次 prefill，然后输出逐 token 的：
          - nll (=-log p_true)
          - p_true
          - top1 token / margin（以及可选 top1_prob）
        默认只返回 target 区间；return_prefix_tokens=True 会把 prefix 区间也返回（pos=1..prefix_len-1）。
        """
        # ---- 0) 准备输入 ----
        prefix_ids = prefix_ids.detach().to("cpu", dtype=torch.long).view(-1)
        target_ids = target_ids.detach().to("cpu", dtype=torch.long).view(-1)

        seq = torch.cat([prefix_ids, target_ids], dim=0)  # (L,)
        L = int(seq.numel())
        prefix_len = int(prefix_ids.numel())
        assert L > 0

        # ---- 1) 选一个“极大的 block_size”（尽量一块吃完，且 padding 最少）----
        # 推荐：block_size 取到 >=L 且是 bd_size 的倍数，最多补 31 个 token
        if block_size_big is None:
            B = int(self._round_up_to(L, int(self.bd_size)))  # e.g., round up to multiple of 32
        else:
            B = int(block_size_big)
            if B < L:
                # 保证 block_size >= L（否则无法“一块 prefill”）
                B = int(self._round_up_to(L, B))

        # ---- 2) padding 到长度 B（padding 内容不会影响前面 logits，因为 causal）----
        if B > L:
            pad = torch.full((B - L,), self.mask_id, dtype=torch.long)
            seq_pad = torch.cat([seq, pad], dim=0)  # (B,)
        else:
            seq_pad = seq  # (L==B)

        input_ids = seq_pad.unsqueeze(0).to(self.device)  # (1, B)

        # ---- 3) prefill forward：固定一个很大的 block_size ----
        # 注意：你这个模型在别处用过 block_size=...，所以这里沿用同一路径
        out = self.model(input_ids, use_cache=False, block_size=B)
        logits = out.logits[:, :L, :]  # 只取真实长度部分 (1, L, V)

        # ---- 4) shift 对齐：logits_shift[t] 预测 seq[t] ----
        logits_shift = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)  # (1, L, V)
        labels = seq.unsqueeze(0).to(self.device)  # (1, L)

        # ---- 5) 逐 token NLL：用 cross_entropy 一次性算（避免 full softmax）----
        V = logits_shift.size(-1)
        # 对 token 1..L-1 计算 nll；pos=0 没有“前文”不计算
        nll_1_to_end = F.cross_entropy(
            logits_shift[:, 1:, :].reshape(-1, V),
            labels[:, 1:].reshape(-1),
            reduction="none",
        ).view(1, L - 1)  # (1, L-1)

        # 拼成 (L,) 并给 pos0 填 NaN
        nll = torch.empty((L,), device="cpu", dtype=torch.float32)
        nll[0] = float("nan")
        nll[1:] = nll_1_to_end.squeeze(0).detach().cpu()

        p_true = torch.empty((L,), device="cpu", dtype=torch.float32)
        p_true[0] = float("nan")
        p_true[1:] = torch.exp(-nll[1:])  # p_true = exp(-nll)

        # ---- 6) top1 / margin（不需要 softmax）----
        top2 = torch.topk(logits_shift, k=2, dim=-1)  # values/indices: (1, L, 2)
        top1_ids = top2.indices[..., 0].squeeze(0).detach().cpu()  # (L,)
        top1_logits = top2.values[..., 0].squeeze(0).detach().cpu().float()
        top2_logits = top2.values[..., 1].squeeze(0).detach().cpu().float()
        margin_logit = top1_logits - top2_logits  # (L,)

        correct_top1 = (top1_ids == seq)  # (L,) bool on CPU

        top1_prob = None
        if compute_top1_prob:
            # 需要 logsumexp，但输出只有 (L,) 很小
            lse = torch.logsumexp(logits_shift, dim=-1).squeeze(0).detach().cpu().float()  # (L,)
            top1_prob = torch.exp(top1_logits - lse)  # (L,)

        # ---- 7) 组织逐 token 输出 ----
        start_pos = 0 if return_prefix_tokens else prefix_len
        rows: List[Dict[str, Any]] = []
        for t in range(start_pos, L):
            tid = int(seq[t].item())
            row = {
                "pos": t,
                "is_target": bool(t >= prefix_len),
                "token_id": tid,
                "token": self._safe_id_to_token(tid),
                "nll": float(nll[t].item()),
                "p_true": float(p_true[t].item()),
                "top1_id": int(top1_ids[t].item()),
                "top1_token": self._safe_id_to_token(int(top1_ids[t].item())),
                "top1_correct": bool(correct_top1[t].item()),
                "margin_logit": float(margin_logit[t].item()),
            }
            if top1_prob is not None:
                row["top1_prob"] = float(top1_prob[t].item())

            rows.append(row)

        # ---- 8) 汇总：target 的总 loglikelihood / avg nll / ppl ----
        # target token 位置是 [prefix_len, L-1]，其中 token=prefix_len 对应 nll[prefix_len]
        target_nll = nll[prefix_len:]  # CPU float32
        total_target_nll = float(torch.tensor(target_nll).nan_to_num(0.0).sum().item())
        avg_target_nll = float(torch.tensor(target_nll).nan_to_num(0.0).mean().item()) if (
                                                                                                      L - prefix_len) > 0 else float(
            "nan")
        ppl = float(math.exp(avg_target_nll)) if math.isfinite(avg_target_nll) else float("nan")

        return {
            "prefix_len": prefix_len,
            "seq_len": L,
            "block_size_big": B,
            "rows": rows,  # 逐 token 结果
            "summary": {
                "total_loglikelihood_target": -total_target_nll,
                "total_nll_target": total_target_nll,
                "avg_nll_target": avg_target_nll,
                "ppl_target": ppl,
            },
        }
if __name__ == "__main__":
    cli_evaluate()
