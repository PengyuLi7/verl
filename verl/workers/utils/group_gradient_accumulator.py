# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Accumulate independently finalized group gradients without re-reducing old gradients."""

import math
from collections.abc import Mapping

import torch


class GroupGradientAccumulator:
    """Weighted sum of group means, with an exact-once barrier before an optimizer step.

    Megatron's schedule finalizes (and sometimes rescales) its gradient buffers on
    every call. Keep the running sum outside those buffers, zero the engine before
    each group, and restore the sum only after the entire synchronous batch arrives.
    Weights are valid response-token counts for token-mean, or valid sequence counts
    for sequence-mean. This uses one additional FP32 gradient buffer.
    """

    def __init__(self, group_ids: list[str]):
        if not group_ids or len(set(group_ids)) != len(group_ids):
            raise ValueError("Expected a nonempty list of unique group IDs")
        self.expected = frozenset(group_ids)
        self.seen: set[str] = set()
        self.sums: dict[str, torch.Tensor] = {}
        self.weight = 0.0
        self.restored = False

    def check_group(self, group_id: str, weight: float) -> None:
        """Reject duplicate, foreign, or invalid groups before running backward."""
        if self.restored or group_id not in self.expected or group_id in self.seen:
            raise ValueError(f"Unexpected or duplicate group: {group_id}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Group weight must be finite and positive, got {weight}")

    @torch.no_grad()
    def add(self, group_id: str, gradients: Mapping[str, torch.Tensor], weight: float) -> None:
        """Copy a group's already synchronized gradients into the running sum."""
        self.check_group(group_id, weight)
        if not gradients:
            raise ValueError("No Megatron gradient buffers were found")
        if self.sums and gradients.keys() != self.sums.keys():
            raise ValueError("Gradient buffer layout changed within a synchronous batch")
        for name, grad in gradients.items():
            if grad.dtype != torch.float32:
                raise ValueError("Group accumulation requires FP32 Megatron gradient buffers")
            if name in self.sums and (self.sums[name].shape != grad.shape or self.sums[name].device != grad.device):
                raise ValueError(f"Gradient buffer changed within a synchronous batch: {name}")
        for name, grad in gradients.items():
            if name not in self.sums:
                self.sums[name] = torch.zeros_like(grad)
            self.sums[name].add_(grad, alpha=weight)
        self.weight += weight
        self.seen.add(group_id)

    @torch.no_grad()
    def restore(self, gradients: Mapping[str, torch.Tensor]) -> None:
        """Restore the global mean exactly once, before clipping and optimizer.step."""
        if self.restored or self.seen != self.expected:
            raise RuntimeError(f"Cannot step: received {len(self.seen)}/{len(self.expected)} groups")
        if gradients.keys() != self.sums.keys():
            raise ValueError("Gradient buffer layout changed before optimizer step")
        for name, grad in gradients.items():
            if grad.shape != self.sums[name].shape or grad.device != self.sums[name].device:
                raise ValueError(f"Gradient buffer changed before optimizer step: {name}")
        for name, grad in gradients.items():
            grad.copy_(self.sums[name]).div_(self.weight)
        self.restored = True
        self.sums.clear()


def megatron_gradient_buffers(modules) -> dict[str, torch.Tensor]:
    """Return native DDP buffers, including expert buffers and distributed-optimizer shards.

    Accumulating the physical grad_data storage preserves reduce-scatter results
    even when a distributed optimizer owns only slices of individual parameters.
    Megatron FSDP / precision-aware optimizer layouts are deliberately unsupported.
    """
    gradients = {}
    seen = set()
    for chunk_index, module in enumerate(modules):
        for kind in ("buffers", "expert_parallel_buffers"):
            for buffer_index, buffer in enumerate(getattr(module, kind, ())):
                grad = buffer.grad_data
                if id(grad) not in seen:
                    gradients[f"{chunk_index}/{kind}/{buffer_index}"] = grad
                    seen.add(id(grad))
    if not gradients:
        raise RuntimeError("group_delivery requires native Megatron DDP gradient buffers")
    return gradients
