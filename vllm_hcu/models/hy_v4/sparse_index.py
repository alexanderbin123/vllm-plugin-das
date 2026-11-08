# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer sparse-index conversion for HYV4's separate FP8 attention path."""

import torch
import triton
import triton.language as tl

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm_hcu.models.hy_v4.sparse_index_reuse import reuse_or_compute


@triton.jit
def _convert_rows(
    req_ptr, bt_ptr, tok_ptr, out_ptr, count_ptr, prefill_ptr, starts_ptr,
    MAX_BLOCKS: tl.constexpr, BLOCK: tl.constexpr, N: tl.constexpr,
    PREFILL: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, N)
    req = tl.load(req_ptr + row)
    tok = tl.load(tok_ptr + row * N + col)
    block, offset = tok // BLOCK, tok % BLOCK
    valid = (tok >= 0) & (block >= 0) & (block < MAX_BLOCKS)
    is_prefill = False
    if PREFILL:
        prefill_req = tl.load(prefill_ptr + row)
        is_prefill = prefill_req >= 0
    base = tl.load(
        bt_ptr + req * MAX_BLOCKS + block,
        mask=valid & ~is_prefill, other=0,
    )
    value = base * BLOCK + offset
    if PREFILL:
        start = tl.load(starts_ptr + prefill_req, mask=is_prefill, other=0)
        value = tl.where(is_prefill, start + tok, value)
    tl.store(out_ptr + row * N + col, tl.where(valid, value, -1))
    # One CTA owns a complete row: neither count initialization nor atomics.
    tl.store(count_ptr + row, tl.sum(valid.to(tl.int32), 0))


def convert_rows(req_id, block_table, indices, block_size, prefill=None):
    """Preserve the upstream non-compact mapping/count contract exactly.

    Negative physical block IDs do not change the logical valid count. Prefill
    indices still obey logical block bounds before workspace-offset mapping.
    DCP compaction has a different contract and must not call this helper.
    """
    kwargs = dict(
        BLOCK_SIZE=block_size, NUM_TOPK_TOKENS=indices.shape[1],
        HAS_PREFILL_WORKSPACE=prefill is not None,
        prefill_workspace_request_ids=None if prefill is None else prefill.request_ids,
        prefill_workspace_starts=None if prefill is None else prefill.workspace_starts,
        return_valid_counts=True,
    )
    if (indices.shape[1] != 2048 or req_id.dtype != torch.int32
            or block_table.dtype != torch.int32 or indices.dtype != torch.int32):
        return triton_convert_req_index_to_global_index(req_id, block_table, indices, **kwargs)

    if prefill is not None:
        assert prefill.request_ids.dtype == torch.int32
        assert prefill.workspace_starts.dtype == torch.int32
        assert prefill.request_ids.is_contiguous()
        assert prefill.workspace_starts.is_contiguous()
    req_id, block_table, indices = (
        req_id.contiguous(), block_table.contiguous(), indices.contiguous()
    )
    out = torch.empty_like(indices)
    counts = torch.empty(req_id.shape[0], dtype=torch.int32, device=indices.device)
    if req_id.shape[0]:
        _convert_rows[(req_id.shape[0],)](
            req_id, block_table, indices, out, counts,
            kwargs["prefill_workspace_request_ids"], kwargs["prefill_workspace_starts"],
            block_table.shape[1], block_size, indices.shape[1], prefill is not None,
            num_warps=8,
        )
    return out, counts


def convert_indices_and_counts(metadata, indices, *, is_producer, mapping_is_read_only):
    fp8 = metadata.fp8_extra_metadata
    context = get_forward_context() if is_forward_context_available() else None
    # FULL capture can be entered with runtime mode NONE during warmup, so the
    # stream capture check is independently required. Captured conversions stay
    # per-layer nodes; this cache never supplies hidden cross-segment inputs.
    allow_reuse = (
        mapping_is_read_only
        and context is not None
        and getattr(context, "ubatch_slices", None) is None
        and not isinstance(getattr(context, "attn_metadata", None), list)
        and context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        and not torch.cuda.is_current_stream_capturing()
    )
    return reuse_or_compute(
        context, metadata, indices, is_producer=is_producer,
        allow_reuse=allow_reuse,
        compute=lambda: convert_rows(
            metadata.req_id_per_token, metadata.block_table, indices,
            metadata.block_size, fp8.prefill,
        ),
    )
