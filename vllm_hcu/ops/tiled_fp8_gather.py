# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Experimental request-tiled FP8 KV gather using exact device metadata.

The dequantization loop is adapted from SGLang's paged tiled kernel; see
THIRD_PARTY_NOTICES.md. Unlike its flattened-slot interface, this implementation
accepts the native vLLM block table and reuses the caller's output workspace.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cp_gather_upconvert_tiled_kernel(
    q_ptr, scale_ptr, rope_ptr, out_ptr, table_ptr, lens_ptr, starts_ptr,
    TABLE_STRIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Request ownership is explicit in the grid, rather than binary-searched
    # for each output token. The rectangular grid uses a host capacity bound;
    # exact device lengths determine which programs/rows actually do work.
    req = tl.program_id(1)
    tile_start = tl.program_id(0) * BLOCK_T
    length = tl.load(lens_ptr + req)
    if tile_start >= length:
        return
    token = tile_start + tl.arange(0, BLOCK_T)
    start = tl.load(starts_ptr + req).to(tl.int64)
    dst_row = start + token.to(tl.int64)
    valid = (token < length) & (dst_row >= 0) & (dst_row < CAPACITY)
    page = tl.load(
        table_ptr + req * TABLE_STRIDE + token // BLOCK_SIZE,
        mask=valid, other=0,
    ).to(tl.int64)
    source = page * BLOCK_SIZE + token % BLOCK_SIZE
    element = tl.arange(0, 128)
    for group in tl.static_range(4):
        q = tl.load(
            q_ptr + source[:, None] * 656 + group * 128 + element[None, :],
            mask=valid[:, None], other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            scale_ptr + source * 164 + group, mask=valid, other=0.0,
        )
        tl.store(
            out_ptr + dst_row[:, None] * 576 + group * 128 + element[None, :],
            (q * scale[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid[:, None],
        )
    rope_element = tl.arange(0, 64)
    rope = tl.load(
        rope_ptr + source[:, None] * 328 + rope_element[None, :],
        mask=valid[:, None], other=0.0,
    )
    tl.store(
        out_ptr + dst_row[:, None] * 576 + 512 + rope_element[None, :],
        rope, mask=valid[:, None],
    )


def supports_tiled_gather(
    src_cache, dst, block_table, seq_lens, workspace_starts, num_reqs,
) -> bool:
    """Inspect host-visible layout metadata only; never read device lengths."""
    tensors = (src_cache, dst, block_table, seq_lens, workspace_starts)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    if type(num_reqs) is not int or not 0 <= num_reqs <= 65535:
        return False
    if not all(t.device == src_cache.device and t.is_cuda for t in tensors):
        return False
    if src_cache.ndim not in (3, 4) or src_cache.shape[-1] != 656:
        return False
    if src_cache.ndim == 4 and src_cache.shape[2] != 1:
        return False
    if src_cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return False
    if not src_cache.is_contiguous() or src_cache.shape[1] <= 0:
        return False
    # Reinterpret the scale bytes as float32 without an allocation. Reject
    # misaligned storage views before constructing those typed views.
    if src_cache.data_ptr() % 16:
        return False
    if dst.ndim != 2 or dst.shape[1] != 576:
        return False
    if dst.dtype != torch.bfloat16 or not dst.is_contiguous():
        return False
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        return False
    if block_table.stride(1) != 1 or block_table.shape[0] < num_reqs:
        return False
    return all(
        t.ndim == 1 and t.dtype == torch.int32
        and t.is_contiguous() and t.numel() >= num_reqs
        for t in (seq_lens, workspace_starts)
    )


def cp_gather_and_upconvert_fp8_kv_cache_tiled(
    src_cache: torch.Tensor, dst: torch.Tensor, block_table: torch.Tensor,
    seq_lens: torch.Tensor, workspace_starts: torch.Tensor, num_reqs: int,
) -> None:
    """Write valid request rows; retain the native valid-metadata contract.

    The caller supplies non-overlapping output ranges and valid physical block
    IDs, as for the native op. No CPU speculative upper bound is used to derive
    an exact offset. Unused workspace rows remain untouched.
    """
    if num_reqs == 0 or dst.shape[0] == 0 or block_table.shape[1] == 0:
        return
    flat = src_cache.view(torch.float8_e4m3fn).view(-1, 656)
    scale = flat[:, 512:528].view(torch.float32)
    rope = flat[:, 528:].view(torch.bfloat16)
    max_length = min(block_table.shape[1] * src_cache.shape[1], dst.shape[0])
    _cp_gather_upconvert_tiled_kernel[(triton.cdiv(max_length, 16), num_reqs)](
        flat, scale, rope, dst, block_table, seq_lens, workspace_starts,
        TABLE_STRIDE=block_table.stride(0), BLOCK_SIZE=src_cache.shape[1],
        CAPACITY=dst.shape[0], BLOCK_T=16, num_warps=4,
    )
