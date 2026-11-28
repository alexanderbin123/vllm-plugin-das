# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""FP8 PA regression on real fused pages, including changed-input Graph replay."""

import pytest
import torch

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is None,
        reason="requires an HCU runtime",
    ),
]


@pytest.mark.parametrize("dim,max_q", [
    (128, 1), (128, 4), (256, 1), (256, 4), (256, 32),
])
@pytest.mark.parametrize("quantized_query", [False, True])
def test_e4m3_paged_strided_pages_and_changed_graph_replay(
    monkeypatch, dim, max_q, quantized_query,
):
    from vllm_hcu.v1.attention import kv_cache_layout
    from vllm_hcu.v1.attention.backends import fa_utils

    monkeypatch.setattr(kv_cache_layout, "_worker_kv_cache_layout", "HND")
    torch.manual_seed(553)
    device = "cuda"
    batch, heads, qheads, page = 3, 4, 24, 64
    storage = (torch.randn(10, 2, heads, page, dim, device=device) * 0.4).to(
        torch.float8_e4m3fn
    )
    key, value = storage.permute(0, 1, 3, 2, 4).unbind(1)
    dtype = torch.float8_e4m3fn if quantized_query else torch.bfloat16
    query = (torch.randn(batch * max_q, qheads, dim * 2, device=device) * 0.3).to(
        dtype
    )[..., ::2]
    output = torch.empty(batch * max_q, qheads, dim * 2, device=device,
                         dtype=torch.bfloat16)[..., ::2]
    table = torch.tensor([[8, 3, 6], [5, 1, 7], [9, 0, 4]],
                         device=device, dtype=torch.int32)
    lengths = torch.tensor([63, 129, 190], device=device, dtype=torch.int32)
    cuq = torch.arange(batch + 1, device=device, dtype=torch.int32) * max_q
    qscale = torch.tensor([0.75 if quantized_query else 1.0], device=device)
    kscale = torch.tensor([0.5], device=device)
    vscale = torch.tensor([0.25], device=device)

    def run():
        return fa_utils.flash_attn_varlen_func(
            q=query, k=key, v=value, out=output, cu_seqlens_q=cuq,
            max_seqlen_q=max_q, max_seqlen_k=190, seqused_k=lengths,
            block_table=table, softmax_scale=dim ** -0.5, causal=True,
            q_descale=qscale if quantized_query else None,
            k_descale=kscale.expand(batch, heads),
            v_descale=vscale.expand(batch, heads),
        )

    def reference(sequence_lengths):
        expected = []
        for b, length in enumerate(sequence_lengths):
            k = key[table[b].long()].float().flatten(0, 1)[:length]
            v = value[table[b].long()].float().flatten(0, 1)[:length]
            k = k.repeat_interleave(qheads // heads, dim=1) * kscale
            v = v.repeat_interleave(qheads // heads, dim=1) * vscale
            q = query[b * max_q:(b + 1) * max_q].float() * qscale
            logits = torch.einsum("qhd,lhd->hql", q, k) * dim ** -0.5
            allowed = torch.arange(length, device=device)[None, :] <= (
                length - max_q + torch.arange(max_q, device=device)
            )[:, None]
            logits.masked_fill_(~allowed[None], float("-inf"))
            expected.append(torch.einsum("hql,lhd->qhd", logits.softmax(-1), v))
        return torch.cat(expected)

    assert run() is output
    torch.testing.assert_close(output.float(), reference([63, 129, 190]),
                               atol=0.003, rtol=0.04)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Neither cached lengths, block IDs nor query values may be frozen by capture.
    table.copy_(table.roll(1, dims=1))
    lengths.copy_(torch.tensor([59, 120, 183], device=device, dtype=torch.int32))
    query.copy_((query.float() + 0.05).to(dtype))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference([59, 120, 183]),
                               atol=0.003, rtol=0.04)

@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("quantized_query", [False, True])
def test_e4m3_shared_mtp_prefix_uses_non_cascade_attention(
    monkeypatch, layout, page, quantized_query,
):
    """8 x Q4 / prefix256 must reach an accurate, graph-safe output-only path."""
    from types import SimpleNamespace

    import numpy as np

    from vllm_hcu.v1.attention import kv_cache_layout
    from vllm_hcu.v1.attention.backends import flash_attn

    monkeypatch.setattr(kv_cache_layout, "_worker_kv_cache_layout", layout)
    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "varlen")
    monkeypatch.setattr(flash_attn, "_SPLIT_PD", False)
    torch.manual_seed(553)
    batch, max_q, heads, qheads, dim = 8, 4, 4, 24, 256
    shape = (heads, page, dim) if layout == "HND" else (page, heads, dim)
    storage = (torch.randn(16, 2, *shape, device="cuda") * 0.4).to(
        torch.float8_e4m3fn
    )
    cache = storage.permute(0, 1, 3, 2, 4) if layout == "HND" else storage
    key, value = cache.unbind(1)
    dtype = torch.float8_e4m3fn if quantized_query else torch.bfloat16
    query = (torch.randn(32, qheads, dim, device="cuda") * 0.3).to(dtype)
    output = torch.empty(32, qheads, 2 * dim, device="cuda",
                         dtype=torch.bfloat16)[..., ::2]
    prefix_ids = [7, 3, 0, 5][:256 // page]
    table = torch.tensor([prefix_ids + [8 + b] for b in range(batch)],
                         device="cuda", dtype=torch.int32)
    lengths = torch.full((batch,), 260, device="cuda", dtype=torch.int32)
    cuq_cpu = torch.arange(batch + 1, dtype=torch.int32) * max_q
    common = SimpleNamespace(
        num_reqs=batch, num_actual_tokens=32, max_query_len=max_q,
        max_seq_len=268, query_start_loc=cuq_cpu.cuda(),
        query_start_loc_cpu=cuq_cpu, seq_lens=lengths,
        block_table_tensor=table, slot_mapping=torch.arange(32, device="cuda"),
        causal=True,
    )
    builder = object.__new__(flash_attn.FlashAttentionMetadataBuilder)
    builder.aot_schedule = False
    builder.aot_sliding_window = (-1, -1)
    builder.cache_config = SimpleNamespace(cache_dtype="fp8_e4m3")
    builder.num_heads_q, builder.num_heads_kv, builder.headdim = qheads, heads, dim
    builder.block_size, builder.dcp_world_size = page, 1
    builder.use_full_cuda_graph, builder.max_cudagraph_size = False, None
    builder.kv_cache_spec = SimpleNamespace(sliding_window=None)
    selection = dict(
        common_prefix_len=256, query_lens=np.full(batch, max_q),
        num_query_heads=qheads, num_kv_heads=heads,
        use_alibi=False, use_sliding_window=False, use_local_attention=False,
        num_sms=120, dcp_world_size=1,
    )
    assert flash_attn.use_cascade_attention(**selection)
    assert not builder.use_cascade_attention(**selection)
    metadata = builder.build(256, common)
    assert not metadata.use_cascade
    assert metadata.common_prefix_len == 0
    impl = flash_attn.FlashAttentionImpl(
        num_heads=qheads, head_size=dim, scale=dim ** -0.5, num_kv_heads=heads,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype="fp8_e4m3",
    )
    assert not impl.can_return_lse_for_decode
    impl.supports_quant_query_input = quantized_query
    layer = SimpleNamespace(
        _q_scale=torch.tensor([0.75 if quantized_query else 1.0], device="cuda"),
        _k_scale=torch.tensor([0.5], device="cuda"),
        _v_scale=torch.tensor([0.25], device="cuda"),
    )
    unused_kv = torch.empty(32, heads, dim, device="cuda", dtype=torch.bfloat16)

    def run():
        return impl.forward(layer, query, unused_kv, unused_kv, cache, metadata, output)

    def reference(length):
        expected = []
        for b in range(batch):
            k = key[table[b].long()].float().flatten(0, 1)[:length]
            v = value[table[b].long()].float().flatten(0, 1)[:length]
            k = k.repeat_interleave(qheads // heads, dim=1) * layer._k_scale
            v = v.repeat_interleave(qheads // heads, dim=1) * layer._v_scale
            q = query[b * max_q:(b + 1) * max_q].float() * layer._q_scale
            logits = torch.einsum("qhd,lhd->hql", q, k) * dim ** -0.5
            allowed = torch.arange(length, device="cuda")[None, :] <= (
                length - max_q + torch.arange(max_q, device="cuda")
            )[:, None]
            logits.masked_fill_(~allowed[None], float("-inf"))
            expected.append(torch.einsum("hql,lhd->qhd", logits.softmax(-1), v))
        return torch.cat(expected)

    assert run() is output
    torch.testing.assert_close(output.float(), reference(260), atol=0.003, rtol=0.04)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    table[:, :-1].copy_(table[:, :-1].roll(1, dims=1))
    table[:, -1].copy_(table[:, -1].roll(1))
    lengths.fill_(264)
    query.copy_((query.float() + 0.05).to(dtype))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference(264), atol=0.003, rtol=0.04)
