"""Actual native E4M3 writer -> Core gather -> vendor Flash, eager and Graph."""
import os
from types import SimpleNamespace

import pytest
import torch

from test_flash_attention_hcu import _make_impl, _make_packed_cache, _reference_attention

pytestmark = [pytest.mark.hcu, pytest.mark.skipif(
    os.getenv("VLLM033_HCU_TEST") != "1",
    reason="set VLLM033_HCU_TEST=1 on a prepared HCU node",
)]


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("dimension", (64, 128, 192, 256))
@pytest.mark.parametrize("cache_dtype", ("fp8", "fp8_e4m3"))
@pytest.mark.parametrize("query_lengths", ((1, 0, 17), (1, 0, 1)))
@torch.inference_mode()
def test_native_e4m3_bytes_scaled_attention_and_graph(layout, dtype, dimension, cache_dtype, query_lengths):
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm.platforms import current_platform

    torch.manual_seed(9117)
    fp8 = current_platform.fp8_dtype()
    assert fp8 in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    heads, kv_heads, page, blocks = 4, 2, 64, 12
    keys = torch.randn(3 * 96 + 1, kv_heads, dimension, device='cuda', dtype=dtype)
    values = torch.randn_like(keys)
    query = torch.randn(sum(query_lengths), heads, dimension, device='cuda', dtype=dtype)
    packed = _make_packed_cache(layout_name=layout, num_blocks=blocks, num_heads=kv_heads,
        block_size=page, head_size=dimension, device=query.device, dtype=torch.uint8,
        block_padding=64)
    impl = _make_impl(layout, dimension)
    impl.kv_cache_dtype = cache_dtype
    impl.num_heads, impl.num_kv_heads = heads, kv_heads
    impl.scale = dimension ** -.5
    impl.alibi_slopes = None
    impl.logits_soft_cap = 0.0
    write_k, write_v = impl._get_kv_cache_write_views(packed)
    slots = torch.empty(len(keys), device='cuda', dtype=torch.int64)
    table = torch.tensor([[8, 3], [4, 9], [1, 6]], device='cuda', dtype=torch.int32)
    initial_table = table.clone()
    lengths = torch.tensor([13, 0, 77], device='cuda', dtype=torch.int32)
    cuq = torch.tensor([0, query_lengths[0], query_lengths[0], sum(query_lengths)], device='cuda', dtype=torch.int32)
    ks, vs = torch.tensor(.25, device='cuda'), torch.tensor(1.5, device='cuda')
    layer = SimpleNamespace(_k_scale=ks, _v_scale=vs)
    backing = torch.full((sum(query_lengths)+2, heads, dimension), 99, device='cuda', dtype=dtype)
    output = backing[1:-1]
    metadata = SimpleNamespace(num_actual_tokens=sum(query_lengths), causal=True,
        mm_prefix_query_range_tensor=None, rswa_prefix_lens=None, dcp_context_kv_lens=None,
        query_start_loc=cuq, seq_lens=lengths, max_query_len=max(query_lengths), max_seq_len=96,
        block_table=table)
    all_bytes = packed.as_strided((packed.untyped_storage().nbytes(),), (1,), storage_offset=0)
    limits = torch.finfo(fp8)

    def produce(wave):
        table.copy_(initial_table.roll(wave % 2, dims=1))
        lengths.copy_(torch.tensor(([13, 0, 77], [9, 0, 65])[wave % 2], device='cuda', dtype=torch.int32))
        if wave:
            keys.normal_(); values.normal_(); query.normal_()
            ks.fill_(.25 if wave % 2 == 0 else .125)
            vs.fill_(1.5 if wave % 2 == 0 else .75)
        positions = torch.arange(96, device='cuda')
        slots[:-1].copy_((table[:, positions // page].long() * page + positions % page).flatten())
        slots[-1] = -1  # padding must not write any cache byte
        all_bytes.fill_(0xA5)

    def run():
        impl.do_kv_cache_update(layer, keys, values, packed, slots)
        assert impl.forward(layer, query, keys, values, packed, metadata, output) is output

    def check():
        ek = torch.clamp(keys.float() / ks, limits.min, limits.max).to(fp8)
        ev = torch.clamp(values.float() / vs, limits.min, limits.max).to(fp8)
        expected_bytes = torch.full_like(all_bytes, 0xA5)
        expected = torch.as_strided(expected_bytes, packed.shape, packed.stride())
        ekc, evc = impl._get_kv_cache_write_views(expected)
        block_ids, positions = slots[:-1] // page, slots[:-1] % page
        ekc[block_ids, positions] = ek[:-1].view(torch.uint8)
        evc[block_ids, positions] = ev[:-1].view(torch.uint8)
        assert torch.equal(all_bytes, expected_bytes)
        decoded_k = (ek.to(dtype).float() * ks).to(dtype)
        decoded_v = (ev.to(dtype).float() * vs).to(dtype)
        expected_outputs = []
        at = 0
        for b, nq in enumerate(query_lengths):
            if nq:
                nk = int(lengths[b])
                expected_outputs.append(_reference_attention(query[at:at+nq],
                    decoded_k[b*96:b*96+nk], decoded_v[b*96:b*96+nk]))
                at += nq
        reference = torch.cat(expected_outputs)
        torch.testing.assert_close(output, reference, atol=.016, rtol=.016)
        assert torch.all(backing[[0, -1]] == 99)
        return output.clone()

    produce(0)
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    previous = check()
    addresses = [x.data_ptr() for x in (query, keys, values, packed, slots, table, lengths, cuq, ks, vs, output)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for wave in (1, 2, 3):
        produce(wave)
        graph.replay()
        torch.cuda.synchronize()
        observed = check()
        assert not torch.equal(observed, previous)
        run()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, observed, atol=0, rtol=0)
        assert addresses == [x.data_ptr() for x in (query, keys, values, packed, slots, table, lengths, cuq, ks, vs, output)]
        previous = observed
