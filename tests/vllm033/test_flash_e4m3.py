"""Portable E4M3 admission, writer and public gather/Flash composition contracts."""
from types import ModuleType, SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from test_flash_attention import load_flash_attention, make_impl


@pytest.mark.parametrize("cache_dtype", ("fp8", "fp8_e4m3"))
def test_e4m3_admitted_and_written_by_core(monkeypatch, cache_dtype):
    flash = load_flash_attention(monkeypatch)
    assert flash.HcuFlashAttentionBackend.supports_kv_cache_dtype(cache_dtype)
    impl = make_impl(flash, kv_cache_dtype=cache_dtype)
    calls = []
    native = ModuleType("vllm._custom_ops")
    native.reshape_and_cache_flash = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", native)
    packed = torch.empty(2, 2, 64, 128, dtype=torch.uint8)
    key = torch.randn(3, 2, 64)
    slots = torch.tensor([0, 1, 2])
    scale = torch.tensor(.5)
    from types import SimpleNamespace
    layer = SimpleNamespace(_k_scale=scale, _v_scale=scale)
    impl.do_kv_cache_update(layer, key, key, packed, slots)
    assert calls[0][0] is key
    assert calls[0][4] is slots
    assert calls[0][5] == cache_dtype
    assert calls[0][6] is scale and calls[0][7] is scale


@pytest.mark.parametrize("cache_dtype", ("fp8", "fp8_e4m3"))
def test_e4m3_routes_to_its_own_reader(monkeypatch, cache_dtype):
    flash = load_flash_attention(monkeypatch)
    impl = make_impl(flash, kv_cache_dtype=cache_dtype)
    calls = []
    helper = ModuleType("hcu_vllm_das.attention.e4m3_attention")
    sentinel = object()
    helper.run_e4m3_attention = lambda vendor, args, layout: calls.append((vendor, args, layout)) or sentinel
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    vendor = lambda **kwargs: pytest.fail("raw E4M3 must not bypass the reader")
    monkeypatch.setattr(flash, "_vendor_varlen_func", lambda: vendor)
    k = torch.empty(2, 2, 64, 64, dtype=torch.uint8)
    assert impl._run_flash_attention(k=k, v=k) is sentinel
    assert calls == [(vendor, {"k": k, "v": k}, "LBHNC")]


def load_reader(monkeypatch, fp8_dtype):
    path = Path(__file__).resolve().parents[2] / "adapter_extension/src/hcu_vllm_das/attention/e4m3_attention.py"
    spec = importlib.util.spec_from_file_location("_e4m3_reader_contract", path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    native = ModuleType("vllm._custom_ops")
    def gather(source, destination, table, cumulative, batch):
        assert source.dtype == destination.dtype == torch.uint8
        at = 0
        for b in range(batch):
            for token in range(int(cumulative[b + 1] - cumulative[b])):
                destination[at].copy_(source[table[b, token // source.shape[1]], token % source.shape[1]])
                at += 1
    native.cp_gather_cache = gather
    platform = ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(fp8_dtype=lambda: fp8_dtype)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", native)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform)
    return reader


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("fp8_dtype", (torch.float8_e4m3fn, torch.float8_e4m3fnuz))
@pytest.mark.parametrize("returns_lse", (False, True))
@pytest.mark.parametrize("dimension", (64, 192))
def test_reader_preserves_e4m3_bytes_scales_query_and_output(monkeypatch, layout, fp8_dtype, returns_lse, dimension):
    reader = load_reader(monkeypatch, fp8_dtype)
    generator = torch.Generator().manual_seed(44)
    k = torch.randn(4, 4, 2, dimension, generator=generator).to(fp8_dtype).view(torch.uint8)
    v = torch.randn(4, 4, 2, dimension, generator=generator).to(fp8_dtype).view(torch.uint8)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    lengths = torch.tensor([3, 5], dtype=torch.int32)
    q = torch.randn(3, 4, dimension, generator=generator, dtype=torch.bfloat16)
    before = q.clone()
    out = torch.empty_like(q)
    ks, vs = torch.tensor(.25), torch.tensor(1.5)
    expected_k = torch.cat((k[2, :3], k[3], k[1, :1])).view(fp8_dtype).float() * ks
    expected_v = torch.cat((v[2, :3], v[3], v[1, :1])).view(fp8_dtype).float() * vs
    lse = object()
    def vendor(**args):
        if dimension == 192:
            torch.testing.assert_close(args['q'][..., :dimension], q, rtol=0, atol=0)
            assert args['q'].shape[-1] == 256 and torch.all(args['q'][..., dimension:] == 0)
            assert args['softmax_scale'] == dimension ** -.5
            assert torch.all(args['k'][..., dimension:] == 0) and torch.all(args['v'][..., dimension:] == 0)
        else:
            assert args['q'] is q
        assert not any(name in args for name in ('block_table', 'seqused_k', 'layout'))
        assert args['k_descale'] is args['v_descale'] is args['q_descale'] is None
        torch.testing.assert_close(args['cu_seqlens_k'], torch.tensor([0, 3, 8], dtype=torch.int32))
        torch.testing.assert_close(args['k'][:8, :, :dimension], expected_k.to(q.dtype), rtol=0, atol=0)
        torch.testing.assert_close(args['v'][:8, :, :dimension], expected_v.to(q.dtype), rtol=0, atol=0)
        assert torch.all(args['k'][8:] == 0) and torch.all(args['v'][8:] == 0)
        # The real non-paged vendor is allowed to allocate a different output.
        result = torch.full_like(args['q'], 7)
        return (result, lse) if returns_lse else result
    if layout == 'LBHNC':
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    result = reader.run_e4m3_attention(vendor, dict(q=q, k=k, v=v, out=out,
        cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32), seqused_k=lengths,
        max_seqlen_k=5, block_table=table, k_descale=ks.expand(2, 2),
        v_descale=vs.expand(2, 2)), layout)
    assert (result[0] if returns_lse else result) is out
    if returns_lse:
        assert result[1] is lse
    assert torch.all(out == 7)
    assert torch.equal(q, before)


def test_reader_rejects_real_per_sequence_scales(monkeypatch):
    reader = load_reader(monkeypatch, torch.float8_e4m3fn)
    with pytest.raises(ValueError, match='shared per-layer'):
        reader._scale(torch.ones(2, 2), torch.empty(1), 2, 2)
