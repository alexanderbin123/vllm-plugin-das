# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP correctness tests for HYV4 index conversion; no serving gain claims."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("Requires a HIP GPU", allow_module_level=True)

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index as original,
)
from vllm_hcu.models.hy_v4 import sparse_index


@pytest.fixture
def stream():
    selected = torch.cuda.Stream()
    with torch.cuda.stream(selected):
        yield selected
        selected.synchronize()


def inputs(rows, width, strided):
    stride = 2 if strided else 1
    req = torch.randint(
        0, 3, (rows * stride,), device="cuda", dtype=torch.int32
    )[::stride]
    table = torch.randint(
        0, 128, (3, 16 * stride), device="cuda", dtype=torch.int32
    )[:, ::stride]
    indices = torch.randint(
        -8, 1200, (rows, width * stride), device="cuda", dtype=torch.int32
    )[:, ::stride]
    table[:, -1] = -1  # Negative physical blocks retain upstream semantics.
    return req, table, indices


def reference(req, table, indices, prefill):
    return original(
        req, table, indices, NUM_TOPK_TOKENS=indices.shape[1],
        HAS_PREFILL_WORKSPACE=prefill is not None,
        prefill_workspace_request_ids=None if prefill is None else prefill.request_ids,
        prefill_workspace_starts=None if prefill is None else prefill.workspace_starts,
        return_valid_counts=True,
    )


@pytest.mark.parametrize("rows", [1, 4, 32])
@pytest.mark.parametrize("width", [1024, 2048])
@pytest.mark.parametrize("has_prefill", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_mapping_matches_upstream_and_torch_oracle(
    stream, rows, width, has_prefill, strided
):
    req, table, indices = inputs(rows, width, strided)
    prefill = None
    if has_prefill:
        ids = torch.arange(rows, dtype=torch.int32, device="cuda") % 2
        ids[::2] = -1
        prefill = SimpleNamespace(
            request_ids=ids,
            workspace_starts=torch.tensor([17, 4096], dtype=torch.int32, device="cuda"),
        )
    for pattern in ["negative", "zero", "dynamic"]:
        if pattern == "negative":
            indices.fill_(-1)
        elif pattern == "zero":
            indices.fill_(0)
        else:
            indices.random_(-8, 1200)
        actual = sparse_index.convert_rows(req, table, indices, 64, prefill)
        expected = reference(req, table, indices, prefill)
        block = indices // 64
        valid = (indices >= 0) & (block >= 0) & (block < table.shape[1])
        physical = table[
            req.long()[:, None], block.clamp(0, table.shape[1] - 1).long()
        ]
        oracle = physical * 64 + indices % 64
        if prefill is not None:
            starts = prefill.workspace_starts[
                prefill.request_ids.clamp_min(0).long()
            ]
            workspace = starts[:, None] + indices
            oracle = torch.where((prefill.request_ids >= 0)[:, None], workspace, oracle)
        oracle = torch.where(valid, oracle, -1)
        assert torch.equal(actual[0], expected[0]) and torch.equal(actual[0], oracle)
        assert torch.equal(actual[1], expected[1])
        assert torch.equal(actual[1], valid.sum(-1).int())


def test_actual_capture_never_reuses_an_eager_entry(stream, monkeypatch):
    req, table, indices = inputs(4, 2048, False)
    metadata = SimpleNamespace(
        req_id_per_token=req, block_table=table, block_size=64,
        fp8_extra_metadata=SimpleNamespace(prefill=None),
    )
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(sparse_index, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(sparse_index, "get_forward_context", lambda: context)

    def convert(producer):
        return sparse_index.convert_indices_and_counts(
            metadata, indices, is_producer=producer, mapping_is_read_only=True,
        )

    warm = convert(True)
    assert convert(False) is warm
    # Also compiles the specialization before entering actual graph capture.
    convert(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        first, second = convert(True), convert(False)
    assert first is not second
    for _ in range(3):
        indices.random_(-8, 1200)
        table.random_(0, 128)
        req.random_(0, 3)
        graph.replay()
        expected = reference(req, table, indices, None)
        for result in [first, second]:
            assert all(torch.equal(a, e) for a, e in zip(result, expected))
    indices.fill_(-1)
    eager = convert(False)
    assert eager is not first and eager is not second and eager is not warm
    assert torch.equal(eager[0], torch.full_like(indices, -1))
    assert torch.equal(eager[1], torch.zeros_like(eager[1]))
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    assert convert(True) is not convert(False)


def test_native_sink_attention_preserves_mapping_inputs(stream):
    pytest.importorskip("flash_mla")
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl
    from vllm_hcu.v1.attention.ops.flashmla import get_mla_metadata
    import vllm_hcu.platforms.envs as henvs

    if henvs.VLLM_HCU_HYV4_FP8_KV_DEQUANT:
        pytest.skip("Native FP8 fixture requires the direct FP8 path")
    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.num_heads = impl.prefill_padding = impl.fp8_decode_padded_heads = 64
    impl.kv_lora_rank, impl.head_size = 512, 576
    impl.softmax_scale, impl._dcp_sinks = 576 ** -0.5, None
    impl.sinks = torch.zeros(64, dtype=torch.float32, device="cuda")
    query = torch.randn((2, 64, 576), dtype=torch.bfloat16, device="cuda")
    indices = torch.full((2, 2048), -1, dtype=torch.int32, device="cuda")
    indices[:, :512] = torch.randperm(512, dtype=torch.int32, device="cuda")
    counts = torch.full((2,), 512, dtype=torch.int32, device="cuda")
    saved_indices, saved_counts = indices.clone(), counts.clone()
    bf16_cache = torch.zeros((512, 576), dtype=torch.bfloat16, device="cuda")
    fp8_cache = torch.zeros((8, 64, 656), dtype=torch.uint8, device="cuda")
    fp8_cache[..., 512:528].view(torch.float32).fill_(1.0)
    scheduler, _ = get_mla_metadata()
    kernel_metadata = SimpleNamespace(
        scheduler_metadata=scheduler,
        dummy_block_table=torch.zeros((2, 1), dtype=torch.int32, device="cuda"),
        cache_lens=torch.full((2,), 512, dtype=torch.int32, device="cuda"),
    )
    impl._bf16_flash_mla_kernel(query, bf16_cache, indices, counts)
    impl._fp8_flash_mla_kernel(
        query.unsqueeze(1), fp8_cache, indices.unsqueeze(1), kernel_metadata
    )
    stream.synchronize()
    assert torch.equal(indices, saved_indices) and torch.equal(counts, saved_counts)
