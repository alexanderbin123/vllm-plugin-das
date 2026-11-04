# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Routing regression for the small-request eager launch overhead crossover."""

import pytest

from vllm_hcu.ops import tiled_fp8_gather
from vllm_hcu.patch.worker.op_opt import patch_tiled_fp8_gather as adapter


@pytest.mark.parametrize("batch_size", [0, 1, 2, 3])
def test_small_batches_keep_native_dispatch(monkeypatch, batch_size):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_TILED_KV_GATHER", "1")
    monkeypatch.setattr(adapter, "_enabled", True)
    # Isolate routing: real tensors/kernels are covered by the GPU validation.
    monkeypatch.setattr(tiled_fp8_gather, "supports_tiled_gather", lambda *args: True)
    monkeypatch.setattr(
        tiled_fp8_gather, "cp_gather_and_upconvert_fp8_kv_cache_tiled",
        lambda *args: "tiled",
    )
    dispatch = adapter.make_dispatch(lambda *args: "native")
    assert dispatch(None, None, None, None, None, batch_size) == "native"


@pytest.mark.parametrize("batch_size", [4, 8, 12])
def test_large_batches_select_tiled_dispatch(monkeypatch, batch_size):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_TILED_KV_GATHER", "1")
    monkeypatch.setattr(adapter, "_enabled", True)
    monkeypatch.setattr(tiled_fp8_gather, "supports_tiled_gather", lambda *args: True)
    monkeypatch.setattr(
        tiled_fp8_gather, "cp_gather_and_upconvert_fp8_kv_cache_tiled",
        lambda *args: "tiled",
    )
    dispatch = adapter.make_dispatch(lambda *args: "native")
    assert dispatch(None, None, None, None, None, batch_size) == "tiled"
