# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in eager KV-producer gather; all other workers retain native dispatch."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.platforms import envs as henvs

from ._common import (
    already_applied,
    load_exact_module,
    require_exact_signature,
    require_unpatched,
)

TARGET_MODULE = "vllm._custom_ops"
PATCH_ID = "worker.op_opt.tiled_fp8_gather"
TARGETS = (f"{TARGET_MODULE}.cp_gather_and_upconvert_fp8_kv_cache",)
_MARKER = "_hcu_tiled_fp8_gather_patch_applied"
_WRAPPER_MARKER = "_hcu_tiled_fp8_gather_wrapper"
_enabled = False


def _field(config: object, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(
        config, name, default,
    )


def bind_runtime_config(vllm_config: object | None) -> None:
    """Bind once at the Worker/config boundary, before model imports.

    The custom-op module can be imported before that boundary. Its wrapper
    stays native until bound, without relying on a forward-time global config
    context (which is not guaranteed to be set during model execution).
    """
    global _enabled
    _enabled = False
    if not (henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_TILED_KV_GATHER):
        return
    transfer = _field(vllm_config, "kv_transfer_config")
    model = _field(vllm_config, "model_config")
    if (_field(transfer, "kv_role") != "kv_producer"
            or _field(model, "enforce_eager") is not True):
        return
    from vllm_hcu.platforms.hcu import on_gfx938

    _enabled = on_gfx938()


def make_dispatch(original):
    @functools.wraps(original)
    def dispatch(src_cache, dst, block_table, seq_lens, workspace_starts,
                 batch_size):
        arguments = (src_cache, dst, block_table, seq_lens,
                     workspace_starts, batch_size)
        if (_enabled and henvs.VLLM_HCU_USE_CUSTOM_OPS
                and henvs.VLLM_HCU_USE_TILED_KV_GATHER):
            from vllm_hcu.ops.tiled_fp8_gather import (
                cp_gather_and_upconvert_fp8_kv_cache_tiled,
                supports_tiled_gather,
            )

            if supports_tiled_gather(*arguments):
                return cp_gather_and_upconvert_fp8_kv_cache_tiled(*arguments)
        return original(*arguments)

    return dispatch


def apply_to_module(module: ModuleType) -> bool:
    if not (henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_TILED_KV_GATHER):
        return False
    module = load_exact_module(TARGET_MODULE, module)
    name = "cp_gather_and_upconvert_fp8_kv_cache"
    if already_applied(module, _MARKER,
                       ((module, name, TARGETS[0], _WRAPPER_MARKER),)):
        return False
    original = require_unpatched(module, name, TARGETS[0], _WRAPPER_MARKER)
    require_exact_signature(
        original, TARGETS[0],
        positional=("src_cache", "dst", "block_table", "seq_lens",
                    "workspace_starts", "batch_size"),
    )
    dispatch = make_dispatch(original)
    setattr(dispatch, _WRAPPER_MARKER, True)
    module.cp_gather_and_upconvert_fp8_kv_cache = dispatch
    setattr(module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
