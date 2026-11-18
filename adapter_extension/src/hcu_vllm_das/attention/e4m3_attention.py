"""Read native E4M3 cache bytes through Core gather and vendor BF16/FP16 Flash.

The vendor's paged E4M3 prefill does not cover all supported head sizes and
may quantize Q. Keep Q unchanged and dequantize only the gathered KV workspace.
Allocation uses static metadata bounds; device cumulative lengths remain live
during Graph replay. This adds workspace and gather/conversion overhead, not
a native E4M3 Flash-kernel performance claim.
"""
from collections.abc import Callable
from typing import Any

import torch


def _scale(scale: torch.Tensor | None, query: torch.Tensor, batch: int, heads: int):
    if scale is None:
        return None
    if scale.device != query.device or scale.dtype != torch.float32:
        raise ValueError("E4M3 KV scales must be FP32 on the query device")
    if scale.numel() == 1:
        return scale.reshape(1, 1, 1)
    # vLLM expands the per-layer scalar to [batch, KV heads]. Do not silently
    # treat genuinely different per-sequence scales as a shared layer scale.
    if scale.shape == (batch, heads) and (batch == 1 or scale.stride(0) == 0):
        return scale[0].reshape(1, heads, 1)
    raise ValueError("E4M3 Flash requires shared per-layer KV scales")


def run_e4m3_attention(
    vendor: Callable[..., Any], kwargs: dict[str, Any], layout: str,
) -> Any:
    from vllm._custom_ops import cp_gather_cache
    from vllm.platforms import current_platform

    query, key, value = (kwargs[name] for name in ("q", "k", "v"))
    cuq, lengths = kwargs["cu_seqlens_q"], kwargs["seqused_k"]
    batch = cuq.shape[0] - 1
    max_k = kwargs["max_seqlen_k"]
    fp8_dtype = current_platform.fp8_dtype()
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or kwargs.get("q_descale") is not None
        or layout not in ("LBHNC", "LBNHC")
        or batch < 1 or type(max_k) is not int or max_k < 1
        or key.ndim != 4 or key.shape != value.shape
        or key.dtype != value.dtype or key.dtype not in (torch.uint8, fp8_dtype)
        or fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    ):
        raise ValueError("invalid HCU E4M3 Flash query/cache configuration")
    if layout == "LBHNC":
        key, value = key.transpose(1, 2), value.transpose(1, 2)
    heads, dimension = key.shape[2:]
    scales = [_scale(kwargs.get(name), query, batch, heads)
              for name in ("k_descale", "v_descale")]
    capacity = batch * max_k
    cuk = torch.zeros_like(cuq)
    torch.cumsum(lengths, dim=0, out=cuk[1:])

    def gather(source: torch.Tensor, scale: torch.Tensor | None):
        # Clear the static-capacity tail on every invocation, avoiding arbitrary
        # FP8 NaNs in rounded token tiles and stale values after shorter replays.
        gathered = torch.zeros((capacity, heads, dimension),
                               device=source.device, dtype=torch.uint8)
        # Core's byte gather retains page/token strides, offsets and live
        # block tables. Each head's content plane is contiguous in both layouts.
        source = source.view(torch.uint8)
        for head in range(heads):
            cp_gather_cache(source[:, :, head, :], gathered[:, head, :],
                            kwargs["block_table"], cuk, batch)
        decoded = gathered.view(fp8_dtype).to(query.dtype)
        if scale is not None:
            # E4M3 values are exactly representable in BF16/FP16. TensorIterator
            # computes with the FP32 scale, rounding only the dequantized result.
            torch.mul(decoded, scale, out=decoded)
        return decoded

    arguments = dict(kwargs, k=gather(key, scales[0]), v=gather(value, scales[1]),
                     cu_seqlens_k=cuk, q_descale=None, k_descale=None, v_descale=None,
                     fa_version=2)
    if dimension == 192:
        # The installed vendor's non-paged D192 Graph tile leaves some live
        # output components undefined. Zero-extension to its D256 tile preserves
        # QK, V's original channels and the original attention temperature.
        for name in ("q", "k", "v"):
            arguments[name] = torch.nn.functional.pad(arguments[name], (0, 64))
        arguments["out"] = torch.empty_like(arguments["q"])
        if arguments.get("softmax_scale") is None:
            arguments["softmax_scale"] = dimension ** -.5
    for name in ("block_table", "seqused_k", "layout"):
        arguments.pop(name, None)
    result = vendor(**arguments)
    actual = result[0] if isinstance(result, tuple) else result
    if dimension == 192:
        actual = actual[..., :dimension]
    output = kwargs["out"]
    if actual.data_ptr() != output.data_ptr():
        output.copy_(actual)
    return (output, *result[1:]) if isinstance(result, tuple) else output
