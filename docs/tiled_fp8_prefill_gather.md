# Experimental tiled FP8 gather for the P worker

This draft adapts the SGLang paged tiled dequantization loop to vLLM's
`cp_gather_and_upconvert_fp8_kv_cache` interface. Initial gfx938 operator checks
match native/reference values, but **model accuracy and end-to-end performance
remain unvalidated**. Existing deployments keep native unless opted in.

## Activation and scope

Set before starting the P server:

```bash
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_TILED_KV_GATHER=1
```

The initial implementation requires gfx938, `kv_transfer_config.kv_role`
exactly `kv_producer`, and `model_config.enforce_eager=True`. The Worker binds
these conditions before importing model/operator modules. Consumers, `kv_both`,
standalone servers and graph-mode servers continue to use native dispatch,
even if the new environment variable is set. The default is off; the custom-op
master switch takes precedence. Restart workers after changing these settings.

Eligible calls have a contiguous packed 656-byte FP8 E4M3FN cache
(`uint8` or `float8_e4m3fn` storage), a contiguous BF16 `[capacity, 576]`
workspace, and same-device int32 metadata. Other layouts call the saved native
function. This changes neither the D worker nor the indexer's TopK selection.
It gathers all valid KV rows described by the current prefill chunk.
Calls with fewer than four requests retain native dispatch: the initial gfx938
eager measurements showed a single-request regression and gains at 4/8/12
requests. The 2/3-request region is conservatively kept native as well.

## Mapping and expected tradeoffs

Each program handles up to 16 tokens for one request, four 128-element NoPE
groups with FP32 scales, and the 64-element BF16 RoPE tail. Request ownership
comes from the second grid dimension, avoiding the native per-token binary
search. Physical slots still come from vLLM's block table: there is no flattened
slot-ID temporary allocation or preparatory kernel.

Both lengths and output offsets come from the existing **device tensors**.
CPU speculative sequence-length upper bounds never determine physical output
offsets. Like native, the caller must supply valid block IDs and disjoint output
ranges. Padding rows are left untouched. Typed input views reuse storage;
dispatch performs no device-to-host copy, synchronization, or output allocation.

The launch uses a rectangular capacity bound from host-visible shapes. Shorter
requests exit before reading cache rows, but uneven lengths can still waste
program launches. No speedup is claimed, and this is not a direct port of
SGLang's flattened-slot preprocessing.

## Validation still required before enabling by default

- Native-vs-tiled GPU comparison: mixed/zero lengths, non-multiples of 16,
  noncontiguous physical block IDs, repeated prefix pages, nonzero workspace
  starts and sentinel-filled unused rows; compare NoPE and RoPE separately.
- Dispatch/lifecycle coverage: default off, master off, pre-config import,
  producer/consumer/kv_both, eager/graph, unsupported layouts and idempotency.
- Async speculative rejection and mixed prefill/decode metadata with CPU upper
  bounds greater than exact device lengths.
- Paired HumanEval accuracy checks, then P-only and P/D benchmarks at the same
  workload, including prefix-cache hits, workspace splitting, TTFT, RPS and
  complete indexing/gather overhead. Capture traces only in a separate run.

Initial operator validation covers native/reference bit patterns, tail tiles,
zero/mixed lengths, output gaps, 3D/4D cache layouts, changed device metadata,
and dispatch/lifecycle behavior. NoPE and RoPE (including special BF16 bit
patterns) match. Preliminary eager measurements at 85000-token context show
about 1.32x/1.40x/1.46x at 4/8/12 requests, while one request regresses without
the small-batch fallback. These are operator-level results, not P/D capacity or
model-accuracy results. Unit timing scripts and full results accompany the
benchmark artifacts; the small-batch routing regression is checked in.
