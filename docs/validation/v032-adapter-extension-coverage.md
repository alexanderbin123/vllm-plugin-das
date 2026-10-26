# vLLM 0.32 DAS extension coverage and next gates

Date: 2026-10-26 (UTC)

## Reference boundary

This inventory compares `hcu-vllm-das-plugin` `0.2.0.dev0` with
`HYGON-AI/vllm-plugin-das` branch `v0.28.1-dev` at commit
`cd412a0d53b9b517fbc3dcdf96121dc3daa1096c`. It prevents the two validated
operators from being mistaken for a complete port of every optimization in the
reference repository.

The reference `vllm_hcu.ops` initializer imports five modules. Their vLLM 0.32
status is:

| Reference module | vLLM 0.32 disposition | Production state |
| --- | --- | --- |
| `silu_and_mul` | Ported through Adapter extension API and vLLM OOT `CustomOp` registration | Validated on `gfx938` |
| `rms_norm` | Ported with stricter vLLM 0.32 semantic guards and native fallback | Validated on `gfx938` |
| `rotary_embedding` | Do not port | Upstream vLLM 0.32 already has a ROCm `forward_hip`; the reference wrapper only redirects to `forward_cuda` and would discard newer upstream HIP behavior |
| `gemma_rms_norm` | Separate checkpoint-specific candidate | Not shipped; requires Gemma kernel and model evidence |
| `rms_norm_gated` | Separate Qwen3.5/Qwen3.6 candidate | Not shipped; requires a vLLM 0.32 OOT integration plus hybrid-model evidence |

Other reference operator files are not imported by the global operator
initializer or are consumed by HCU-specific model/runner implementations. They
must not be copied into the independent extension without their consumers:

- fused RMSNorm/Silu quantization belongs with the corresponding quantized
  model and quantization-provider integration;
- MoE gate, FP8, Step5, token quantization, and RMS-RoPE paths require their
  model, layout, and backend contracts;
- top-k/top-p sampling is coupled to the HCU model runner and needs sampling
  distribution/correctness evidence;
- depthwise convolution is tied to hybrid/Mamba execution;
- `test_concat` is test support, not a production capability.

Attention, MoE, quantization, sampler, checkpoint-specific model code, and
topology tuning therefore remain intentionally outside `0.2.0.dev0`.

## Qwen3-8B optimized-versus-native A/B gate

The first model-level check uses the local dense BF16 checkpoint
`/models/Qwen3-8B`. It validates that enabling the two plugin operators does
not change deterministic output relative to the plugin's master rollback. It
is a correctness smoke, not a performance result.

Prerequisites:

1. The exact final wheels and installed attestation from
   `v032-adapter-extension-rmsnorm.md` pass unchanged.
2. The selected device reports `gfx938`, no foreign process, and no more than
   256 MiB used VRAM before either run.
3. Both runs use the same model files, prompt, seed, token limit, vLLM wheel,
   Adapter wheel, Plugin wheel, and device.
4. The native-control process sets `VLLM_HCU_USE_CUSTOM_OPS=0`; the optimized
   process sets the master, SiluAndMul, and RMSNorm switches to `1`.

The prepared probe is the Adapter's isolated
`tools/candidate/probe_model_smoke.py`, invoked with:

```text
--model /models/Qwen3-8B
--prompt "Return only the integer immediately after 41."
--request-id smoke-arithmetic
--max-tokens 32
--seed 0
--max-model-len 512
--gpu-memory-utilization 0.30
```

Acceptance requires both processes to exit 0, select
`hcu_vllm_adapter.platform.hcu`, emit complete schema-v1 records, contain no
fatal runtime diagnostics, and produce identical response text, token IDs, and
finish reason. VRAM must return to the preflight band after each process. A
second pass can then compare latency, but no throughput claim is allowed from
this one-request eager test.

## Current execution state

At 2026-10-26 16:16 UTC, this gate was not started because all eight visible
devices were occupied by an external workload. Each device showed roughly
133 GiB VRAM in use and non-zero compute utilization; the owning processes
were outside this container's process namespace. No process was terminated and
no HCU memory was allocated for the model A/B run.

Status: `blocked-by-device-occupancy`, not pass and not fail. Re-run the
preflight and execute the two bounded processes only after one `gfx938` device
is genuinely idle.
