# vLLM 0.32 + adapter-das multi-model service smoke

Validation date: 2026-10-28

## Fixed baseline

- vLLM HCU branch: `OpenDAS/vllm:v0.32.0@7c13676e497262b71a8b107a3186b0f6641dcbf1`
- adapter-das branch: `feat/vllm032-runtime-integration@81361880e203cb33cf631f77716ac8244f5f299f`
- DAS Plugin merged base: `compat/vllm-v0.32.0@cb7851f5b7222892f9188e8f1ceb8be0bacfa6f8`
- installed versions: vLLM `0.32.0+rocm723`, `sgl-kernel 0.1.0`, and
  `hcu-vllm-das-plugin 0.2.0.dev0`
- hardware: one `gfx936` card for each service, with all other cards idle

The vLLM build tree was content-identical to the fixed OpenDAS branch. The
adapter wheel's runtime source was unchanged between its build commit and the
fixed adapter head; only documentation changed. The plugin wheel's runtime
source was unchanged between its build commit and the merged base; only the
installed verifier changed.

## Common service policy

Each model was started separately with the adapter as the only vLLM
platform/general plugin owner and the DAS extension enabled. The commands used
TRITON_ATTN, LBNHC KV cache, one visible card, one maximum sequence, and a
512-token smoke context. No command passed `--enforce-eager`.

For every model, EngineCore reported:

- `enforce_eager=False`;
- compilation mode `VLLM_COMPILE`;
- CUDA Graph mode `FULL_AND_PIECEWISE`;
- successful PIECEWISE and FULL Graph capture.

The common environment was:

```bash
PYTHONPATH=/dev/shm/adapter-vllm032-contract-20261028/lib/python3.12/site-packages
VLLM_PLUGINS=hcu_adapter
HCU_ADAPTER_HOST=vllm
VLLM_HCU_USE_CUSTOM_OPS=1
VLLM_HCU_USE_CUSTOM_RMS_NORM=1
VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
HIP_VISIBLE_DEVICES=0
HSA_VISIBLE_DEVICES=0
```

The service arguments were equivalent to:

```bash
python -m vllm.entrypoints.cli.main serve MODEL_PATH \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 512 \
  --served-model-name SERVED_NAME \
  --attention-backend TRITON_ATTN \
  --disable-custom-all-reduce \
  --gpu-memory-utilization GPU_FRACTION \
  --enable-prefix-caching --max-num-seqs 1
```

Local requests used `--noproxy '*'` because the shared node HTTP proxy
otherwise intercepts localhost.

## Results

| Model | Path covered | GPU fraction | Online result | Deterministic response | Fingerprint |
| --- | --- | ---: | --- | --- | --- |
| Qwen2-0.5B-Instruct | Qwen2 BF16 | 0.25 | health/models/chat HTTP 200 | `42`, stop | `vllm-0.32.0-a06f98bc` |
| Qwen3-4B-Channel-INT8-w8a8 | compressed-tensors Channel INT8 W8A8 | 0.25 | health/chat HTTP 200 | `42`, stop | `vllm-0.32.0-fcd62070` |
| Qwen3-VL-2B-Instruct | Qwen3-VL text and image paths | 0.30 | health and two chat requests HTTP 200 | text: `42`; red-square image: `红色`; both stop | `vllm-0.32.0-02dcd749` |
| Qwen3-14B | larger Qwen3 BF16 | 0.60 | health/version/chat HTTP 200 | `42`, stop | `vllm-0.32.0-0b6812e3` |

The Channel INT8 model was automatically resolved to
`CompressedTensorsW8A8Int8` and selected the upstream
`TritonInt8ScaledMMLinearKernel`; no duplicate plugin implementation was used.

Qwen3-VL initialized its multimodal processor, used Flash Attention for the
vision encoder and TRITON_ATTN for the language backbone, warmed its vision
rotary and M-RoPE Triton kernels, and answered a real inline PNG request. The
test image contained a red square and the model returned `红色`.

Qwen3-14B loaded 27.51 GiB of weights, compiled and captured Graphs, then
remained ready while idle for about 59 minutes before its successful request.
This is limited idle-readiness evidence, not a formal soak test.

## Cleanup and scope

Each service was stopped before the next model. After the final teardown no
vLLM or EngineCore process remained, all eight cards reported 0% use, and each
card returned to the 2 MiB VRAM baseline.

These are startup and deterministic smoke checks. They do not replace
enabled/disabled numerical parity, HumanEval-64, longer-context, concurrency,
topology, throughput, or production soak gates.
