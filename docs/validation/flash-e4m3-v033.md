# HCU Flash E4M3 KV migration (v0.33.0)

This change accepts `fp8` and `fp8_e4m3` through the existing public
`FLASH_ATTN` registration. Core owns cache allocation, its native E4M3 writer,
block tables and byte gather. The Plugin dequantizes gathered KV with the
platform FP8 encoding and per-layer scales, then calls the vendor BF16/FP16
varlen Flash API. Queries remain unquantized. No Adapter/Core patch is added.

The persistent cache remains FP8. The additional workspace/copy cost is
explicit in `adapter_extension/README.md`; no performance result is claimed.
D192 uses zero-extension to D256 while preserving the original softmax scale
and returning the original 192 channels. Other advertised head dimensions
(64, 128, 256) call the vendor at their original size.

## Validation

Local foundation: Core `c440c089fe9621c5b03e1618cf86a13e95de8bf1`,
Adapter `1c9448e1b09514a8d1424ad0d50d70cb88a6d078`, Plugin base
`13bc3236e0dbe1110e5de5ef63d9fec46917d0d2`,
FlashAttention `2.8.4+dtk2610.torch2130.2611111718.g7b3562`, Torch 2.13/DTK26.10.
Versions and SHAs identify this evidence, not runtime exact-version locks.

- The original admission reproducer fails all four cases before the change.
- 153 focused CPU contracts pass, including existing Flash admission, activation,
  packaging, E5M2 contracts and optional gather, plus E4M3 FN/FNUZ byte/scale,
  output identity, zero-extension and tuple-result checks. CPU provider doubles
  are separate from native evidence.
- 64 installed native cases pass on gfx938, with the source test guard enabled.
  They cover both aliases, HND/NHD, BF16/FP16 Q, all four head sizes, ragged and
  empty sequences, prefill and decode, negative slots, byte-for-byte cache and
  untouched padding, non-unit scales, and three changing-input Graph replays
  per case with unchanged addresses. Independent FP32 attention tolerances
  remain `atol=rtol=0.016`; eager/replay equality is exact.
- Initial native selection: 48 pass / 16 D192 Graph failures. Clearing workspace
  alone still gives 48/16. Zero-extension resolves those failures without
  modifying the original numerical assertions; both failed runs are retained.
- CI-tool 97 and skill-tool 48 tests pass. The initial CI-tool invocation has
  three fixture failures from inherited system `PYTHONPATH`; the unchanged
  suite passes with that injection removed. No fixture assertion was weakened.
- Paired real-model BF16/E4M3 HumanEval32 is pending at initial submission.
  An initial launch used removed `--calculate-kv-scales` and exited before
  model execution; it has no accuracy score. The corrected launch follows
  the frozen v0.33 CLI. Do not transfer prior E5M2/model scores to this path.

Run the focused suites with the matching installed dependencies and target:

```bash
VLLM_TARGET_ROOT=/path/to/core python -m pytest -q \
  tests/vllm033/test_flash_e4m3.py tests/vllm033/test_flash_attention.py \
  tests/vllm033/test_prefill_gather.py tests/vllm033/test_flash_activation.py \
  tests/vllm033/test_package_boundary.py
VLLM033_HCU_TEST=1 HCU_V033_VERIFY_INSTALLED=1 python -m pytest -q \
  tests/vllm033/test_flash_e4m3_hcu.py
```

The gfx936, checkpoint accuracy, lifecycle and performance gates remain
separate. This does not fix the existing E5M2 accuracy gap or qualify the full
v0.33 migration. DeepSeek V4.1 checkpoints were not evaluated.
