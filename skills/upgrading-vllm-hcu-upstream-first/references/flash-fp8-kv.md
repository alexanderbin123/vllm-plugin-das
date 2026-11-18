# Flash FP8 KV admission and execution

When a selected Flash backend rejects `kv_cache_dtype`, inspect the class
actually published in the public backend registry and its dtype predicate.
Platform selection can be functioning while the Plugin's reader is incomplete.
Do not add a dtype to the support list without its full data path.

- Bind the public cache name, platform FP8 encoding, physical byte storage,
  native writer and real reader. On ROCm, E4M3 FN and FNUZ have different byte
  contracts; use the platform's actual encoding. Never reinterpret E4M3 as E5M2.
- Check prefill and decode independently, including all advertised head sizes,
  layouts, mixed and empty sequences, non-unit K/V scales, and live prefix
  block tables. A vendor's FP8 decode kernel does not imply FP8 prefill support.
- Preserve unquantized queries unless Q quantization has its own explicit
  contract and accuracy evidence. A public varlen wrapper may cast Q based on
  K's dtype or ignore descales on one internal route.
- Prefer Core's native KV writer and gather interfaces. A Flash composition
  can keep the persistent cache in FP8 and dequantize gathered KV to BF16/FP16,
  but must disclose transient memory/copy costs and identify the actual Flash
  compute dtype. Do not call that a native FP8 compute performance result.
- Use static metadata bounds for Graph allocations and device cumulative
  lengths for actual reads. Verify changed lengths, tables, scales and values
  with unchanged addresses. Preserve padding and untouched cache bytes against
  an independent encoding oracle; clear unused workspace when needed.
- When a vendor tile fails, keep its original failure. A zero-extension to a
  supported head dimension preserves attention only if Q/K/V added channels
  are zero, the original softmax scale is retained, and the original output
  channels are restored. Test eager/Graph against independent FP32 attention;
  do not weaken tolerances or mask NaNs.
- Record source, installed wheel/RECORD, worker origins and live cache metadata.
  Then compare the same checkpoint and HumanEval requests with the existing
  BF16 path, keeping raw and normalized scores and pass-set changes separate.
  Unit/Graph correctness does not waive model, lifecycle or performance gaps.
