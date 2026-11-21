# Validate a native provider built from source

Resolve the provider's full commit in its actual repository and freeze the build
profile, dependencies and wheel checksum. Matching installed Python files can
identify a source candidate but does not establish the original native build.
Inspect the selected packaging entry point: unified and legacy projects can
declare different versions at the same commit. Preserve the resulting metadata;
do not rename the new artifact to impersonate the installed vendor release.

Check native dependencies after installation outside the source and build trees.
A wheel repair can bundle ordinary libraries while leaving DTK/HSA or network
libraries external. Record those runtime requirements separately from the wheel's
platform tag. Verify the full installed RECORD, wheel payload and actual parent
and child module/DSO origins; build success alone does not qualify the artifact.

Preserve the original packaging regression when it fails. If a facade assertion
conflicts with an intentional private helper, inspect the actual callers and
dedicated facade tests before changing runtime behavior or the assertion. Check
attribute access, public exports and backend-specific rejection separately. A
passing diagnostic does not make the unchanged packaging regression pass.

Run source fixtures that require a missing native extension in an environment
without the installed provider. A namespace package using `extend_path` can find
the installed extension even when PYTHONPATH points at source, invalidating that
negative control. Use a separate installed-wheel environment for native tests;
retain initial environment failures and bind each result to its actual source.

Use existing transfer tests where possible. Native TCP read/write, async/scatter
and GIL tests establish host transport behavior only. HIP IPC, device registration,
RDMA, P/D model output, Graph, accuracy, lifecycle and performance retain their
own applicable gates. Record owned-process exits and listener cleanup separately
from numerical assertions; retain an initial bind refusal and any later recovery.


For a default-OFF native optimization, trace real activation and the ordinary
model path as well as the enable switch: registering an OOT class or fake/custom
op need not import its vendor DSO. Preserve official fallback, compile-cache
factors and explicit required-capability admission while deferring an unused
provider. Required extensions use published entry-point names, which can differ
from declared spec names; exercise the real host activation with that metadata.
Reuse Adapter's process-local lifecycle for serialized first-use ABI checks
and fork refusal; a health checker reports failures without initializing
or repairing the provider. Verify enabled ordinary/fused execution and changing
input Graph replay after an installed-wheel binding check. Count shutdown leaks
and the original cleanup deadline separately: fewer imports do not establish a
clean exit or a speedup. The Plugin RMS example is documented in
`docs/validation/vllm033-rms-lazy-provider.md`.

Before carrying a constructor patch that forces a native operator, inspect the
target's dispatch and precision guards. For clamp SwiGLU, current v0.33 Core
already selects its ROCm kernel for default alpha/beta and retains its Torch
path for nondefault parameters. Preserve that policy and official custom-op
enablement rather than forcing every configuration through the kernel. Reuse
upstream eager tests and add changing-input Graph coverage on the real class;
see `docs/validation/vllm033-official-clamp-swiglu.md`. Component ownership does
not qualify a checkpoint, HumanEval, performance or whole-model shutdown.

## Cross-framework GDN verify contracts

For Adapter's GDN target-verify bridge, describe the upstream and vendor
signatures independently. An upstream-only `lower_bound=None` may be omitted
from the vendor call after reviewing its meaning. Every non-None request,
including zero and `False`, must retain the original upstream semantics;
do not silently discard it. Unknown added parameters still reject publication.
Preserve the original callable object, signature and previously cached aliases.

Inspect the actual native gate strides. In the observed AITER verify kernel,
`a` has no token/head stride and `b` has no head stride. Materialize interleaved
gate views without changing their values; retain the existing Q/K/V stride
contract. Check independent FP32 recurrence, snapshot slots, unchanged initial
state and changing-input Graph replay. Successful binding alone does not
establish numerical compatibility.

Keep explicit-provider numerical results separate from automatic admission.
A passing explicit suite on a new stack does not qualify that stack's default
selection or skipped automatic-admission tests. Extending admission still needs
its actual eager/Graph/model evidence; do not loosen the table to make skips pass.

## Native JIT and interrupted cold builds

When a vendor builds after a native-module miss, compatibility stubs must not
report a missing native module as successfully imported. Verify real discovery,
the missing-module exception, the vendor build and native execution separately.
Use privately owned package and cache directories for default JIT writes;
checking wheel RECORDs does not identify newly generated native artifacts.

After a cold-build timeout, a live retry can be waiting on an abandoned
FileBaton lock. Inspect its actual stack, compiler descendants and artifact
progress rather than inferring compilation from a live parent or more jobs.
Preserve the timeout and waiting evidence. Before recovery, confirm all users
of that private cache have exited, bind the lock path/inode/time to the original
run, and preserve the old lock in a new run's evidence. Do not clear shared or
another client's locks. An interrupted run retains its failed lifecycle result;
cache recovery does not retroactively qualify its tests.

## Varlen FlashAttention and allocation-sensitive failures

When identical `cu_seqlens_q` and `cu_seqlens_k` values produce different native
results depending on pointer identity, compare shared and separate buffers in
fresh processes while always allocating both buffers. This controls allocation
count as well as metadata values; retaining or synchronizing a temporary tensor
alone does not establish a lifetime bug. Preserve the original separate-buffer
case rather than forcing aliasing to pass.

To probe a suspected sequence-boundary read, preserve logical Q/K/V and metadata
while varying only allocated storage after the logical K/V end (zero versus NaN,
K and V independently). Compare with an independent FP32 attention reference in
eager and changing-input Graph replay. A result that depends on this tail
localizes a native memory-boundary problem; it does not prove which source line
is defective, or establish FP8 KV support.

Resolve the installed provider's real source revision before attributing the
failure to a current vendor branch. A masking-loop bound found in different
source remains a candidate until that source is built and the original failing
cases pass on the device. Avoid an Adapter/Plugin padding or metadata-alias
workaround that conceals a native out-of-bounds read; keep provider ownership and
unverified capability explicit.

## CPU import probes and device namespaces

Separate framework management queries from provider computation when isolating
CPU contracts. In the inspected v0.33 Core, importing `RocmPlatform` queries
architecture through AMD SMI and falls back to Torch device properties if that
query fails. Hiding every device node can therefore fail collection before a
selector or provider is tested. Check the actual management query and Torch
initialization state before choosing the namespace; preserve a real architecture
query rather than inventing architecture or availability results.

Vendor imports have separate requirements. The observed DeepGEMM import runs
`rocminfo` for a CU count and fails on `None` when its device namespace prevents
that query. A parsed count alone does not identify a GPU or prove usable kernels.
Keep that failure and its original JUnit. If matching production CI's namespace
repairs the environment, rerun the unchanged affected contract with real provider
imports, empty compute masks and a Torch-initialization guard where applicable.
Reuse unaffected completed cases only with unchanged source, dependencies and
configuration evidence. If combining JUnit reports, label the derived report,
retain each original report and verify the exact case multiset. CPU import/ABI
success does not qualify native numerics, Graph, model accuracy or lifecycle.

If an installed contract forwards the parent's `sysconfig.get_path("purelib")`
to a child, use an environment whose real prefix and purelib contain the
candidate distribution. A `--target` install plus a `sys.path`/executable wrapper
does not change sysconfig; the observed Step5 probes therefore imported an older
global Plugin. Preserve the original child interpreter flags and verify its
actual module origins after repairing the environment. On Python 3.12, `-S`
can retain the base prefix even for a venv executable, so prefix alone is not
proof of which payload ran.

Check conditional vendor exports before attributing a CPU activation failure to
the Plugin. The inspected FlashAttention package exports its native callables
only when `torch.cuda.is_available()` is true at import. Empty compute masks
leave the package discoverable without `flash_attn_varlen_func`. Preserve that
failure and qualify the unchanged activation on a real supported device; do not
fake availability or inject the missing callable to claim host activation.

## Inverse-RoPE rounding boundaries

Compare the target Core's emitted instructions, not only its Python arithmetic.
One DTK 26.10/gfx938 Triton build rounds the sine product before fusing the
cosine product; another supported installed build emits separate products
and sum even with `enable_fp_fusion=True`. Do not transfer either contract
across installations or infer it from that compiler flag or a wheel string.
At provider preparation, compare both native modes with the actual Core helper
using distinguishing cancellation inputs and both position dtypes. Admit only
one consistent match; reject unknown or ambiguous behavior before execution.
Bind the selected mode once, before Graph capture; forward and health checks
must not rerun calibration, initialize vendors or change the selection.
Keep the original failed gates and the old installation's results separately.
Freeze cancellation witnesses with independent FP64 oracles for both modes,
retain exact eager and changed-input Graph assertions, and cover both position
index dtypes. This component result does not qualify a model or other builds.
PCP forward Q shares the native layout but requires separate FP32 products;
validate both users together rather than changing the shared rounding globally.


## Optional frontend providers and later requirements

An optional OFF activation must not satisfy a later explicitly requested entry
point or capability. Recheck requirement-specific eligibility on every activation,
including the already-activated branch; preserve published policy/owner drift as
commit-fatal. Exercise both first activation and optional-OFF followed by required
activation through the real Adapter loader. Keep frontend/Core import guards and
default-off behavior. Adding a production entry point also updates the single
exact packaging table, checked against both source metadata and the actual wheel;
do not weaken it to a subset check. These CPU contracts do not prove frontend
process reachability, model output or serving performance.


For dependent extensions, verify the actual public loader's priority direction.
The Adapter loader orders higher priorities first: a provider that only checks an
Adapter-owned registration must run after that registration. Exercise the real
loader and actual descriptor priorities with reversed entry-point discovery order,
then check the installed Core public registry with the paired Adapter/Plugin.
A source-only provider delegation test cannot establish registration reachability.

## HY4 target and draft static composition on v0.33

Use the [HY4 migration ledger](../../../docs/validation/hy4-v033-migration.md)
for the current deployment scope and remaining model gates. Require both
`hy4-ct-w4a8` and `hy4-static-eplb`; the static owner follows the Adapter CT
registration and its HY4 health descriptor. Check the real installed loader,
quantization registry and target/MTP model registry together, preserving the
official quantization classes and foreign owners.

Retain Core's decorated constructors and forward methods when substituting
child factories. Request physical expert capacity before allocation; validate
logical-to-physical fan-out and routing for both backbone and natural MTP.
The separate legacy version-2 cookbook reader does not invent checkpoint or
calibration provenance absent from that artifact. Verify both native model
keys, declared counts, placement shape and cross-rank fingerprint; keep the
existing stricter V41 map parser independent.

A GPU-hidden ABI check cannot qualify the enabled host. For actual startup,
retain the Adapter platform/general entry point and the deployment's verified
DTK roots: a CPU `VLLM_PLUGINS=none` setting disables the platform owner too.
Registration and CPU constructor/loader checks still do not establish PCP16
execution, full-model Graph, checkpoint accuracy, raw HumanEval or performance.


For HY4 CT checkpoints, keep logical expert keys in the native mapping when the
static loader already performs replica fan-out; physical allocation and routing
still retain redundant slots. A symmetric channel-INT8 indexer WK may reuse the
native WK/scale pairing, scaled dequantization and fused shard loader. Cover both
arrival orders and target/MTP loaders, reject unsupported zero points, and check
that native loader bodies and module globals remain intact. Ordinary explicitly
required CT admission and static CT composition have separate registry owners;
test both through the real extension loader.

Sparse MLA configuration may reach backend selection before MLAAttention's cache
canonicalization. If that call order applies to the frozen Core, reuse its helper
for the selected sparse backend after provider admission; do not broaden dtype
capability tables or convert unrelated backends. Persistent conversion tables can
reuse Core's decode-row bound while larger prefills use its temporary fallback.
For PCP memory profiling, validate any local token bound against actual Core
partitioning and preserve the scheduler budget and non-profile runner behavior.
Kernel/CPU registration evidence does not establish full-checkpoint PCP memory,
throughput, accepted MTP tokens or original HumanEval accuracy.


Native compilation can invoke Python console tools through PATH; the parent's
`-I` does not isolate their shebang interpreter. A recursively starting
`ninja --version` helper that loads global Torch/HIP is a setup failure, not a
kernel result. Preserve the original process tree and stop only its verified
owned group. Select a tool from the isolated prefix, or the same installed
package's native executable, recording its identity; do not rewrite the shared
wrapper or framework/vendor files. Re-run the original unfinished tests with
unchanged shapes, tolerances and Graph paths after all prior children terminate.


For PCP MTP, inspect the target's native proposal loop and synchronize each
sample before the next draft input/KV write; broadcasting only the completed
proposal is too late. Keep the native sampler, verification and request updates,
and validate the public post-load seam on the actual installed Adapter pair.
Sequential CPU group doubles establish ordering contracts, not a real collective
or accepted-token accuracy. Retain real distributed and complete model gates.

For a PCP Mooncake sender, retain the official bootstrap predicate's TP/PP/DP
and external-server rules before narrowing to canonical PCP rank zero. Keep
the public connector identifier and native scheduler/transport methods; replica
completion must still flow through Core's world-size aggregator so cache is not
freed before the real sender acknowledges. Constructor tests with transport and
thread leaves doubled do not qualify real port ownership, failure cleanup,
transfer bytes or P/D model lifecycle.


Static EPLB placement and load recording are separate contracts. Disabling
recording does not disable Core token-hash replica selection: retain the actual
loaded replica counts for HY4, and let the official router use every published
column. Rank-local nearest/locality ordering must preserve the replica set.
Keep other model owners explicitly scoped to their existing primary-only policy.
Regress target/MTP, rank and ordering policies, repeated publication identity,
invalid expert IDs and changed inputs against the actual installed Core router.
CPU hash emulation alone is insufficient; real kernel checks still do not prove
Graph replay, real EP collectives, full-checkpoint accuracy or throughput.


Replica routing Graph validation must capture the actual public Core mapping
function, then replay changed expert IDs through fixed input and output
addresses. Warm up before capture, compare each replay with an independent
scalar token-hash/placement oracle, and check disabled load counters remain
zero. Cover target/MTP, rank-local ordering and supported batch shapes;
separate graphs per batch do not qualify dynamic-batch model scheduling.
Model/EP doubles establish this component contract only, not a real collective
or full-model Graph. Verify complete shared Core/Torch/Triton/provider RECORDs
before and after; resolve each dist-info symlink to its actual owner before
interpreting RECORD-relative paths, including console-script entries.

For PCP prefill workspace coalescing, keep the policy on the owned model's
metadata builder and retain Adapter's implementation, cache format and native
forward. Merge only adjacent segments of one request; a request identity is not
evidence of shared physical pages across different requests. Preserve the raw
decode-row boundary when collective padding differs from active decode counts,
and test request-to-query mapping, workspace capacity and unsupported-layout
fallback against the actual Core builder. Default-off policy, invalid-input
refusal and the existing kill switch need separate checks. CPU metadata oracles
do not establish native numerical parity, Graph or model performance.

For PCP MTP broadcasts, reuse Core's initialized device communicator when its
PyNCCL transport is available and enabled; retain the official group fallback
otherwise. Do not create a second communicator during allocator-heavy profiling
or replace Core's communicator globally. A transport double verifies dispatch
and ordering only; real collective, draft acceptance and lifecycle remain gates.

Before reusing an installed Adapter for a newer Plugin review, compare the
actual Python production tree as well as native inputs. Unchanged native code
does not mean the query-capacity constructor or metadata builder is unchanged.
Compose the current owners, build the official host project in isolation and
verify source/wheel/installed bytes, raw RECORDs and parent origins. Label an
unstamped host-only development build separately from a newly paired native
release; reuse an older kernel only with matching native inputs and actual ABI
admission. Check that the Plugin builder retains Adapter's implementation and
that Adapter's local query bound still preserves the official global KV bound.
An import/binding check on a real HCU does not exercise native attention or a
model, and it cannot clear an older whole-model shutdown failure.
