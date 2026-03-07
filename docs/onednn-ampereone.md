# OneDNN ACL Dispatch Bug on AmpereOne — Diagnostic Trail

## Summary

The TF 2.13.1 aarch64 wheel bundles OneDNN with ACL (Arm Compute Library)
statically linked, compiled for Neoverse-N1. On AmpereOne (CPU part `0xac3`,
Armv8.6-A), ACL's advanced ISA dispatch path triggers SIGILL after a
non-deterministic number of `predict_on_batch` calls on the small dense model
(70-dim input, `small_model/inference.py:141`).

**Fix:** `TF_ENABLE_ONEDNN_OPTS=0` for ALL binaries (Eigen fallback).
call_variants also crashes with OneDNN+BF16 on AmpereOne — the SIGILL affects
both the small dense model in make_examples AND the InceptionV3 CNN in
call_variants. Use INT8 ONNX for call_variants instead.

**Why not `ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD`?** This ISA cap works for 2-way
concurrency but still triggers SIGILL under 16-way parallel shards. The bug
persists even in ASIMD-only mode under high concurrency, suggesting the root
cause is deeper than ISA dispatch — possibly memory corruption in ACL's thread
pool or allocator under many concurrent TF sessions.

**Impact:** The small model is a 70-dimensional dense network that runs in
microseconds per call. Disabling OneDNN for it has negligible performance
impact on make_examples. All BF16 speedup (1.61x call_variants) comes from the
large CNN in call_variants, which is unaffected.

## Root Cause

ACL's kernel dispatcher selects optimized code paths based on runtime CPU
feature detection. On AmpereOne, the dispatcher selects an advanced path
(likely SVE matmul kernels) that contains instructions AmpereOne does not
implement correctly — despite AmpereOne advertising the relevant CPU flags.

The bug is **latent**: it does not trigger on the first inference call. It
manifests after a non-deterministic number of `predict_on_batch` calls
(observed range: 20K–58K candidates processed, corresponding to thousands of
individual inference calls). This suggests the problematic kernel is selected
lazily — either via a JIT-like optimization path or a statistics-based dispatch
that switches kernels after observing enough calls.

ACL is statically linked into `libtensorflow_cc.so.2` (569 MB). There is no
separate `libarm_compute.so` to replace. The only options are:
1. Cap ISA to avoid the dispatch path (`ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD`)
2. Disable OneDNN entirely (`TF_ENABLE_ONEDNN_OPTS=0`)
3. Rebuild TF from source on AmpereOne (ACL compiles natively for host CPU)

Option 1 is the production fix. Option 3 is a potential future improvement.

## Diagnostic Tests (2026-03-07)

All tests on Oracle A2 VM.Standard.A2.Flex (16 OCPU / 32 vCPU, 64 GB RAM),
AmpereOne (Siryn, CPU part 0xac3), Docker image `deepvariant-arm64:v1.9.0-arm64.2`.

Test data: HG003 chr20, GRCh38 reference, 35x WGS PCR-free BAM.

### Test Matrix

| Test | Shards | Concurrency | ISA Setting | OMP | Result | Crash Point |
|------|--------|-------------|-------------|-----|--------|-------------|
| Single shard | 1 | 1 process | Full (default) | 8 | **SIGILL** | ~40K candidates |
| 2-way no stagger | 2 | 2 concurrent | Full (default) | 16 | **SIGILL** | Task 1: ~20K, Task 0: ~58K |
| 2-way sequential | 2 | 1 at a time | Full (default) | 16 | **SIGILL** | Task 0: ~34K candidates |
| 2-way stagger 3s | 2 | 2 concurrent | Full (default) | 16 | Completed* | *lucky timing — not reliable* |
| 2-way NEON cap | 2 | 2 concurrent | ADVANCED_SIMD | 16 | **PASS** | — |
| 4-way NEON cap | 4 | 4 concurrent | ADVANCED_SIMD | 8 | Partial* | 2/4 shards incomplete |
| **16-way NEON cap** | 16 | 16 concurrent | ADVANCED_SIMD | 2 | **SIGILL** | 4/16 shards crashed |
| **16-way OneDNN OFF** | 16 | 16 concurrent | N/A (Eigen) | 2 | **PASS** | — |

*The staggered test completed because both shards finished their half of chr20
before the buggy dispatch path triggered. This is not a reliable fix — the
crash point is non-deterministic and varies by 3x (20K–58K candidates).*

*`ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD` works at low concurrency (2-way) but fails
under 16-way parallel shards. The bug is deeper than ISA dispatch. Only
`TF_ENABLE_ONEDNN_OPTS=0` (complete Eigen fallback) is reliable.*

### Key Observations

1. **Not a concurrency bug.** Single-process and sequential tests both SIGILL.
   Concurrency changes the timing but is not the root cause.

2. **Non-deterministic crash point.** The SIGILL occurs after processing
   20K–58K candidates (thousands of `predict_on_batch` calls). The variation
   suggests a lazy dispatch or warmup-triggered kernel switch in ACL.

3. **Both models affected.** make_examples crashes in the small model
   (`small_model/inference.py:141` → `predict_on_batch`). call_variants also
   crashes with `TF_ENABLE_ONEDNN_OPTS=1` — SIGILL during the first inference
   batch on InceptionV3 (confirmed 2026-03-07). The ACL dispatch bug affects
   all OneDNN code paths on AmpereOne, not just the small dense model.

4. **`ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD` is NOT reliable.** Works at 2-way
   concurrency but fails under 16-way parallel shards (4/16 crashed). The bug
   is deeper than ISA dispatch.

5. **AmpereOne CPU flags include `bf16` and `i8mm`** but NOT `sve`. The buggy
   dispatch path is not SVE-specific — it's in the advanced SIMD or matrix
   multiply extension path that ACL selects based on `bf16`/`i8mm` flags.

### objdump Analysis (2026-03-06)

```
# Zero BFMMLA/SMMLA/UMMLA/FMLAL instructions in TF binaries
objdump -d libtensorflow_framework.so.2 | grep -cE "(bfmmla|smmla|ummla|fmlal)" → 0
objdump -d _pywrap_tensorflow_internal.so | grep -cE "(bfmmla|smmla|ummla|fmlal)" → 0

# No separate ACL shared objects — statically linked
find / -name "libarm_compute*.so" → (empty)
```

The matrix multiply instructions (BFMMLA etc.) are not present as static code.
ACL likely emits them via a runtime dispatch mechanism or uses other advanced
instructions (e.g., dot product, advanced SIMD encodings) that happen to be
incompatible with AmpereOne's implementation.

## Production Configuration

```bash
# docker_entrypoint.sh — AmpereOne section
# OneDNN OFF for ALL binaries (ACL SIGILL affects both ME and CV)
if [[ "${_part}" == "0xac3" ]]; then
  export TF_ENABLE_ONEDNN_OPTS=0
fi
```

**call_variants also crashes** with `TF_ENABLE_ONEDNN_OPTS=1` on AmpereOne.
Tested 2026-03-07: CV with BF16 produces SIGILL during the first inference
batch on the InceptionV3 model (different ACL dispatch path than the small
model, but same root cause — N1-targeted ACL on AmpereOne ISA).

Best working backend for AmpereOne call_variants: INT8 ONNX (`--use_onnx`,
0.358 s/100, $2.32/genome with jemalloc).

## Future: Rebuild TF from Source

Building TF v2.13.1 natively on AmpereOne would compile ACL for the host CPU's
actual ISA profile, likely eliminating the dispatch bug. This would remove the
need for `TF_ENABLE_ONEDNN_OPTS=0` and unlock the full BF16 BFMMLA path for
both make_examples and call_variants.

**Estimated effort:** 4–8 hours build time on 32 vCPU, ~$3–5 compute cost.
**Estimated benefit:** Significant — BF16 call_variants would drop from
0.358 s/100 (INT8 ONNX) to ~0.232 s/100 (projected from Graviton3 BF16),
reducing $/genome from $2.32 to ~$1.44.

## References

- [docs/oracle-a2-sigill.md](oracle-a2-sigill.md) — Original SIGILL investigation
- [docs/oracle-a2-wheel-test.md](oracle-a2-wheel-test.md) — Wheel swap test procedure
- CLAUDE.md § Phase 2.2d — Oracle A2 benchmark results
- AmpereOne CPU flags: `fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
  fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp
  sha512 asimdfhm dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm
  svebf16 i8mm bf16 dgh rng`
