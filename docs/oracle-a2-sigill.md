# Oracle A2 (AmpereOne) OneDNN SIGILL Investigation

## Problem

DeepVariant crashes with SIGILL (illegal instruction) on Oracle A2 instances
when OneDNN+ACL is enabled (`TF_ENABLE_ONEDNN_OPTS=1`).

## Environment

- **Instance:** Oracle A2 (AmpereOne / Siryn), 16 OCPU (32 vCPU), 64 GB RAM
- **CPU flags:** `fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm svebf16 i8mm bf16 dgh rng`
- **Docker image:** `ghcr.io/antomicblitz/deepvariant-arm64:optimized`
- **TF version:** 2.13.1 (aarch64 wheel)
- **OneDNN:** Bundled with TF wheel (linked against ACL)

## Failing Command

```bash
docker run -e TF_ENABLE_ONEDNN_OPTS=1 ... /opt/deepvariant/bin/run_deepvariant ...
```

Crash occurs during `make_examples` at `small_model/inference.py:141` during
TF eager execution. The crash is in the OneDNN+ACL code path, not in
call_variants.

## Workaround

Set `TF_ENABLE_ONEDNN_OPTS=0` to fall back to TF Eigen backend. The Docker
entrypoint already does this automatically when SIGILL is not caught, but
explicitly setting it avoids the crash entirely.

Alternatively, use ONNX backend (`--use_onnx`) which bypasses TF entirely for
call_variants. make_examples still uses TF Eigen for the small model.

## Root Cause Hypothesis

The TF aarch64 wheel bundles ACL (Arm Compute Library) compiled for
Neoverse-N1. AmpereOne is Armv8.6-A but has a different optional extension
profile than Neoverse-N1. ACL likely emits instructions (e.g., specific SVE or
advanced SIMD encodings) that are valid on Neoverse-N1 but not on AmpereOne's
microarchitecture.

Despite both being Armv8.6-A, the ISA is a superset specification — individual
cores choose which optional extensions to implement. The ACL binary may use
extensions present on N1 but absent on AmpereOne (or vice versa).

## Gate: Confirm Before Rebuilding

Do NOT invest 6-10 hours rebuilding TF/ACL until `objdump` confirms the
illegal instruction is in ACL (not TF core). Run these commands inside the
Docker container on an AmpereOne instance:

```bash
# Find instructions that may cause SIGILL on AmpereOne
objdump -d /opt/conda/lib/python3.10/site-packages/tensorflow/libtensorflow_framework.so.2 \
  | grep -E "(bfmmla|smmla|ummla|fmlal)" | head -20

# Check ACL shared object if present
find / -name "libarm_compute*.so" 2>/dev/null -exec objdump -d {} \; \
  | grep -cE "(bfmmla|smmla)"
```

If these instructions appear in the binary but AmpereOne doesn't support the
specific encoding variant used, that confirms the rebuild hypothesis.

## Proposed Experiments (Not Yet Implemented)

1. **Rebuild ACL** with `-march=armv8.6-a+bf16+i8mm` natively on Oracle A2
2. **Try `tensorflow-cpu-aws` wheel** — may target a more generic ISA profile
3. **`objdump` analysis** — identify the exact illegal instruction encoding

## Expected Impact

If BF16 is unlocked on AmpereOne (which has `bf16` and `i8mm` CPU flags),
call_variants should see a ~38% speedup (matching Graviton3 BF16 results).
This would push Oracle A2 cost from $2.32/genome to approximately $1.44/genome
— the cheapest ARM64 configuration by far.
