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

Set `TF_ENABLE_ONEDNN_OPTS=0` to fall back to TF Eigen backend:

```bash
docker run -e TF_ENABLE_ONEDNN_OPTS=0 ...
```

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

## Gate Results: objdump Analysis (2026-03-06)

Ran the gate commands inside the Docker container on AmpereOne:

```bash
# Result: ZERO matches for BFMMLA/SMMLA/UMMLA/FMLAL
objdump -d /usr/local/lib/python3.10/dist-packages/tensorflow/libtensorflow_framework.so.2 \
  | grep -cE "(bfmmla|smmla|ummla|fmlal)"
# Output: 0

objdump -d /usr/local/lib/python3.10/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so \
  | grep -cE "(bfmmla|smmla|ummla|fmlal)"
# Output: 0

# No separate ACL shared objects found — ACL is statically linked
find / -name "libarm_compute*.so" 2>/dev/null
# Output: (empty)
```

**Finding:** The SIGILL is NOT caused by BF16 matrix multiply instructions
(BFMMLA/SMMLA). ACL is statically linked into the TF wheel, and neither the
framework .so nor the Python binding .so contain these instructions.

**Revised hypothesis:** The illegal instruction is likely in a different ACL
kernel path — possibly SVE instructions (AmpereOne has SVE but with a
potentially different vector length or feature subset than what ACL targets),
or advanced SIMD encodings specific to Neoverse-N1's implementation. The crash
occurs during eager execution of the small model CNN, suggesting a Conv2D or
MatMul kernel is the culprit.

**Next steps:**
1. Run under `gdb` to capture the exact faulting instruction address and
   disassemble the surrounding code
2. Check `DNNL_VERBOSE=1` output to see which OneDNN primitive is being
   dispatched before the crash
3. Broader `objdump` search for SVE instructions (`sve` prefix, `z0-z31`
   register operands) in the TF binaries

## Proposed Experiments (Not Yet Implemented)

1. **GDB analysis** — `gdb --batch -ex run -ex bt -ex 'x/4i $pc'` to identify
   the exact illegal instruction
2. **Rebuild ACL** with `-march=armv8.6-a+bf16+i8mm` natively on Oracle A2
3. **Try `tensorflow-cpu-aws` wheel** — may target a more generic ISA profile
4. **SVE vector length check** — `cat /proc/sys/abi/sve_default_vector_length`
   to verify AmpereOne's SVE configuration matches ACL's expectations

## Expected Impact

If BF16 is unlocked on AmpereOne (which has `bf16` and `i8mm` CPU flags),
call_variants should see a ~38% speedup (matching Graviton3 BF16 results).
This would push Oracle A2 cost from $2.32/genome to approximately $1.44/genome
— the cheapest ARM64 configuration by far.
