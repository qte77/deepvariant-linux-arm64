# Oracle A2 (AmpereOne) — Generic TF Wheel Test Procedure

## Background

The DeepVariant Docker image bundles a TF 2.13.1 aarch64 wheel that includes
OneDNN+ACL compiled for Neoverse-N1. On AmpereOne (Siryn), this causes SIGILL
during make_examples' small model inference (`small_model/inference.py:141`).

The objdump gate (2026-03-06) found **zero** BFMMLA/SMMLA/UMMLA/FMLAL
instructions in the TF binaries — ACL is statically linked. The SIGILL is
likely from SVE instructions emitted by the Graviton-tuned TF wheel running
on AmpereOne, which may not implement the same SVE feature subset.

This document captures the test procedure for a generic (non-Graviton-tuned)
TF aarch64 wheel on AmpereOne.

## Prerequisites

- Oracle A2 instance (16 OCPU AmpereOne, 64 GB RAM recommended)
- Docker with `deepvariant-arm64:onnx-fix-v3` image
- SSH access: `ssh -i ~/.ssh/oci_a1 ubuntu@158.180.47.165`

## Step 1: Confirm SVE Status on AmpereOne

```bash
grep " sve " /proc/cpuinfo && echo "SVE PRESENT" || echo "NO SVE"
```

If "NO SVE" — this strongly suggests the SIGILL is from SVE instructions in
the TF binary that AmpereOne does not implement. AmpereOne has the `sve` flag
in some configurations but not others depending on firmware.

## Step 2: Identify Current TF Wheel

```bash
docker run --rm --entrypoint bash deepvariant-arm64:onnx-fix-v3 -c \
  "pip3 show tensorflow 2>/dev/null || pip3 show tensorflow-cpu 2>/dev/null"
```

Note the wheel version and source (pypi, AWS-specific, etc.).

## Step 3: Scan for SVE Instructions in TF Binaries

`ptrue` is uniquely SVE with no ASIMD equivalent — any match confirms SVE
code in the binary:

```bash
docker run --rm --entrypoint bash deepvariant-arm64:onnx-fix-v3 -c "
  TF_DIR=\$(python3 -c 'import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))')
  echo 'Scanning TF binaries for SVE instructions...'
  find \"\${TF_DIR}\" -name '*.so*' | while read -r so; do
    count=\$(objdump -d \"\${so}\" 2>/dev/null | grep -cE '^\s+[0-9a-f]+:\s+[0-9a-f ]+\s+(ptrue|whilelo|ld1[bhdw]|st1[bhdw])' || echo 0)
    if [[ \${count} -gt 0 ]]; then
      echo \"  \${so}: \${count} SVE instructions\"
    fi
  done
  echo 'Done.'
"
```

If SVE instructions are found, this confirms the hypothesis.

## Step 4: Install Generic TF Wheel

Run inside the container interactively:

```bash
docker run -it --entrypoint bash deepvariant-arm64:onnx-fix-v3

# Inside container:
pip3 install --force-reinstall tensorflow-cpu==2.13.0
```

The generic `tensorflow-cpu` wheel from PyPI should target a baseline aarch64
ISA without SVE-specific code paths.

## Step 5: Run SIGILL Reproducer

```bash
# Still inside container:
TF_ENABLE_ONEDNN_OPTS=1 python3 -c "
import tensorflow as tf
import numpy as np
x = tf.constant(np.zeros((1, 100, 221, 7), dtype=np.uint8))
print('TF eager execution works:', x.shape)
# Try loading the model if available:
import os
model_path = '/opt/models/wgs'
if os.path.exists(model_path):
    m = tf.saved_model.load(model_path)
    print('Model loaded successfully')
    result = m.signatures['serving_default'](input=tf.cast(x, tf.float32))
    print('Inference SUCCESS:', {k: v.shape for k, v in result.items()})
else:
    print('No model at', model_path, '- basic TF test passed')
"
```

## Step 6: Quick Benchmark (if Step 5 Succeeds)

```bash
# Exit container, run from host:
bash scripts/benchmark_jemalloc_ablation.sh \
  --runs 2 \
  --usd-per-hr 0.32 \
  --num-shards 16 \
  --tf-onednn-opts 1 \
  --image deepvariant-arm64:generic-tf-test \
  --data-dir /data
```

## Expected Outcomes

### Success: Generic wheel eliminates SIGILL

BF16 path becomes available on AmpereOne. With `ONEDNN_DEFAULT_FPMATH_MODE=BF16`
(AmpereOne has the `bf16` CPU flag):

- Expected call_variants speedup: ~38% (matching Graviton3 BF16 results)
- Projected cost: 584s × 0.62 ≈ 362s → **~$1.55/genome**
- With jemalloc: possibly below **$1.44/genome**

This would make Oracle A2 the cheapest platform by a wide margin.

### Failure: SIGILL persists with generic wheel

The SIGILL is not SVE-related. Possible causes:
- ASIMD encodings specific to Armv8.6-A optional extensions that differ
  between Neoverse-N1 and AmpereOne
- ACL runtime dispatch selecting a code path based on CPUID that doesn't
  match AmpereOne's actual capabilities

Next steps in this case:
1. **GDB analysis:** `gdb --batch -ex run -ex bt -ex 'x/4i $pc'` to identify
   the exact illegal instruction encoding
2. **DNNL_VERBOSE=1** to see which OneDNN primitive is dispatched before crash
3. **Build ACL from source** with `-march=armv8.6-a+bf16+i8mm` natively on
   Oracle A2

## References

- [docs/oracle-a2-sigill.md](oracle-a2-sigill.md) — Full SIGILL investigation
- CLAUDE.md § Phase 2.2d — Oracle A2 benchmark results
- AmpereOne CPU flags: `fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
  fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp
  sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm
  svebf16 i8mm bf16 dgh rng`
