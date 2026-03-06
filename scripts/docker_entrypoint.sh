#!/bin/bash
# Docker entrypoint for DeepVariant ARM64.
# Sets TensorFlow/ONNX Runtime optimizations based on CPU features.
# All variables can be overridden via docker run -e VAR=value.

set -e

# TensorFlow OneDNN+ACL optimizations
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-1}"

# Threading configuration
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"

# TF threading: all cores for intra-op (GEMM), single thread for inter-op
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$(nproc)}"

# BF16 fast math on Graviton3+ (Neoverse V1/V2 with BF16 support)
if [[ -z "${DNNL_DEFAULT_FPMATH_MODE:-}" ]] && grep -q bf16 /proc/cpuinfo 2>/dev/null; then
  export DNNL_DEFAULT_FPMATH_MODE=BF16
fi

# Optional jemalloc allocator — reduces malloc contention under many concurrent
# shards. Preliminary testing shows ME and CV improvements on AmpereOne.
# Enable: docker run -e DV_USE_JEMALLOC=1 ...
# Override path: docker run -e DV_JEMALLOC_PATH=/custom/path/libjemalloc.so ...
if [[ "${DV_USE_JEMALLOC:-0}" == "1" ]]; then
  _JEMALLOC="${DV_JEMALLOC_PATH:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}"
  if [[ -f "${_JEMALLOC}" ]]; then
    export LD_PRELOAD="${_JEMALLOC}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "deepvariant: jemalloc enabled (${_JEMALLOC})" >&2
  else
    echo "deepvariant: WARNING: DV_USE_JEMALLOC=1 but ${_JEMALLOC} not found, continuing without jemalloc" >&2
  fi
fi

exec "$@"
