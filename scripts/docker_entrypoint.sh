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
# shards. Verified ~14-17% ME improvement on both Graviton3 and AmpereOne.
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

# Hard safety: AmpereOne + OneDNN causes SIGILL (ACL compiled for Neoverse-N1).
# This override is always active — a SIGILL crash is worse than an unexpected
# env change. See docs/oracle-a2-sigill.md for details.
if [[ -f /proc/cpuinfo ]] && grep -qm1 "0xc0" /proc/cpuinfo 2>/dev/null; then
  _part=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || true)
  if [[ "${_part}" == "0xac3" && "${TF_ENABLE_ONEDNN_OPTS:-0}" == "1" ]]; then
    export TF_ENABLE_ONEDNN_OPTS=0
    echo "deepvariant: SAFETY: AmpereOne detected — forcing TF_ENABLE_ONEDNN_OPTS=0 (OneDNN+ACL causes SIGILL on this CPU)" >&2
  fi
fi

# Optional autoconfig: detect CPU and apply recommended settings for unset vars.
# Enable: docker run -e DV_AUTOCONFIG=1 ...
# User-provided env vars always win — autoconfig only sets vars not already set.
if [[ "${DV_AUTOCONFIG:-0}" == "1" ]]; then
  _AUTOCONFIG_SCRIPT="/opt/deepvariant/scripts/autoconfig.sh"
  if [[ -x "${_AUTOCONFIG_SCRIPT}" ]]; then
    _ac_json=$("${_AUTOCONFIG_SCRIPT}" --json 2>/dev/null || true)
    if [[ -n "${_ac_json}" ]]; then
      # Parse JSON and apply env vars that are not already set
      # shellcheck disable=SC2154  # k,v are Python variables in the embedded script
      eval "$(echo "${_ac_json}" | python3 -c "
import sys, json
try:
    cfg = json.load(sys.stdin)
    env = cfg.get('env', {})
    for k, v in env.items():
        if v is not None:
            print(f'[[ -z \"\${{${k}:-}}\" ]] && export {k}=\"{v}\" && echo \"deepvariant: autoconfig set {k}={v}\" >&2')
except Exception:
    pass
" 2>/dev/null || true)"
      _ac_cpu=$(echo "${_ac_json}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cpu_family',''))" 2>/dev/null || true)
      _ac_backend=$(echo "${_ac_json}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('backend',''))" 2>/dev/null || true)
      echo "deepvariant: autoconfig applied (${_ac_backend} on ${_ac_cpu})" >&2
    fi
  else
    echo "deepvariant: WARNING: DV_AUTOCONFIG=1 but autoconfig.sh not found at ${_AUTOCONFIG_SCRIPT}" >&2
  fi
fi

exec "$@"
