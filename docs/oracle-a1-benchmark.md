# Oracle A1 (Ampere Altra / Neoverse N1) — Benchmark Results

## Platform Specifications

| Attribute | Value |
|-----------|-------|
| Instance | VM.Standard.A1.Flex |
| CPU | Ampere Altra (Neoverse N1) |
| CPU part | `0xd0c` (implementer `0x41`) |
| OCPUs | 16 (= 16 vCPU, 1:1 mapping) |
| RAM | 32 GB |
| Region | Frankfurt (eu-frankfurt-1) |
| Pricing | **$0.01/OCPU/hr** ($0.16/hr for 16 OCPU) |
| ISA features | ASIMD, CRC32, AES — **no BF16, no SVE, no i8mm** |
| OS | Ubuntu 22.04 aarch64 |
| Docker image | `ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2` |

**Key hardware notes:**
- Neoverse N1 lacks `i8mm` (no SMMLA instruction for INT8 acceleration). INT8 ONNX
  relies on standard NEON multiply-accumulate rather than the dedicated matrix
  multiply extensions available on Graviton3/4 (Neoverse V1/V2).
- No BF16 support — TF+OneDNN BF16 fast math is not available.
- OneDNN+ACL should work on N1 (it's the native target ISA), but was not tested
  (`TF_ENABLE_ONEDNN_OPTS=0` for all runs).

---

## Benchmark Configuration

- **Sample:** GIAB HG003, Illumina NovaSeq PCR-free 35x WGS
- **Region:** Full chr20 (63 Mbp, ~80K examples)
- **Reference:** GRCh38 (no alt analysis set)
- **Shards:** 16 (`--num_shards=16`)
- **Memory limit:** 28 GB (`--memory=28g`)
- **OneDNN:** OFF (`TF_ENABLE_ONEDNN_OPTS=0`) for all runs
- **INT8 model:** Static quantized ONNX (21 MB), mounted from host
- **FP32 ONNX model:** In-image at `/opt/models/wgs/model.onnx` (84 MB)

---

## Results

### INT8 ONNX — Primary Configuration

Interleaved jemalloc ON/OFF runs to eliminate cache-warming bias. Clean instance
(no competing workloads — an earlier rogue container was identified and killed
before these runs).

| Run | jemalloc | ME | CV | PP | Wall |
|-----|----------|-----|-----|-----|------|
| 1 | OFF | 242s | 239s | 14s | 498s |
| 2 | OFF | 247s | 267s | 14s | 532s |
| 3 | OFF | 248s | 240s | 13s | 505s |
| 4 | OFF | ~243s | ~243s | 14s | 502s |
| 1 | ON | 218s | 242s | 13s | 478s |
| 2 | ON | 220s | 242s | 14s | 479s |
| 3 | ON | 218s | 267s | 14s | 502s |

| Config | N | Mean | sigma | ME (avg) | CV (avg) | CV rate | $/genome |
|--------|---|------|-------|----------|----------|---------|----------|
| **INT8, jemalloc OFF** | 4 | **509s** | 15.4s | 245s | 247s | 0.309 s/100 | **$1.09** |
| **INT8, jemalloc ON** | 3 | **486s** | 13.6s | 219s | 250s | 0.313 s/100 | **$1.04** |
| **jemalloc delta** | | **-23s** | | **-10.6%** | ~0% | | **-4.5%** |

### Variance Note

Oracle A1 shows higher run-to-run variance than Graviton3 or Oracle A2.
Two of 7 INT8 runs hit a slow CV (267s vs typical ~240s). This produces
sigma = 14-15s vs sigma < 1s on Graviton3 and sigma = 4s on A2 (clean instance).
The variance appears in CV only (ME is stable at 242-248s OFF, 218-220s ON).

**Recommendation:** Future A1 benchmarks should use N >= 6 runs per config
to get tight confidence intervals.

### TF Eigen FP32 — Baseline

| Run | ME | CV | PP | Wall |
|-----|-----|-----|-----|------|
| 1 | 247s | 470s | 14s | **735s** |

| Config | N | Wall | CV rate | $/genome |
|--------|---|------|---------|----------|
| TF Eigen FP32 | 1* | **735s** | 0.588 s/100 | **$1.57** |

### ONNX FP32 — Comparison

| Run | ME | CV (est.) | PP | Wall |
|-----|-----|-----------|-----|------|
| 1 | 240s | ~567s | 14s | **826s** |
| 2 | 240s | ~570s | 14s | **829s** |

| Config | N | Mean | sigma | CV rate | $/genome |
|--------|---|------|-------|---------|----------|
| ONNX FP32 | 2* | **828s** | 2.1s | 0.711 s/100 | **$1.77** |

\*N < 4 runs — wider confidence interval.

### INT8 Speedup over Baselines

| Comparison | INT8+jemalloc | Baseline | Speedup | $/genome saved |
|------------|--------------|----------|---------|----------------|
| vs TF Eigen FP32 | 486s | 735s | **1.51x** | **$0.53 (34%)** |
| vs ONNX FP32 | 486s | 828s | **1.70x** | **$0.73 (41%)** |

---

## A1 vs A2 — The Key Comparison

Oracle A1 ($0.01/OCPU/hr) vs A2 ($0.04/OCPU/hr), both at 16 vCPU:

| Metric | Oracle A1 (Altra/N1) | Oracle A2 (AmpereOne) | A1 vs A2 |
|--------|---------------------|----------------------|----------|
| CPU | Neoverse N1 (`0xd0c`) | AmpereOne/Siryn (`0xac3`) | Older uarch |
| vCPU/OCPU | 1:1 | 2:1 | — |
| OCPUs for 16 vCPU | 16 | 8 | 2x more OCPUs |
| $/hr (16 vCPU) | **$0.16** | $0.32 | **2x cheaper** |
| INT8 jemalloc=OFF wall | 509s | 584s | **13% faster** |
| INT8 jemalloc=ON wall | 486s | 544s | **11% faster** |
| INT8 CV rate | 0.309 s/100 | 0.389 s/100 | **21% faster** |
| ME (jemalloc ON) | 219s | 210s | 4% slower |
| jemalloc improvement | -4.5% | -6.9% | Smaller on A1 |
| Run-to-run variance | sigma=14-15s | sigma=4s | Higher on A1 |
| **$/genome (INT8+jemalloc)** | **$1.04** | **$2.32** | **55% cheaper** |
| **$/genome (INT8, no jemalloc)** | **$1.09** | **$2.49** | **56% cheaper** |
| **$/genome (TF Eigen FP32)** | **$1.57** | **$2.69** | **42% cheaper** |

### Why A1 is Both Faster AND Cheaper

**A1 is faster per-core despite older microarchitecture.** This is unexpected.
Two factors explain it:

1. **INT8 CV is 21% faster on N1 than AmpereOne.** N1's NEON GEMM code path is
   well-optimized (it's the primary ACL target). AmpereOne uses the same N1-compiled
   NEON path (ACL SIGILL prevents native AmpereOne code), but AmpereOne's IPC is
   34% lower for this workload (1.72 vs 2.61 on Graviton3/V1). Altra/N1 likely
   has better IPC than AmpereOne for NEON-heavy workloads.

2. **ME is ~4% slower on A1 than A2.** A2's AmpereOne has newer memory subsystem
   and higher single-thread bandwidth. But the ME gap is small vs the CV advantage.

**The cost advantage comes from both factors:**
- 2x lower OCPU pricing ($0.01 vs $0.04)
- ~11-13% faster wall time (more work per hour)

Combined: A1 delivers **55% lower $/genome** than A2.

### Cost Ranking (All Platforms, Best Config, 16 vCPU)

| Rank | Platform | $/hr | Best Config | Wall | $/genome |
|------|----------|------|-------------|------|----------|
| **1** | **Oracle A1 (Altra)** | **$0.16** | **INT8+jemalloc** | **486s** | **$1.04** |
| 2 | Oracle A2 (AmpereOne) | $0.32 | INT8+jemalloc | 544s | $2.32 |
| 3 | Graviton4 (c8g.4xlarge) | $0.68 | INT8 ONNX | 366s | $3.33 |
| 4 | Graviton3 (c7g.4xlarge) | $0.58 | BF16+jemalloc | 443s | $3.43 |
| — | Google x86 (official, 96 vCPU) | $3.81 | — | — | $5.01 |

**Oracle A1 is 4.8x cheaper than Google's official x86 reference.**

---

## jemalloc Analysis

| Metric | jemalloc OFF | jemalloc ON | Delta |
|--------|-------------|-------------|-------|
| ME (avg) | 245s | 219s | **-10.6%** |
| CV (avg) | 247s | 250s | +1.2% (noise) |
| PP (avg) | 14s | 14s | ~0% |
| **Wall (avg)** | **509s** | **486s** | **-4.5%** |

ME improvement (-10.6%) is the dominant factor, consistent with other ARM64
platforms (Graviton3: -13.8%, Oracle A2: -17.0%). The smaller A1 improvement
suggests Neoverse N1's malloc is less pathological than AmpereOne's (A2 had
23.2% of cycles in libc.so.6 malloc vs Graviton3's 18.1%).

CV is unchanged — ONNX Runtime uses internal allocators that bypass glibc malloc.

---

## Parallel call_variants (4-way)

Tested on a second A1 instance (AD-2, same spec) using `run_parallel_cv.sh`.
Splits 16 ME output shards across 4 call_variants workers (4 shards each,
4 OMP threads each). Uses ONNX backend exclusively (~3-5 GB RSS per worker).

### INT8 ONNX + jemalloc — 4-way parallel CV

| Run | ME | CV (4-way) | PP | Wall |
|-----|-----|-----------|-----|------|
| 1 | 220s | 147s | 14s | 381s |
| 2 | 212s | 147s | 13s | 372s |
| 3 | 216s | 147s | 13s | 376s |

| Config | N | Mean | sigma | CV speedup | $/genome |
|--------|---|------|-------|------------|----------|
| **INT8 4-way + jemalloc** | 3 | **376s** | 4.5s | **1.70x** | **$0.80** |

CV is rock-stable at 147s across all 3 runs (sigma=0). ME variance (212-220s)
is the dominant source of wall time variance.

### FP32 ONNX + jemalloc — 4-way parallel CV

| Run | ME | CV (4-way) | PP | Wall |
|-----|-----|-----------|-----|------|
| 1 | 219s | 379s | 12s | 610s |

| Config | N | Wall | CV speedup | $/genome |
|--------|---|------|------------|----------|
| FP32 4-way + jemalloc | 1* | **610s** | 1.50x\*\* | **$1.30** |

\*\*FP32 sequential CV estimated at ~567s (from ONNX FP32 wall=828s, ME=240s, PP=14s).

### Parallel CV Analysis

| Metric | Sequential INT8 | 4-way INT8 | Delta |
|--------|----------------|------------|-------|
| ME | 219s | 216s | ~same |
| CV | 250s | 147s | **1.70x faster** |
| PP | 14s | 13s | ~same |
| **Wall** | **486s** | **376s** | **1.29x faster** |
| **$/genome** | **$1.04** | **$0.80** | **23% cheaper** |

**Why 1.70x (not 2-2.5x like 32-vCPU platforms):** On 32-vCPU machines,
sequential CV wastes threads beyond the ~16-thread GEMM saturation point.
4-way parallel recovers those wasted threads. On 16 vCPU, sequential CV
already runs near optimal (16 threads = saturation). Splitting into 4×4
puts each worker below the saturation point (4 threads), so parallelism
compensates for per-worker slowdown rather than recovering wasted capacity.
Net effect: 1.70x vs 1.90-2.47x on 32-vCPU machines.

### Updated Cost Ranking (All Platforms, Best Config)

| Rank | Platform | vCPU | $/hr | Config | Wall | $/genome |
|------|----------|------|------|--------|------|----------|
| **1** | **Oracle A1 (Altra)** | **16** | **$0.16** | **INT8 4-way + jemalloc** | **376s** | **$0.80** |
| 2 | Oracle A1 (Altra) | 16 | $0.16 | INT8 sequential + jemalloc | 486s | $1.04 |
| 3 | Oracle A2 4-way (projected) | 32 | $0.64 | INT8 4-way + jemalloc | ~250s | ~$2.14 |
| 4 | Oracle A2 (AmpereOne) | 16 | $0.32 | INT8 sequential + jemalloc | 544s | $2.32 |
| 5 | Graviton4 4-way (projected) | 32 | $1.36 | INT8/BF16 4-way | ~172s | ~$3.13 |
| 6 | Graviton4 (c8g.4xlarge) | 16 | $0.68 | INT8 ONNX | 366s | $3.33 |
| 7 | Graviton3 4-way (projected) | 32 | $1.15 | BF16 4-way | ~218s | ~$3.35 |
| — | Google x86 (official) | 96 | $3.81 | — | — | $5.01 |

**Oracle A1 + parallel CV at $0.80/genome is 6.3x cheaper than Google's official x86 reference.**

---

## Capacity and Availability

Oracle A1 instances have historically been capacity-constrained in Frankfurt
(previous attempts hit "Out of host capacity"). This benchmark ran on a paid
instance (not free tier) with no capacity issues at time of testing (2026-03-07).
Users should plan for potential capacity delays, especially in high-demand regions.

**A1 instance cost for this benchmark session (~2.5 hours): ~$0.40.**

---

## Data Quality Note

An initial batch of 12 runs was invalidated because a rogue Docker container
(`:optimized` image, single-shard, from an earlier session) was running
concurrently throughout the entire batch. This inflated wall times by ~12%
(e.g., 571s contaminated vs 509s clean for INT8 OFF). The contamination was
discovered when a clean replacement run (502s) was significantly faster than
the contaminated mean (571s). All results in this document are from clean runs
performed after the rogue container was identified and killed.

---

## Methodology

- Cost formula: `$/genome = chr20_wall_s x 48.1 / 3600 x $/hr`
- 48.1x scaling factor: chr20 is 2.08% of the human genome
- WGS extrapolation has ~15-20% uncertainty (chr20 variant density may differ)
- ME/CV/PP times extracted from `time` wrapper output of `run_deepvariant`
- N >= 4 runs: reported as verified (no asterisk)
- N < 4 runs: flagged with \* (wider confidence interval)
- Interleaved jemalloc ablation runs eliminate cache-warming bias
