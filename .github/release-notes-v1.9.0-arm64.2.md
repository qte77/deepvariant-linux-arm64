## DeepVariant ARM64 v1.9.0-arm64.2

Compatible with google/deepvariant v1.9.0. Native ARM64 Linux build with
hardware-accelerated inference.

### What's new in this release

**Parallel call_variants** — breaks through the CV bottleneck at 32 vCPU.
4 independent workers process 8 shards each, giving 1.9-2.5x CV speedup.
Projected: Oracle A2 ~$2.14/genome, Graviton4 ~$3.13/genome.
See `scripts/benchmark_parallel_cv.sh`.

**32-vCPU benchmarks** — Graviton3 (c7g.8xlarge), Graviton4 (c8g.8xlarge),
and Oracle A2 (16 OCPU). BF16 and INT8 converge at 32 vCPU (~232s on
Graviton4) because CV rate doesn't improve beyond 16 ORT threads.

**jemalloc allocator integration** — reduces glibc malloc contention under
parallel shards. Verified 14-17% make_examples speedup on Graviton3 and
AmpereOne. Enable with `-e DV_USE_JEMALLOC=1`.

**CPU-aware autoconfig** — automatically selects backend, thread counts, and
safety settings for your ARM64 CPU. Run `scripts/autoconfig.sh` or enable
with `-e DV_AUTOCONFIG=1`.

**Verified benchmark data** — all cost/performance numbers include run counts
and asterisks for N<4. Oracle A2 INT8+jemalloc verified at N=4.

### Recommended configurations

| Use case | Platform | Config | $/genome |
|---|---|---|---|
| Cheapest (projected) | Oracle A2 (16 OCPU, 32 vCPU) | INT8 + jemalloc + 4-way CV | ~$2.14 |
| Cheapest (sequential, verified) | Oracle A2 (8 OCPU, 16 vCPU) | INT8 + jemalloc | $2.32 |
| Fastest ARM64 (projected) | Graviton4 (c8g.8xlarge, 32 vCPU) | BF16 + jemalloc + 4-way CV | ~$3.13 |
| Fastest sequential | Graviton4 (c8g.8xlarge, 32 vCPU) | BF16 + jemalloc | $4.22* |

\*N<4 runs. Parallel CV rows are projected (measured CV + measured sequential ME/PP).

### Quick start

```bash
docker run -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  -v /path/to/data:/data --memory=28g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

### Accuracy

All backends pass GIAB HG003 gates (SNP F1 >= 0.9974, INDEL F1 >= 0.9940)
including stratified validation across homopolymers, tandem repeats, and
segmental duplications.

### Full changelog

See [CHANGELOG.md](../CHANGELOG.md) for the complete list of changes.
