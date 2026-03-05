#!/bin/bash
set -euo pipefail

# Small benchmark for DeepVariant ARM64 — runs a 5MB region of chr20.
# Designed for memory-constrained instances (16GB RAM).
#
# Usage:
#   bash scripts/benchmark_small.sh [--onnx] [--docker-image IMAGE]
#
# Results from GCP t2a-standard-8 (8 vCPU Ampere Altra, 32GB RAM):
#   make_examples: ~56s (5MB region, 8 shards, ~2324 examples)
#   call_variants: ~32s (batch_size=256, 2.2s/100 examples)
#   postprocess_variants: ~5s
#   Total: ~1m33s

DOCKER_IMAGE="deepvariant-arm64"
USE_ONNX=false
DATA_DIR="${HOME}/benchmark-data"
OUTPUT_DIR="${DATA_DIR}/output"
REGION="chr20:10000000-15000000"
NUM_SHARDS=8
# TF threads — use all available by default.
TF_THREADS=8
# Docker memory limit prevents TF from over-allocating.
# TF's allocator grabs all available RAM; capping at 28GB keeps ~4GB for OS.
DOCKER_MEM="28g"
# Batch size for call_variants — 256 keeps peak memory reasonable.
BATCH_SIZE=256

while [[ $# -gt 0 ]]; do
  case $1 in
    --onnx) USE_ONNX=true; shift ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; OUTPUT_DIR="${DATA_DIR}/output"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --shards) NUM_SHARDS="$2"; shift 2 ;;
    --tf-threads) TF_THREADS="$2"; shift 2 ;;
    --docker-mem) DOCKER_MEM="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "========== DeepVariant ARM64 Small Benchmark =========="
echo "Docker image: ${DOCKER_IMAGE}"
echo "Region: ${REGION}"
echo "Shards: ${NUM_SHARDS}"
echo "TF intra-op threads: ${TF_THREADS}"
echo "Docker memory limit: ${DOCKER_MEM}"
echo "Batch size: ${BATCH_SIZE}"
echo "ONNX: ${USE_ONNX}"
echo ""

# System info
echo "========== System Info"
echo "Architecture: $(uname -m)"
echo "Cores: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Swap: $(free -h | awk '/^Swap:/ {print $2}')"
echo ""

# Data download (same as benchmark_arm64.sh)
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

NCBI_REF_DIR="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids"
GCS_DIR="https://storage.googleapis.com/deepvariant/case-study-testdata"
REF="GRCh38_no_alt_analysis_set.fasta"
BAM="HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"

echo "========== Checking test data"
if [[ ! -f "${DATA_DIR}/${REF}" ]]; then
  echo "  Downloading reference FASTA (compressed, ~900MB)..."
  curl -s "${NCBI_REF_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz" \
    | gunzip > "${DATA_DIR}/${REF}"
fi
if [[ ! -f "${DATA_DIR}/${REF}.fai" ]]; then
  echo "  Downloading reference FASTA index..."
  curl -s -o "${DATA_DIR}/${REF}.fai" \
    "${NCBI_REF_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai"
fi
for FILE in "${BAM}" "${BAM}.bai"; do
  if [[ ! -f "${DATA_DIR}/${FILE}" ]]; then
    echo "  Downloading ${FILE}..."
    curl -s -o "${DATA_DIR}/${FILE}" "${GCS_DIR}/${FILE}"
  else
    echo "  Cached: ${FILE}"
  fi
done

# Clean previous output
rm -rf "${OUTPUT_DIR}/intermediate"
mkdir -p "${OUTPUT_DIR}"

CV_EXTRA_ARGS="--batch_size=${BATCH_SIZE}"
if [[ "${USE_ONNX}" == "true" ]]; then
  CV_EXTRA_ARGS="${CV_EXTRA_ARGS},--use_onnx=true,--onnx_model=/opt/models/wgs/model.onnx"
  echo "Using ONNX Runtime for inference"
fi

echo ""
echo "========== Running DeepVariant (${REGION})"
START_TIME=$(date +%s)

docker run --rm \
  --memory="${DOCKER_MEM}" \
  -v "${DATA_DIR}:/data" \
  -v "${OUTPUT_DIR}:/output" \
  -e TF_NUM_INTRAOP_THREADS="${TF_THREADS}" \
  -e TF_NUM_INTEROP_THREADS=1 \
  -e OMP_NUM_THREADS="${TF_THREADS}" \
  "${DOCKER_IMAGE}" \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref="/data/${REF}" \
    --reads="/data/${BAM}" \
    --output_vcf=/output/HG003_arm64.vcf.gz \
    --output_gvcf=/output/HG003_arm64.g.vcf.gz \
    --num_shards="${NUM_SHARDS}" \
    --regions="${REGION}" \
    --intermediate_results_dir=/output/intermediate \
    --call_variants_extra_args="${CV_EXTRA_ARGS}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========== Results"
echo "Total wall time: ${MINUTES}m ${SECONDS}s (${ELAPSED}s)"
echo "Region: ${REGION}"
echo "TF threads: ${TF_THREADS}"
echo "ONNX: ${USE_ONNX}"

# Quick variant count
echo ""
echo "========== Output"
docker run --rm -v "${OUTPUT_DIR}:/output" "${DOCKER_IMAGE}" \
  bash -c "zcat /output/HG003_arm64.vcf.gz | grep -v '^#' | wc -l" \
  2>/dev/null && echo " variants called" || true

echo ""
echo "========== Done =========="
