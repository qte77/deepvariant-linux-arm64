#!/bin/bash
# Download GIAB training data for DeepVariant EfficientNet-B3 training.
# Downloads HG001 (training) and HG003 (validation) data.
# Requires ~80GB disk space.
# Run as: bash scripts/download_training_data.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/dv-training-data}"
NPROC=$(nproc)

echo "=== Downloading GIAB Training Data ==="
echo "Data directory: ${DATA_DIR}"
echo "This will download ~80GB of data."

# 1. Reference genome (GRCh38 no-alt)
echo ""
echo ">>> [1/6] Downloading reference genome..."
if [[ ! -f "${DATA_DIR}/reference/GRCh38_no_alt_analysis_set.fasta" ]]; then
  wget -q --show-progress -O "${DATA_DIR}/reference/GRCh38_no_alt_analysis_set.fasta" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/GRCh38_no_alt_analysis_set.fasta
  wget -q --show-progress -O "${DATA_DIR}/reference/GRCh38_no_alt_analysis_set.fasta.fai" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/GRCh38_no_alt_analysis_set.fasta.fai
else
  echo "  Already exists, skipping."
fi

# 2. HG001 (NA12878) BAM — training sample
echo ""
echo ">>> [2/6] Downloading HG001 BAM for training (~50GB)..."
if [[ ! -f "${DATA_DIR}/bam/HG001.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam" ]]; then
  wget -q --show-progress -O "${DATA_DIR}/bam/HG001.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG001.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam
  wget -q --show-progress -O "${DATA_DIR}/bam/HG001.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam.bai" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG001.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam.bai
else
  echo "  Already exists, skipping."
fi

# 3. HG001 truth VCF and confident regions
echo ""
echo ">>> [3/6] Downloading HG001 truth VCF and confident regions..."
if [[ ! -f "${DATA_DIR}/truth/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" ]]; then
  wget -q --show-progress -O "${DATA_DIR}/truth/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
  wget -q --show-progress -O "${DATA_DIR}/truth/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi
  wget -q --show-progress -O "${DATA_DIR}/truth/HG001_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG001_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed
else
  echo "  Already exists, skipping."
fi

# 4. HG003 BAM — validation sample (chr20 only for speed)
echo ""
echo ">>> [4/6] Downloading HG003 BAM for validation (~4GB, chr20 only)..."
if [[ ! -f "${DATA_DIR}/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam" ]]; then
  wget -q --show-progress -O "${DATA_DIR}/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam
  wget -q --show-progress -O "${DATA_DIR}/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai
else
  echo "  Already exists, skipping."
fi

# 5. HG003 truth VCF and confident regions (for validation)
echo ""
echo ">>> [5/6] Downloading HG003 truth data..."
if [[ ! -f "${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" ]]; then
  wget -q --show-progress -O "${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
  wget -q --show-progress -O "${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi
  wget -q --show-progress -O "${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed" \
    https://storage.googleapis.com/deepvariant/case-study-testdata/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed
else
  echo "  Already exists, skipping."
fi

# 6. Verify downloads
echo ""
echo ">>> [6/6] Verifying downloads..."
MISSING=0
for f in \
  "${DATA_DIR}/reference/GRCh38_no_alt_analysis_set.fasta" \
  "${DATA_DIR}/reference/GRCh38_no_alt_analysis_set.fasta.fai" \
  "${DATA_DIR}/bam/HG001.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam" \
  "${DATA_DIR}/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam" \
  "${DATA_DIR}/truth/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
  "${DATA_DIR}/truth/HG001_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed" \
  "${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
  "${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"; do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f"
    MISSING=1
  fi
done

if [[ $MISSING -eq 0 ]]; then
  echo "  All files present."
  echo ""
  echo "=== Download complete ==="
  echo "Total size:"
  du -sh "${DATA_DIR}"
  echo ""
  echo "Next step: bash scripts/train_efficientnet_b3.sh"
else
  echo ""
  echo "ERROR: Some files are missing. Re-run this script."
  exit 1
fi
