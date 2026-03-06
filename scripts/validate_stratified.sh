#!/bin/bash
# Stratified region validation for DeepVariant ARM64 INT8/BF16/FP32 accuracy.
# Runs rtg vcfeval per GIAB stratification region to detect localized accuracy
# degradation that aggregate F1 may mask (e.g., homopolymers, tandem repeats).
#
# Usage:
#   bash scripts/validate_stratified.sh \
#     --data-dir /data \
#     --vcf-int8 /data/output/int8/output.vcf.gz \
#     --vcf-bf16 /data/output/bf16/output.vcf.gz \
#     [--vcf-fp32 /data/output/fp32/output.vcf.gz]
#
# Prerequisites:
#   - rtg-tools installed and on PATH (rtg vcfeval)
#   - Reference SDF at <data-dir>/reference/GRCh38.sdf
#   - GIAB truth VCF at <data-dir>/truth/
#   - wget for downloading stratification BEDs

set -euo pipefail

# --- Parse arguments ---
DATA_DIR=""
VCF_INT8=""
VCF_BF16=""
VCF_FP32=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --vcf-int8) VCF_INT8="$2"; shift 2 ;;
    --vcf-bf16) VCF_BF16="$2"; shift 2 ;;
    --vcf-fp32) VCF_FP32="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" || -z "$VCF_INT8" || -z "$VCF_BF16" ]]; then
  echo "Usage: $0 --data-dir DIR --vcf-int8 VCF --vcf-bf16 VCF [--vcf-fp32 VCF]"
  exit 1
fi

REF_SDF="${DATA_DIR}/reference/GRCh38.sdf"
TRUTH_VCF="${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
TRUTH_BED="${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"
STRAT_DIR="${DATA_DIR}/stratification"
OUTPUT_DIR="${DATA_DIR}/stratified_validation"
CHR20_BED="${DATA_DIR}/truth/chr20.bed"

# Verify prerequisites
for f in "$REF_SDF" "$TRUTH_VCF" "$TRUTH_BED"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: Required file not found: $f"
    exit 1
  fi
done

if ! command -v rtg &>/dev/null; then
  echo "ERROR: rtg-tools not found on PATH"
  exit 1
fi

# Create chr20 BED if not exists
if [[ ! -f "$CHR20_BED" ]]; then
  echo -e "chr20\t0\t64444167" > "$CHR20_BED"
fi

# --- Download GIAB stratification BEDs ---
mkdir -p "$STRAT_DIR"

GIAB_STRAT_BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/genome-stratifications/v3.3/GRCh38@all"

# Key stratification regions for INT8 validation
declare -A STRAT_BEDS=(
  ["LowComplexity_AllHomopolymers_ge7bp"]="LowComplexity/GRCh38_AllHomopolymers_ge7bp_imperfectge11bp_slop5.bed.gz"
  ["LowComplexity_AllTandemRepeats"]="LowComplexity/GRCh38_AllTandemRepeats_201to10000bp_slop5.bed.gz"
  ["LowComplexity_SimpleRepeats"]="LowComplexity/GRCh38_SimpleRepeat_imperfecthomopolgt10_slop5.bed.gz"
  ["SegmentalDuplications"]="SegmentalDuplications/GRCh38_segdups.bed.gz"
  ["Functional_Difficult"]="FunctionalTechnicallyDifficultRegions/GRCh38_alldifficultregions.bed.gz"
)

echo "=== Downloading GIAB stratification BEDs ==="
for name in "${!STRAT_BEDS[@]}"; do
  bed_path="${STRAT_DIR}/${name}.bed.gz"
  if [[ ! -f "$bed_path" ]]; then
    echo "  Downloading ${name}..."
    wget -q -O "$bed_path" "${GIAB_STRAT_BASE}/${STRAT_BEDS[$name]}" || {
      echo "  WARNING: Failed to download ${name}, skipping"
      rm -f "$bed_path"
    }
  else
    echo "  ${name} already downloaded"
  fi
done

# --- Run rtg vcfeval per region per backend ---
run_vcfeval() {
  local label="$1"
  local vcf="$2"
  local region_name="$3"
  local region_bed="$4"
  local out_dir="${OUTPUT_DIR}/${label}/${region_name}"

  mkdir -p "$out_dir"

  # Intersect region BED with chr20 BED (we only have chr20 calls)
  local intersected_bed="${out_dir}/chr20_intersected.bed"
  bedtools intersect -a "$region_bed" -b "$CHR20_BED" > "$intersected_bed" 2>/dev/null || {
    # If bedtools not available, use the region BED directly
    # rtg vcfeval will just find fewer variants outside chr20
    cp "$region_bed" "$intersected_bed"
  }

  # Skip if intersection is empty
  if [[ ! -s "$intersected_bed" ]]; then
    echo "  SKIP: ${label}/${region_name} — no overlap with chr20"
    return
  fi

  rm -rf "${out_dir}/rtg_output"
  rtg vcfeval \
    --baseline "$TRUTH_VCF" \
    --calls "$vcf" \
    --template "$REF_SDF" \
    --evaluation-regions "$intersected_bed" \
    --bed-regions "$CHR20_BED" \
    --output "${out_dir}/rtg_output" \
    --ref-overlap 2>/dev/null || {
      echo "  ERROR: rtg vcfeval failed for ${label}/${region_name}"
      return
    }

  # Extract F1 scores
  if [[ -f "${out_dir}/rtg_output/summary.txt" ]]; then
    echo "  ${label}/${region_name}:"
    grep -E "SNP|INDEL" "${out_dir}/rtg_output/summary.txt" | while read -r line; do
      echo "    $line"
    done
  fi
}

echo ""
echo "=== Running stratified validation ==="
echo "  Reference: $REF_SDF"
echo "  Truth: $TRUTH_VCF"
echo ""

# Run for each backend × each region
BACKENDS=()
BACKEND_LABELS=()

BACKENDS+=("$VCF_INT8"); BACKEND_LABELS+=("INT8")
BACKENDS+=("$VCF_BF16"); BACKEND_LABELS+=("BF16")
if [[ -n "$VCF_FP32" ]]; then
  BACKENDS+=("$VCF_FP32"); BACKEND_LABELS+=("FP32")
fi

# Also run aggregate (no region restriction) for comparison
for i in "${!BACKENDS[@]}"; do
  label="${BACKEND_LABELS[$i]}"
  vcf="${BACKENDS[$i]}"

  echo "--- ${label} ---"

  # Aggregate chr20
  agg_dir="${OUTPUT_DIR}/${label}/aggregate_chr20"
  mkdir -p "$agg_dir"
  rm -rf "${agg_dir}/rtg_output"
  rtg vcfeval \
    --baseline "$TRUTH_VCF" \
    --calls "$vcf" \
    --template "$REF_SDF" \
    --bed-regions "$CHR20_BED" \
    --output "${agg_dir}/rtg_output" \
    --ref-overlap 2>/dev/null
  echo "  ${label}/aggregate_chr20:"
  grep -E "SNP|INDEL" "${agg_dir}/rtg_output/summary.txt" | while read -r line; do
    echo "    $line"
  done

  # Per-region
  for name in "${!STRAT_BEDS[@]}"; do
    bed_path="${STRAT_DIR}/${name}.bed.gz"
    if [[ -f "$bed_path" ]]; then
      run_vcfeval "$label" "$vcf" "$name" "$bed_path"
    fi
  done

  echo ""
done

# --- Summary comparison table ---
echo "=== SUMMARY: INT8 vs BF16 F1 Comparison ==="
echo ""
printf "%-45s  %12s  %12s  %12s  %12s\n" "Region" "INT8 SNP F1" "BF16 SNP F1" "INT8 IND F1" "BF16 IND F1"
printf "%-45s  %12s  %12s  %12s  %12s\n" "------" "-----------" "-----------" "-----------" "-----------"

extract_f1() {
  local summary="$1"
  local variant_type="$2"
  if [[ -f "$summary" ]]; then
    grep "$variant_type" "$summary" | awk '{print $NF}' | head -1
  else
    echo "N/A"
  fi
}

for region in "aggregate_chr20" "${!STRAT_BEDS[@]}"; do
  int8_snp=$(extract_f1 "${OUTPUT_DIR}/INT8/${region}/rtg_output/summary.txt" "SNP")
  bf16_snp=$(extract_f1 "${OUTPUT_DIR}/BF16/${region}/rtg_output/summary.txt" "SNP")
  int8_indel=$(extract_f1 "${OUTPUT_DIR}/INT8/${region}/rtg_output/summary.txt" "INDEL")
  bf16_indel=$(extract_f1 "${OUTPUT_DIR}/BF16/${region}/rtg_output/summary.txt" "INDEL")
  printf "%-45s  %12s  %12s  %12s  %12s\n" "$region" "$int8_snp" "$bf16_snp" "$int8_indel" "$bf16_indel"
done

echo ""
echo "Gate: All INT8 F1 scores must be within 0.005 of BF16."
echo "If any region shows >0.005 degradation, INT8 cannot be used in production without caveat."
