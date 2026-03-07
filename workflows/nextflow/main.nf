#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * DeepVariant ARM64 — Nextflow workflow
 *
 * Runs the full DeepVariant pipeline (make_examples → call_variants →
 * postprocess_variants) on ARM64 Linux using the pre-built Docker image.
 * Supports single-sample WGS and WES with automatic backend selection.
 *
 * Usage:
 *   nextflow run main.nf \
 *     --bam /data/sample.bam \
 *     --ref /data/GRCh38.fasta \
 *     --outdir results/
 *
 *   # WES with capture BED
 *   nextflow run main.nf \
 *     --bam /data/sample.bam \
 *     --ref /data/GRCh38.fasta \
 *     --model_type WES \
 *     --regions /data/capture.bed \
 *     --outdir results/
 *
 *   # Batch mode (sample sheet)
 *   nextflow run main.nf \
 *     --input samples.csv \
 *     --ref /data/GRCh38.fasta \
 *     --outdir results/
 */

// ── Parameters ──────────────────────────────────────────────────────────
params.bam        = null          // Single BAM input
params.input      = null          // CSV sample sheet: sample,bam,bai
params.ref        = null          // Reference FASTA (+ .fai must exist)
params.model_type = 'WGS'        // WGS, WES, or PACBIO
params.regions    = null          // BED file or region string (e.g. chr20)
params.outdir     = 'results'
params.num_shards = 0             // 0 = auto (nproc)
params.batch_size = 256
params.memory     = '28g'         // Docker memory limit

// ── Input validation ────────────────────────────────────────────────────
if (!params.ref) { error "Missing --ref parameter (reference FASTA)" }
if (!params.bam && !params.input) {
    error "Provide either --bam (single sample) or --input (sample sheet CSV)"
}

// ── Channels ────────────────────────────────────────────────────────────
ref_fasta = file(params.ref, checkIfExists: true)
ref_fai   = file("${params.ref}.fai", checkIfExists: true)

if (params.input) {
    ch_samples = Channel
        .fromPath(params.input, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            def bam = file(row.bam, checkIfExists: true)
            def bai = row.bai ? file(row.bai, checkIfExists: true)
                              : file("${row.bam}.bai", checkIfExists: true)
            tuple(row.sample, bam, bai)
        }
} else {
    def bam_file = file(params.bam, checkIfExists: true)
    def bai_file = file("${params.bam}.bai", checkIfExists: true)
    def sample_id = bam_file.baseName.replaceAll(/\.dedup$|\.sorted$/, '')
    ch_samples = Channel.of(tuple(sample_id, bam_file, bai_file))
}

// ── Processes ───────────────────────────────────────────────────────────
process DEEPVARIANT {
    tag "${sample_id}"
    publishDir "${params.outdir}/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(bam), path(bai)
    path ref
    path ref_idx

    output:
    tuple val(sample_id), path("${sample_id}.vcf.gz"),     emit: vcf
    tuple val(sample_id), path("${sample_id}.vcf.gz.tbi"), emit: tbi
    tuple val(sample_id), path("${sample_id}.g.vcf.gz"),   emit: gvcf
    path("${sample_id}.visual_report.html"),                emit: report

    script:
    def region_arg = params.regions ? "--regions ${params.regions}" : ""
    def shards = params.num_shards > 0 ? params.num_shards : task.cpus
    """
    /opt/deepvariant/bin/run_deepvariant \\
        --model_type=${params.model_type} \\
        --ref=${ref} \\
        --reads=${bam} \\
        --output_vcf=${sample_id}.vcf.gz \\
        --output_gvcf=${sample_id}.g.vcf.gz \\
        --num_shards=${shards} \\
        --intermediate_results_dir=./intermediate \\
        --call_variants_extra_args="--batch_size=${params.batch_size}" \\
        ${region_arg}
    """
}

// ── Workflow ────────────────────────────────────────────────────────────
workflow {
    DEEPVARIANT(ch_samples, ref_fasta, ref_fai)

    DEEPVARIANT.out.vcf
        .map { sample_id, vcf -> "${sample_id}: ${vcf}" }
        .view { "Done — $it" }
}
