#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FA="data/promoter_2k.fa"
GENOME_FA="/home/ruian7p/Projects/puffin/resources/hg38.fa"
MOTIF_DB="gimme.vertebrate.v5.0"
# 16 threads is often unstable for large scans in multiprocessing pools.
# You can override: NTHREADS=8 bash src/tools/pipeline.sh
NTHREADS="${NTHREADS:-6}"
FPR=0.01
LOG_DIR="logs/motif_scan"
mkdir -p "$LOG_DIR"

# Keep threaded numeric libs from oversubscribing worker processes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Replace this with a K562 ATAC peak BED when available.
# This DNase BED is a reasonable open-chromatin proxy.
PEAKS_BED="data/K562_ABC_EGLinks/DNase_ENCFF257HEE_Neighborhoods/EnhancerList.bed"

EXPR_ANNOT="data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv"

OUT_COUNT="data/promoter_2k_motif_counts.tsv"
OUT_SCORE="data/promoter_2k_motif_scores.tsv"
OUT_HITS_REL_BED="data/promoter_2k_hits_relative.bed"
OUT_HITS_REL_TSV="data/promoter_2k_hits_relative.tsv"
OUT_POS4_GLOBAL="data/promoter_2k_motif_counts_all_pos4plusglobal.tsv"
OUT_FILTERED_COUNT="data/promoter_2k_motif_counts_openchrom.tsv"

if ! command -v gimme >/dev/null 2>&1; then
  echo "ERROR: gimme command not found in current environment."
  echo "Activate the env where gimmemotifs is installed, e.g.: conda activate epinformer"
  exit 1
fi

run_gimme_scan() {
  local mode_flag="$1"
  local out_file="$2"
  local step_name="$3"
  local log_file="$LOG_DIR/${step_name}.log"
  local tmp_out="${out_file}.tmp"

  echo "[$step_name] gimme scan with NTHREADS=${NTHREADS}"
  if gimme scan "$FA" \
      -p "$MOTIF_DB" \
      "$mode_flag" \
      -g "$GENOME_FA" \
      -f "$FPR" \
      -N "$NTHREADS" \
      > "$tmp_out" 2> "$log_file"; then
    mv "$tmp_out" "$out_file"
    return 0
  fi

  echo "[$step_name] failed with NTHREADS=${NTHREADS}; retrying with NTHREADS=1"
  if gimme scan "$FA" \
      -p "$MOTIF_DB" \
      "$mode_flag" \
      -g "$GENOME_FA" \
      -f "$FPR" \
      -N 1 \
      > "$tmp_out" 2>> "$log_file"; then
    mv "$tmp_out" "$out_file"
    return 0
  fi

  rm -f "$tmp_out"
  echo "[$step_name] failed again. See log: $log_file"
  return 1
}

echo "[1/6] Scanning motifs to count matrix"
run_gimme_scan -t "$OUT_COUNT" "step1_count"

echo "[2/6] Scanning motifs to score matrix"
run_gimme_scan -T "$OUT_SCORE" "step2_score"

echo "[3/6] Scanning motifs to per-hit relative BED"
run_gimme_scan -b "$OUT_HITS_REL_BED" "step3_hits_bed"

echo "[4/6] Converting BED hits to normalized TSV"
awk 'BEGIN{OFS="\t"; print "gene_id\tmotif\tstart\tend\tscore\tstrand"} !/^#/ {print $1,$4,$2,$3,$5,$6}' \
  "$OUT_HITS_REL_BED" > "$OUT_HITS_REL_TSV"

echo "[5/6] Building all-motif 4-bin + global count matrix"
python src/tools/build_positional_motif_matrix.py \
  --hits "$OUT_HITS_REL_TSV" \
  --out "$OUT_POS4_GLOBAL" \
  --n-bins 4 \
  --promoter-len 2000 \
  --fill-genes-from "$EXPR_ANNOT" \
  --fill-genes-col gene_id \
  --min-count 0

echo "[6/6] Filtering motif hits by open chromatin and rebuilding gene x motif count matrix"
python src/tools/build_openchrom_motif_matrix.py \
  --motif-hits "$OUT_HITS_REL_TSV" \
  --hits-sep $'\t' \
  --peaks-bed "$PEAKS_BED" \
  --out "$OUT_FILTERED_COUNT" \
  --gene-col gene_id \
  --motif-col motif \
  --start-col start \
  --end-col end \
  --score-col score \
  --agg count \
  --fill-genes-from "$EXPR_ANNOT" \
  --fill-genes-col gene_id \
  --relative-coords \
  --promoter-annot "$EXPR_ANNOT" \
  --promoter-sep "," \
  --promoter-gene-col gene_id \
  --promoter-chrom-col chrom \
  --promoter-start-col start \
  --promoter-end-col end \
  --promoter-strand-col strand \
  --promoter-upstream 1500 \
  --promoter-downstream 500

echo "Done."
echo "Main Moformer-P training matrix: $OUT_POS4_GLOBAL"
echo "Optional open-chromatin filtered matrix: $OUT_FILTERED_COUNT"
echo "If a scan step failed, check logs under: $LOG_DIR"
