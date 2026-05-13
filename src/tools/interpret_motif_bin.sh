#!/usr/bin/env bash
set -euo pipefail

# Example checkpoints / files (replace if needed)
CKPT_MOTIF4="/home/ruian7p/Projects/EPInformer/results/Moformer-P-pos4-cls/fold_enformer_best_Moformer-P.K562.cls.thr0.0.seed42.promoter_2k_motif_counts_all_pos4plusglobal.zscore.mtok5_checkpoint.pt"
MOTIF4="/home/ruian7p/Projects/EPInformer/data/promoter_2k_motif_counts_all_pos4plusglobal.tsv"
MOTIF8="/home/ruian7p/Projects/EPInformer/data/promoter_2k_motif_counts_all_pos8plusglobal.tsv"
TOP10CSV="/home/ruian7p/Projects/EPInformer/results/motif_combo_occlusion/top10.csv"
TOP10_LIST="GM.5.0.C2H2_ZF_Homeodomain,GM.5.0.CBF_NF-Y,GM.5.0.C2H2_ZF,GM.5.0.Homeodomain_Paired_box,GM.5.0.HSF,GM.5.0.Ets,GM.5.0.STAT,GM.5.0.GATA,GM.5.0.p53,GM.5.0.bZIP,GM.5.0.Nuclear_receptor,GM.5.0.AP-2"

# 1) Bin-level impact: mask all motifs in each bin (4-bin example)
python -u src/interpret_motif_bin_impact.py \
  --checkpoint "$CKPT_MOTIF4" \
  --motif_path "$MOTIF4" \
  --motif_zscore --task cls --fold enformer --split test \
  --mode all_bins

# 1) Bin-level impact: mask all motifs in each bin (8-bin example)
# python -u src/interpret_motif_bin_impact.py \
#   --checkpoint "$CKPT_MOTIF8" \
#   --motif_path "$MOTIF8" \
#   --motif_zscore --task cls --fold enformer --split test \
#   --mode all_bins

# 2) Top-10 motif distribution over bins
python -u src/interpret_motif_bin_distribution.py \
  --motif_path "$MOTIF4" \
  --fold enformer --split test \
  --family_level \
  --top_motif_list "$TOP10_LIST" --top_k 12

# 3) Per-bin impact: mask top-10 motifs inside each bin
python -u src/interpret_motif_bin_impact.py \
  --checkpoint "$CKPT_MOTIF4" \
  --motif_path "$MOTIF4" \
  --motif_zscore --task cls --fold enformer --split test \
  --active_only \
  --eval_on_active_subset \
  --mode top_motifs_by_bin \
  --family_level \
  --top_mask_strategy per_motif \
  --top_active_scope motif_any_bin \
  --top_motif_list "$TOP10_LIST" --top_k 12

# Optional: run both analyses in one call
# python -u src/interpret_motif_bin_impact.py \
#   --checkpoint "$CKPT_MOTIF4" \
#   --motif_path "$MOTIF4" \
#   --motif_zscore --task cls --fold enformer --split test \
#   --mode both \
#   --top_motif_list "$TOP10_LIST" --top_k 10
