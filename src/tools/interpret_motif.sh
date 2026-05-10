# python -u src/interpret_motif_combo.py \
#   --checkpoint /home/ruian7p/Projects/EPInformer/results/Moformer-P-pos4-cls/fold_enformer_best_Moformer-P.K562.cls.thr0.0.seed42.promoter_2k_motif_counts_all_pos4plusglobal.zscore.mtok5_checkpoint.pt \
#   --motif_path /home/ruian7p/Projects/EPInformer/data/promoter_2k_motif_counts_all_pos4plusglobal.tsv \
#   --motif_zscore --task cls --fold enformer --split test \
#   --group_mode motif --family_level --exclude_unknown \
#   --expressed_only --expressed_threshold 0 \
#   --motif_count 3 --candidate_top_n 100 --max_combos 2000000



python -u src/interpret_motif_combo.py \
  --checkpoint /home/ruian7p/Projects/EPInformer/results/Moformer-P-pos4-cls/fold_enformer_best_Moformer-P.K562.cls.thr0.0.seed42.promoter_2k_motif_counts_all_pos4plusglobal.zscore.mtok5_checkpoint.pt \
  --motif_path /home/ruian7p/Projects/EPInformer/data/promoter_2k_motif_counts_all_pos4plusglobal.tsv \
  --motif_zscore --task cls --fold enformer --split test \
  --family_level \
  --exclude_unknown \
  --group_mode motif --motif_count 3 --candidate_top_n 100 --max_combos 2000000
