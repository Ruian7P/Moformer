# python -u src/train_EPInformer_cls.py \
#   --fold all \
#   --seed 42 \
#   --expr_threshold 0.0 \
#   --early_stop_patience 10 \
#   | tee logs/train_EPInformer_promoter_cls.log

python -u src/train_Moformer_cls.py \
  --fold all \
  --seed 42 \
  --expr_threshold 0.0 \
  --motif_path /home/ruian7p/Projects/EPInformer/data/promoter_2k_motif_counts_all_pos4plusglobal.tsv \
  --motif_zscore \
  --motif_multitoken \
  --motif_multitoken_include_global \
  --early_stop_patience 10 \
  | tee logs/train_Moformer_P_pos4_cls_2.log
