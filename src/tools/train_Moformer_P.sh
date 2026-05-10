# python -u src/train_Moformer.py \
#   --model_type Moformer-P \
#   --fold enformer \
#   --motif_path /home/ruian7p/Projects/EPInformer/data/motif_bench/promoter_2k_motif_counts_atac_unionDHS_pos8plusglobal_min50.tsv \
#   --motif_zscore \
#   --early_stop_patience 10 \
#   --seed 42 \
#   | tee logs/train_Moformer_P_atac_pos8plusglobal_min50_10.log

# python -u src/train_Moformer.py \
#   --model_type Moformer-P \
#   --fold enformer \
#   --motif_path /home/ruian7p/Projects/EPInformer/data/promoter_2k_motif_hits.tsv \
#   --early_stop_patience 10 \
#   --seed 42 \
#   | tee logs/train_Moformer_P_count_10.log


python -u src/train_Moformer.py \
  --model_type Moformer-P \
  --fold enformer \
  --motif_path /home/ruian7p/Projects/EPInformer/data/promoter_2k_motif_counts_all_pos4plusglobal.tsv \
  --motif_zscore \
  --motif_multitoken \
  --motif_multitoken_include_global \
  --early_stop_patience 10 \
  --seed 42 \
  | tee logs/train_Moformer_P_all_pos4plusglobal_10.log
