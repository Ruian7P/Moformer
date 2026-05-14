# gimme scan data/promoter_2k.fa \
#   -p gimme.vertebrate.v5.0 \
#   -T \
#   -g /home/ruian7p/Projects/puffin/resources/hg38.fa \
#   -f 0.01 \
#   -N 16 \
#   > data/promoter_2k_motif_scores.tsv

gimme scan data/promoter_2k.fa \
  -p gimme.vertebrate.v5.0 \
  -t \
  -g hg38.fa \
  -f 0.01 \
  -N 16 \
  > data/promoter_2k_motif_hits.tsv