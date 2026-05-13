# Moformer: promoter activity prediction from promoter motifs

Moformer is a promoter-only model for predicting whether a gene is expressed from transcription-factor motif features in its promoter. The main experiment in this repository uses K562 gene-expression labels and motif-count features from a 2 kb promoter window around the TSS.

## Repository Structure

```text
src/train_Moformer_cls.py              # Train Moformer classification models
src/interpret_motif_combo.py           # Single motif / motif-combination ablation
src/interpret_motif_bin_impact.py      # Bin-level and per-motif bin ablation
src/interpret_motif_bin_distribution.py# Motif hit distribution across promoter bins
src/tools/gimmemotifs_scan.sh          # Example GimmeMotifs scan command
src/tools/interpret_motif.sh           # Run motif-family ablation analysis
src/tools/interpret_motif_bin.sh       # Run bin-level motif interpretation analyses
data/                                # Expression labels, split files, and motif features
results/                             # Model checkpoints and analysis outputs
logs/                                # Training logs
```

## Environment Setup

Create a conda environment with Python, PyTorch, scientific Python packages, and GimmeMotifs.

```bash
conda create -n moformer python=3.10 -y
conda activate moformer

# GPU PyTorch. Adjust the CUDA version if needed.
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Core Python dependencies.
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm h5py pyfaidx pyranges kipoiseq openpyxl

# Motif scanning toolkit.
conda install -c conda-forge -c bioconda gimmemotifs -y
```

Check that the key commands are available:

```bash
python -c "import torch, pandas, sklearn; print(torch.__version__)"
gimme --help
```

## Data

The main files used by Moformer are:

```text
data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv
data/leave_chrom_out_crossvalidation_split_18377genes.csv
data/promoter_2k_motif_counts_all_pos4plusglobal.tsv
```

The expression table contains 18,377 protein-coding genes with K562 expression labels and promoter sequences. The split table contains chromosome-based train/validation/test splits. The motif table contains promoter motif-count features from four 500 bp promoter bins plus one global promoter-count channel.

If the large EPInformer training files are missing, download them first. The expression table, split table, and motif-count table should be placed under `data/` with the filenames shown above:

```bash
bash download_data.sh
```

## Motif Feature Generation

If `data/promoter_2k_motif_counts_all_pos4plusglobal.tsv` already exists, this step can be skipped.

To scan promoter sequences with GimmeMotifs, use:

```bash
bash src/tools/gimmemotifs_scan.sh
```

The scan command uses the `gimme.vertebrate.v5.0` motif database and an FPR cutoff of `0.01`. You may need to edit the hg38 genome FASTA path inside `src/tools/gimmemotifs_scan.sh` before running it.

The main model expects a gene-by-feature TSV where rows are Ensembl gene IDs and columns are motif features. For the 4-bin model, columns use bin-specific motif counts plus global motif counts.

## Train Moformer

Train the main Moformer-P classification model on the Enformer-style holdout split:

```bash
mkdir -p logs results

python -u src/train_Moformer_cls.py \
  --model_type Moformer-P \
  --fold enformer \
  --motif_path data/promoter_2k_motif_counts_all_pos4plusglobal.tsv \
  --motif_zscore \
  --motif_multitoken \
  --motif_multitoken_include_global \
  --early_stop_patience 10 \
  --seed 42 \
  | tee logs/train_Moformer_P_pos4_cls_seed42.log
```

Train all chromosome-based folds:

```bash
python -u src/train_Moformer_cls.py \
  --model_type Moformer-P \
  --fold all \
  --motif_path data/promoter_2k_motif_counts_all_pos4plusglobal.tsv \
  --motif_zscore \
  --motif_multitoken \
  --motif_multitoken_include_global \
  --early_stop_patience 10 \
  --seed 42 \
  | tee logs/train_Moformer_P_pos4_cls_all_seed42.log
```

Outputs are saved under:

```text
results/Moformer-P-cls/
```

Each fold writes a best checkpoint, prediction CSV, and a summary CSV containing ACC, AUROC, and AUPRC.

## Motif Ablation Analysis

After training, set the checkpoint path in `src/tools/interpret_motif.sh` if needed, then run:

```bash
bash src/tools/interpret_motif.sh
```

This runs single motif-family ablation on the Enformer-style test set. It masks motif families, recomputes prediction performance, and writes CSV summaries and top motif figures to:

```text
results/motif_combo_occlusion/
```

To run the command manually:

```bash
python -u src/interpret_motif_combo.py \
  --checkpoint results/Moformer-P-cls/<checkpoint>.pt \
  --motif_path data/promoter_2k_motif_counts_all_pos4plusglobal.tsv \
  --motif_zscore \
  --task cls \
  --fold enformer \
  --split test \
  --family_level \
  --exclude_unknown \
  --exclude_mixed \
  --group_mode motif \
  --motif_count 1 \
  --active_sample_n 50 \
  --active_sample_trials 10 \
  --active_sample_seed 42 \
  --min_active_genes 50
```

## Bin-Level Interpretation

After training, set `CKPT_MOTIF4` in `src/tools/interpret_motif_bin.sh` if needed, then run:

```bash
bash src/tools/interpret_motif_bin.sh
```

This script performs three analyses:

1. Mask all motif features in each promoter bin and measure performance drop.
2. Plot motif hit distributions across promoter bins for selected motif families.
3. Mask selected motif families in each bin and measure per-motif, per-bin performance drop.

Outputs are saved under:

```text
results/motif_bin_impact/
results/motif_bin_distribution/
```

## Main Model Settings

The main reported Moformer-P model uses:

```text
Cell line: K562
Task: binary expression classification
Positive label: Actual_K562 > 0
Input: four promoter-bin motif-count channels + one global motif-count channel
Motif preprocessing: z-score using training-split statistics
Architecture: multi-token Moformer-P with 5 motif tokens
Training seed: 42
Validation metric for early stopping: AUPRC
Test split for interpretation: Enformer-style holdout test split
```
