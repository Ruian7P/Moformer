# Moformer: Predicting Promoter Activity from Promoter Motifs

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
conda install -c conda-forge -c bioconda gimmemotifs==0.18.0 -y
```


## Data

The main files used by Moformer are:

```text
data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv
data/leave_chrom_out_crossvalidation_split_18377genes.csv
data/promoter_2k_motif_counts_all_pos4plusglobal.tsv
```

The expression table contains 18,377 protein-coding genes with K562 expression labels and promoter sequences. The split table contains chromosome-based train/validation/test splits. The motif table contains promoter motif-count features from four 500 bp promoter bins plus one global promoter-count channel.

Step 1: download the required data files.

```bash
bash download_data.sh
genomepy install hg38 --annotation
python src/tools/export_promoter_fasta.py
```

Step 2: generate promoter motif features with GimmeMotifs.

```bash
bash src/tools/gimmemotifs_scan.sh
```

The motif scan uses the `gimme.vertebrate.v5.0` motif database and an FPR cutoff of `0.01`. Before running it on a new machine, edit the hg38 genome FASTA path inside `src/tools/gimmemotifs_scan.sh`.

## Train Moformer

Train the main Moformer-P classification model on the holdout split:

```bash
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

This runs single motif-family ablation on the holdout test set. It masks motif families, recomputes prediction performance, and writes CSV summaries and top motif figures to:

```text
results/motif_combo_occlusion/
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
Test split for interpretation: holdout test split
```


## Acknowledgement
We greatly appreciate the contributions of [EPInformer](https://github.com/pinellolab/EPInformer). This remarkable repository has significantly benefited our work.
