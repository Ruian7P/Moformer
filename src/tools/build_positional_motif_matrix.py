#!/usr/bin/env python
"""Build global + positional promoter motif count features from motif hits.

Input hits must contain one row per motif hit with at least:
  gene_id, motif, start, end
where start/end are coordinates relative to the promoter sequence.

Output columns are:
  motif                     global count across the full promoter
  motif__bin0 ... __binK    count in each promoter bin when observed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build positional motif count matrix from per-hit motif table.")
    p.add_argument("--hits", required=True, help="Per-hit motif table, TSV by default")
    p.add_argument("--out", required=True, help="Output gene x motif feature TSV")
    p.add_argument("--sep", default="\t")
    p.add_argument("--gene-col", default="gene_id")
    p.add_argument("--motif-col", default="motif")
    p.add_argument("--start-col", default="start")
    p.add_argument("--end-col", default="end")
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--promoter-len", type=int, default=2000)
    p.add_argument("--fill-genes-from", default="data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv")
    p.add_argument("--fill-genes-col", default="gene_id")
    p.add_argument("--min-count", type=int, default=0, help="Drop motif columns with total count below this value. 0 disables filtering.")
    return p.parse_args()


def read_gene_ids(path: str, col: str) -> list[str]:
    if not path:
        return []
    df = pd.read_csv(path, usecols=[col])
    return df[col].astype(str).tolist()


def main():
    args = parse_args()
    hits = pd.read_csv(args.hits, sep=args.sep)
    required = [args.gene_col, args.motif_col, args.start_col, args.end_col]
    missing = [c for c in required if c not in hits.columns]
    if missing:
        raise ValueError(f"Missing required columns in hits table: {missing}")

    hits = hits[required].copy()
    hits[args.gene_col] = hits[args.gene_col].astype(str)
    hits[args.motif_col] = hits[args.motif_col].astype(str)
    hits[args.start_col] = pd.to_numeric(hits[args.start_col], errors="coerce")
    hits[args.end_col] = pd.to_numeric(hits[args.end_col], errors="coerce")
    hits = hits.dropna(subset=[args.start_col, args.end_col])

    if args.min_count > 0:
        motif_totals = hits[args.motif_col].value_counts()
        keep = set(motif_totals[motif_totals >= args.min_count].index)
        hits = hits[hits[args.motif_col].isin(keep)]

    gene_ids = read_gene_ids(args.fill_genes_from, args.fill_genes_col)
    if not gene_ids:
        gene_ids = sorted(hits[args.gene_col].unique())

    # Global motif counts.
    global_counts = pd.crosstab(hits[args.gene_col], hits[args.motif_col])

    # Positional motif counts. Coordinates are relative to promoter sequence.
    bin_width = float(args.promoter_len) / float(args.n_bins)
    mid = (hits[args.start_col].astype(float) + hits[args.end_col].astype(float)) / 2.0
    bin_id = np.floor(mid / bin_width).astype(int).clip(0, args.n_bins - 1)
    hits["_bin"] = bin_id
    hits["_motif_bin"] = hits[args.motif_col] + "__bin" + hits["_bin"].astype(str)
    bin_counts = pd.crosstab(hits[args.gene_col], hits["_motif_bin"])

    motifs = sorted(global_counts.columns.astype(str))
    pieces = []
    for motif in motifs:
        cols = [motif]
        bin_cols = [f"{motif}__bin{i}" for i in range(args.n_bins) if f"{motif}__bin{i}" in bin_counts.columns]
        block = pd.concat(
            [global_counts[[motif]]] + ([bin_counts[bin_cols]] if bin_cols else []),
            axis=1,
        )
        block = block.reindex(columns=cols + bin_cols)
        pieces.append(block)

    out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=gene_ids)
    out = out.reindex(gene_ids).fillna(0.0)
    out.index.name = args.fill_genes_col

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t")
    print(f"saved: {args.out}")
    print(f"n_genes: {out.shape[0]}")
    print(f"n_features: {out.shape[1]}")
    print(f"n_global_motifs: {len(motifs)}")
    print(f"n_bins: {args.n_bins}")


if __name__ == "__main__":
    main()
