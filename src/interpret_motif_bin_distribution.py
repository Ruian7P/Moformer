import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description='Top-motif distribution across bins in selected split.'
    )
    p.add_argument('--motif_path', type=str, required=True)
    p.add_argument('--split_table', type=str, default='./data/leave_chrom_out_crossvalidation_split_18377genes.csv')
    p.add_argument('--fold', type=str, default='enformer')
    p.add_argument('--split', type=str, default='test', choices=['all', 'train', 'valid', 'test'])
    p.add_argument('--output_dir', type=str, default='./results/motif_bin_distribution')

    p.add_argument('--top_motif_csv', type=str, default='')
    p.add_argument('--top_motif_col', type=str, default='group')
    p.add_argument('--top_motif_list', type=str, default='')
    p.add_argument('--top_k', type=int, default=10)

    p.add_argument('--family_level', action='store_true')
    p.add_argument('--exclude_unknown', action='store_true')
    p.add_argument('--min_active_genes', type=int, default=0)
    p.add_argument('--eps', type=float, default=1e-9)
    return p.parse_args()


def motif_to_family(motif_name: str) -> str:
    return re.sub(r'\.\d+$', '', str(motif_name))


def pretty_motif_label(label: str) -> str:
    return re.sub(r'^GM\.5\.0\.', '', str(label))


def parse_col_bin_and_base(col_name):
    col = str(col_name)
    m = re.search(r'__bin(\d+)$', col)
    if m is None:
        return None, re.sub(r'__global$', '', col)
    b = int(m.group(1))
    base = re.sub(r'__bin\d+$', '', col)
    return b, base


def get_top_motifs(args):
    motifs = []
    if args.top_motif_list.strip():
        motifs = [x.strip() for x in args.top_motif_list.split(',') if x.strip()]
    elif args.top_motif_csv and os.path.exists(args.top_motif_csv):
        df = pd.read_csv(args.top_motif_csv)
        if args.top_motif_col in df.columns:
            raw_vals = df[args.top_motif_col].astype(str).tolist()
        elif 'combo' in df.columns:
            raw_vals = df['combo'].astype(str).tolist()
        else:
            raise KeyError(
                f'top_motif_col={args.top_motif_col} not found, and no combo column in {args.top_motif_csv}'
            )
        for v in raw_vals:
            parts = [p.strip() for p in str(v).split('+')]
            for p in parts:
                if p:
                    motifs.append(p)

    dedup = []
    seen = set()
    for m in motifs:
        k = m.lower()
        if args.exclude_unknown and 'unknown' in k:
            continue
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup[: int(args.top_k)]


def build_motif_bin_index(motif_cols, family_level=False):
    out = OrderedDict()
    for i, c in enumerate(motif_cols):
        b, base = parse_col_bin_and_base(c)
        if b is None:
            continue
        if family_level:
            base = motif_to_family(base)
        out.setdefault(base, OrderedDict())
        out[base].setdefault(b, [])
        out[base][b].append(i)
    return out


def safe_entropy(fracs, eps=1e-9):
    x = np.asarray(fracs, dtype=float)
    x = np.clip(x, eps, 1.0)
    return float(-(x * np.log2(x)).sum())


def maybe_plot_heatmap(pivot_df, out_png, title):
    if pivot_df.empty:
        return
    arr = pivot_df.values
    rows = [pretty_motif_label(x) for x in pivot_df.index.astype(str).tolist()]
    cols = [f'bin{c}' for c in pivot_df.columns.tolist()]

    plt.figure(figsize=(max(6, 0.8 * len(cols) + 2), max(4, 0.45 * len(rows) + 1.5)))
    im = plt.imshow(arr, aspect='auto', cmap='viridis')
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.xticks(np.arange(len(cols)), cols, rotation=0)
    plt.yticks(np.arange(len(rows)), rows)
    plt.xlabel('Bin')
    plt.ylabel('Motif')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    split_df = pd.read_csv(args.split_table, index_col=0)
    fold_col = f'fold_{args.fold}'
    if fold_col not in split_df.columns:
        raise KeyError(f'{fold_col} not found in split table')
    split_ids = split_df.index.tolist() if args.split == 'all' else split_df[split_df[fold_col] == args.split].index.tolist()

    motif_df = pd.read_csv(args.motif_path, sep='\t', comment='#', index_col=0, engine='python')
    motif_df = motif_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    gene_ids = [gid for gid in split_ids if gid in motif_df.index]
    if len(gene_ids) == 0:
        raise ValueError('No genes left after filtering split with motif table.')
    motif_df = motif_df.loc[gene_ids]
    motif_cols = list(motif_df.columns)
    motif_vals = motif_df.values.astype(np.float32)

    motif_bin_idx = build_motif_bin_index(motif_cols, family_level=args.family_level)
    if len(motif_bin_idx) == 0:
        raise ValueError('No __bin* columns found in motif table.')

    top_motifs = get_top_motifs(args)
    if len(top_motifs) == 0:
        raise ValueError('No top motifs found. Please provide --top_motif_csv or --top_motif_list.')

    rows = []
    bins_all = sorted({b for m in motif_bin_idx.values() for b in m.keys()})

    for motif in top_motifs:
        key = motif_to_family(motif) if args.family_level else motif
        if key not in motif_bin_idx:
            continue

        by_bin = motif_bin_idx[key]
        total_hits_all_bins = 0.0
        total_active_any = 0

        tmp_bin_stats = []
        union_active_mask = np.zeros(motif_vals.shape[0], dtype=bool)

        for b in bins_all:
            col_idxs = by_bin.get(b, [])
            if len(col_idxs) == 0:
                hits = np.zeros(motif_vals.shape[0], dtype=np.float32)
            else:
                hits = motif_vals[:, col_idxs].sum(axis=1)
            active_mask = hits > 0
            union_active_mask = union_active_mask | active_mask
            total_hits = float(hits.sum())
            active_genes = int(active_mask.sum())
            mean_hits_active = float(hits[active_mask].mean()) if active_genes > 0 else 0.0
            tmp_bin_stats.append((b, total_hits, active_genes, mean_hits_active))
            total_hits_all_bins += total_hits

        total_active_any = int(union_active_mask.sum())
        if total_active_any < int(args.min_active_genes):
            continue

        # concentration stats
        hit_fracs = []
        for b, total_hits, _, _ in tmp_bin_stats:
            frac_hits = float(total_hits / (total_hits_all_bins + args.eps))
            hit_fracs.append(frac_hits)
        top_bin_idx = int(np.argmax(hit_fracs)) if len(hit_fracs) > 0 else 0
        top_bin = tmp_bin_stats[top_bin_idx][0] if len(tmp_bin_stats) > 0 else -1
        top_bin_frac = float(hit_fracs[top_bin_idx]) if len(hit_fracs) > 0 else float('nan')
        hit_entropy = safe_entropy(hit_fracs, eps=args.eps)

        for (b, total_hits, active_genes, mean_hits_active), frac_hits in zip(tmp_bin_stats, hit_fracs):
            rows.append(
                {
                    'motif': motif,
                    'motif_key_used': key,
                    'bin': b,
                    'total_hits_bin': total_hits,
                    'active_genes_bin': active_genes,
                    'active_gene_frac_bin_over_all_genes': float(active_genes / max(1, motif_vals.shape[0])),
                    'active_gene_frac_bin_over_motif_active_genes': float(active_genes / max(1, total_active_any)),
                    'mean_hits_per_active_gene_bin': mean_hits_active,
                    'hit_frac_within_motif': frac_hits,
                    'motif_active_genes_any_bin': total_active_any,
                    'motif_total_hits_all_bins': total_hits_all_bins,
                    'motif_top_bin_by_hits': top_bin,
                    'motif_top_bin_hit_frac': top_bin_frac,
                    'motif_hit_entropy': hit_entropy,
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError('No rows produced. Check top motifs and family_level setting.')

    out = out.sort_values(['motif', 'bin']).reset_index(drop=True)

    motif_stem = Path(args.motif_path).stem
    top_tag = Path(args.top_motif_csv).stem if args.top_motif_csv else 'list'
    fam_tag = '.family' if args.family_level else ''
    unk_tag = '.noUnknown' if args.exclude_unknown else ''
    min_tag = f'.minActive{int(args.min_active_genes)}' if int(args.min_active_genes) > 0 else ''
    tag = f'{motif_stem}.{top_tag}.fold_{args.fold}.{args.split}{fam_tag}{unk_tag}{min_tag}'

    out_csv = os.path.join(args.output_dir, f'{tag}.topMotif_bin_distribution.csv')
    out.to_csv(out_csv, index=False)
    print('saved:', out_csv)

    summary = (
        out[['motif', 'motif_active_genes_any_bin', 'motif_top_bin_by_hits', 'motif_top_bin_hit_frac', 'motif_hit_entropy']]
        .drop_duplicates()
        .sort_values('motif_top_bin_hit_frac', ascending=False)
        .reset_index(drop=True)
    )
    summary_csv = os.path.join(args.output_dir, f'{tag}.topMotif_concentration_summary.csv')
    summary.to_csv(summary_csv, index=False)
    print('saved:', summary_csv)

    pivot_hits = out.pivot_table(index='motif', columns='bin', values='hit_frac_within_motif', fill_value=0.0)
    pivot_active = out.pivot_table(index='motif', columns='bin', values='active_gene_frac_bin_over_motif_active_genes', fill_value=0.0)

    heat1 = os.path.join(args.output_dir, f'{tag}.heatmap_hitFrac.png')
    heat2 = os.path.join(args.output_dir, f'{tag}.heatmap_activeFrac.png')
    maybe_plot_heatmap(pivot_hits, heat1, 'Motif Hit Distribution across Bins')
    maybe_plot_heatmap(pivot_active, heat2, 'Motif Hit Distribution across Bins')
    print('saved:', heat1)
    print('saved:', heat2)

    print('top concentration motifs:')
    print(summary.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
