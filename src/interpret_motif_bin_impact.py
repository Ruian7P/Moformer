import argparse
import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from EPInformer.models_abc import Moformer_P
from scripts.classification_utils import compute_binary_metrics


RNA_FEAT_COLS = [
    'UTR5LEN_log10zscore',
    'CDSLEN_log10zscore',
    'INTRONLEN_log10zscore',
    'UTR3LEN_log10zscore',
    'UTR5GC',
    'CDSGC',
    'UTR3GC',
    'ORFEXONDENSITY',
]


def parse_args():
    p = argparse.ArgumentParser(
        description='Bin-level motif masking impact for Moformer-P checkpoints.'
    )
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--motif_path', type=str, required=True)
    p.add_argument('--cell', type=str, default='K562')
    p.add_argument('--expr_table', type=str, default='./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv')
    p.add_argument('--split_table', type=str, default='./data/leave_chrom_out_crossvalidation_split_18377genes.csv')
    p.add_argument('--fold', type=str, default='enformer')
    p.add_argument('--split', type=str, default='test', choices=['all', 'train', 'valid', 'test'])
    p.add_argument('--output_dir', type=str, default='./results/motif_bin_impact')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--max_genes', type=int, default=0)

    p.add_argument('--mode', type=str, default='both', choices=['all_bins', 'top_motifs_by_bin', 'both'])
    p.add_argument('--mask_value', type=float, default=0.0)
    p.add_argument(
        '--active_only',
        action='store_true',
        help='Mask only active promoters (raw motif > 0 in masked columns), instead of masking all promoters.',
    )
    p.add_argument(
        '--eval_on_active_subset',
        action='store_true',
        help='Compute metrics only on active promoters (raw motif > 0 in masked columns) for each mask group.',
    )
    p.add_argument('--task', type=str, default='auto', choices=['auto', 'reg', 'cls'])
    p.add_argument('--expr_threshold', type=float, default=0.0)
    p.add_argument('--cls_prob_threshold', type=float, default=0.5)
    p.add_argument('--head', type=int, default=4)

    p.add_argument('--top_motif_csv', type=str, default='')
    p.add_argument('--top_motif_col', type=str, default='group')
    p.add_argument('--top_motif_list', type=str, default='')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument(
        '--top_mask_strategy',
        type=str,
        default='per_motif',
        choices=['per_motif', 'combined'],
        help='For mode=top_motifs_by_bin: per_motif masks each motif independently; combined masks all selected motifs in a bin together.',
    )
    p.add_argument(
        '--top_active_scope',
        type=str,
        default='motif_any_bin',
        choices=['masked_cols', 'motif_any_bin'],
        help='For top_motifs_by_bin + per_motif: active/eval subset comes from masked cols only, or all bins of the motif.',
    )
    p.add_argument('--family_level', action='store_true')
    p.add_argument('--exclude_unknown', action='store_true')

    p.add_argument('--motif_log1p', action='store_true')
    p.add_argument('--motif_zscore', action='store_true')
    p.add_argument('--motif_svd_dim', type=int, default=0)
    return p.parse_args()


def preprocess_motif_features(motif_df, train_ids, use_log1p=False, use_zscore=False, svd_dim=0, eps=1e-6):
    out = motif_df.copy()
    if use_log1p:
        out = np.log1p(np.maximum(out.values, 0.0))
        out = pd.DataFrame(out, index=motif_df.index, columns=motif_df.columns)
    if use_zscore:
        train_ids = [gid for gid in train_ids if gid in out.index]
        if len(train_ids) > 0:
            block = out.loc[train_ids]
            mu = block.mean(axis=0)
            sigma = block.std(axis=0).replace(0, np.nan)
            out = (out - mu) / (sigma + eps)
            out = out.fillna(0.0)
    if int(svd_dim) > 0 and out.shape[1] > int(svd_dim):
        train_ids = [gid for gid in train_ids if gid in out.index]
        if len(train_ids) > 1:
            from sklearn.decomposition import TruncatedSVD

            n_comp = min(int(svd_dim), max(1, len(train_ids) - 1), max(1, out.shape[1] - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd.fit(out.loc[train_ids].values)
            out = pd.DataFrame(
                svd.transform(out.values),
                index=out.index,
                columns=[f'motif_svd_{i}' for i in range(n_comp)],
            )
    return out


def build_motif_token_masks(motif_columns):
    bin_to_idx = {}
    for i, col in enumerate(motif_columns):
        if '__bin' not in str(col):
            continue
        try:
            b = int(str(col).rsplit('__bin', 1)[1])
        except Exception:
            continue
        bin_to_idx.setdefault(b, []).append(i)
    if len(bin_to_idx) == 0:
        return None
    token_indices = [bin_to_idx[b] for b in sorted(bin_to_idx.keys())]
    d = len(motif_columns)
    masks = np.zeros((len(token_indices), d), dtype=np.float32)
    for t, idxs in enumerate(token_indices):
        masks[t, idxs] = 1.0
    return masks


def infer_moformerp_config(state_dict):
    out_dim = int(state_dict['attn_encoder.0.norm1.weight'].shape[0])
    ptoexpr_in = int(state_dict['pToExpr.0.weight'].shape[1])
    use_rna_feats = ptoexpr_in > out_dim
    if 'motif_encoder.0.weight' in state_dict:
        motif_feat_dim = int(state_dict['motif_encoder.0.weight'].shape[1])
        motif_hidden_dim = int(state_dict['motif_encoder.0.weight'].shape[0])
    else:
        motif_feat_dim = int(state_dict['motif_token_encoders.0.0.weight'].shape[1])
        motif_hidden_dim = int(state_dict['motif_token_encoders.0.0.weight'].shape[0])
    n_encoder = len(
        {
            int(k.split('.')[1])
            for k in state_dict.keys()
            if k.startswith('attn_encoder.') and k.endswith('.norm1.weight')
        }
    )
    has_multitoken = 'motif_token_masks' in state_dict
    return {
        'out_dim': out_dim,
        'use_rna_feats': use_rna_feats,
        'motif_feat_dim': motif_feat_dim,
        'motif_hidden_dim': motif_hidden_dim,
        'n_encoder': n_encoder,
        'has_multitoken': has_multitoken,
    }


def motif_to_family(motif_name: str) -> str:
    return re.sub(r'\.\d+$', '', str(motif_name))


def pretty_motif_label(label: str) -> str:
    s = str(label)
    s = re.sub(r'^GM\.5\.0\.', '', s)
    s = re.sub(r'(?<=:)GM\.5\.0\.', '', s)
    return s


def metric_display_name(metric_col: str) -> str:
    return {
        'acc_drop': 'ACC Drop',
        'auroc_drop': 'AUROC Drop',
        'auprc_drop': 'AUPRC Drop',
        'mean_pred_drop': 'Mean Prediction Drop',
    }.get(metric_col, metric_col)


def parse_col_bin_and_base(col_name):
    col = str(col_name)
    m = re.search(r'__bin(\d+)$', col)
    if m is None:
        return None, re.sub(r'__global$', '', col)
    b = int(m.group(1))
    base = re.sub(r'__bin\d+$', '', col)
    return b, base


def get_bin_to_cols(motif_cols):
    out = OrderedDict()
    for i, c in enumerate(motif_cols):
        b, _ = parse_col_bin_and_base(c)
        if b is None:
            continue
        out.setdefault(b, []).append(i)
    return out


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
        key = m.lower()
        if args.exclude_unknown and 'unknown' in key:
            continue
        if key not in seen:
            seen.add(key)
            dedup.append(m)
    return dedup[: int(args.top_k)]


def build_bin_motif_map(motif_cols, family_level=False):
    out = OrderedDict()
    for i, c in enumerate(motif_cols):
        b, base = parse_col_bin_and_base(c)
        if b is None:
            continue
        if family_level:
            base = motif_to_family(base)
        out.setdefault(b, OrderedDict())
        out[b].setdefault(base, [])
        out[b][base].append(i)
    return out


def build_motif_to_all_cols(bin_motif_map):
    out = OrderedDict()
    for b in bin_motif_map:
        for motif, idxs in bin_motif_map[b].items():
            out.setdefault(motif, [])
            out[motif].extend(idxs)
    for motif in list(out.keys()):
        out[motif] = sorted(set(out[motif]))
    return out


def predict_batches(model, motif_values, rna_values, batch_size=256, device='cuda'):
    preds = []
    n = motif_values.shape[0]
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            motif_t = torch.from_numpy(motif_values[i:j]).float().to(device)
            rna_t = None if rna_values is None else torch.from_numpy(rna_values[i:j]).float().to(device)
            p, _ = model(rna_feats=rna_t, motif_feats=motif_t)
            preds.append(p.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_logits(y_true, logits, task='cls', cls_threshold=0.5):
    if task == 'cls':
        m = compute_binary_metrics(y_true, logits, threshold=cls_threshold)
        return {
            'acc': float(m['acc']),
            'auroc': float(m['auroc']),
            'auprc': float(m['auprc']),
        }
    return {
        'mean_pred': float(np.mean(logits)),
    }


def evaluate_pair(base_logits, masked_logits, y_true, task='cls', cls_threshold=0.5, eval_rows=None):
    if eval_rows is None:
        base_eval = evaluate_logits(y_true, base_logits, task=task, cls_threshold=cls_threshold)
        cur_eval = evaluate_logits(y_true, masked_logits, task=task, cls_threshold=cls_threshold)
        return base_eval, cur_eval
    if len(eval_rows) == 0:
        return None, None
    if task == 'cls':
        y_sub = y_true[eval_rows]
        base_eval = evaluate_logits(y_sub, base_logits[eval_rows], task=task, cls_threshold=cls_threshold)
        cur_eval = evaluate_logits(y_sub, masked_logits[eval_rows], task=task, cls_threshold=cls_threshold)
        return base_eval, cur_eval
    base_eval = evaluate_logits(None, base_logits[eval_rows], task=task, cls_threshold=cls_threshold)
    cur_eval = evaluate_logits(None, masked_logits[eval_rows], task=task, cls_threshold=cls_threshold)
    return base_eval, cur_eval


def score_row(base_metrics, cur_metrics, task='cls'):
    if task == 'cls':
        return {
            'acc_mask': cur_metrics['acc'],
            'auroc_mask': cur_metrics['auroc'],
            'auprc_mask': cur_metrics['auprc'],
            'acc_drop': float(base_metrics['acc'] - cur_metrics['acc']),
            'auroc_drop': float(base_metrics['auroc'] - cur_metrics['auroc']),
            'auprc_drop': float(base_metrics['auprc'] - cur_metrics['auprc']),
        }
    return {
        'mean_pred_mask': cur_metrics['mean_pred'],
        'mean_pred_drop': float(base_metrics['mean_pred'] - cur_metrics['mean_pred']),
    }


def get_active_rows(motif_raw_values, col_idxs):
    if len(col_idxs) == 0:
        return np.array([], dtype=np.int64)
    active_mask = (motif_raw_values[:, col_idxs] > 0).any(axis=1)
    return np.where(active_mask)[0].astype(np.int64)


def apply_mask_inplace(x, col_idxs, mask_value, row_idxs=None):
    if len(col_idxs) == 0:
        return None
    if row_idxs is None:
        row_idxs = np.arange(x.shape[0], dtype=np.int64)
    if len(row_idxs) == 0:
        return None
    backup = x[np.ix_(row_idxs, col_idxs)].copy()
    x[np.ix_(row_idxs, col_idxs)] = mask_value
    return (row_idxs, np.array(col_idxs, dtype=np.int64), backup)


def restore_inplace(x, backup):
    if backup is None:
        return
    row_idxs, col_idxs, block = backup
    x[np.ix_(row_idxs, col_idxs)] = block


def plot_bin_drops(df, out_png, metric_col):
    if len(df) == 0:
        return
    label_col = 'label'
    if metric_col not in df.columns:
        return
    plot_df = df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=True)
    if len(plot_df) == 0:
        return
    metric_name = metric_display_name(metric_col)
    plt.figure(figsize=(10, max(4, 0.5 * len(plot_df) + 1.2)))
    labels = [pretty_motif_label(x) for x in plot_df[label_col].astype(str).values]
    plt.barh(labels, plot_df[metric_col].values)
    plt.xlabel(metric_name)
    plt.title(f'Bin-level Impact by {metric_name}')
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def summarize_top_per_motif_bins(df, task, out_prefix, motif_order=None):
    if len(df) == 0:
        return
    if 'motif' not in df.columns:
        return
    metric_cols = ['mean_pred_drop'] if task != 'cls' else ['acc_drop', 'auroc_drop', 'auprc_drop']
    bins = sorted(df['bin'].dropna().astype(int).unique().tolist())

    for rank_col in metric_cols:
        if rank_col not in df.columns:
            continue
        work = df[['motif', 'bin', rank_col]].copy()
        work = work.dropna(subset=[rank_col])
        if len(work) == 0:
            continue

        # 1) Winner-bin ratio: for each motif, which bin causes the largest drop.
        winners = (
            work.sort_values(['motif', rank_col, 'bin'], ascending=[True, False, True])
            .groupby('motif', as_index=False)
            .first()
        )

        n_motifs = int(winners['motif'].nunique())
        winner_counts = winners.groupby('bin').size().reindex(bins, fill_value=0).astype(int)
        winner_props = winner_counts / max(1, n_motifs)

        # 2) Positive drop-mass ratio: how much total positive drop each bin contributes.
        work['pos_drop'] = work[rank_col].clip(lower=0.0)
        drop_mass = work.groupby('bin')['pos_drop'].sum().reindex(bins, fill_value=0.0)
        mass_total = float(drop_mass.sum())
        drop_mass_ratio = drop_mass / max(1e-12, mass_total)

        summary_df = pd.DataFrame(
            {
                'metric': rank_col,
                'bin': bins,
                'winner_count': winner_counts.values,
                'winner_ratio': winner_props.values,
                'positive_drop_mass': drop_mass.values,
                'positive_drop_mass_ratio': drop_mass_ratio.values,
                'n_motifs': n_motifs,
            }
        )

        summary_csv = f'{out_prefix}.{rank_col}.bin_influence_summary.csv'
        summary_df.to_csv(summary_csv, index=False)

        winners_csv = f'{out_prefix}.{rank_col}.winner_per_motif.csv'
        winners.rename(columns={rank_col: f'{rank_col}_winner'}).to_csv(winners_csv, index=False)

        plt.figure(figsize=(10, 4.5))
        x = np.arange(len(bins))
        width = 0.38
        plt.bar(x - width / 2, winner_props.values, width=width, label='winner ratio')
        plt.bar(x + width / 2, drop_mass_ratio.values, width=width, label='positive drop-mass ratio')
        plt.xticks(x, [f'bin{b}' for b in bins])
        plt.ylim(0, 1)
        plt.ylabel('Ratio')
        plt.title(f'Top Motif Bin Influence by {metric_display_name(rank_col)}')
        plt.legend()
        plt.tight_layout()
        summary_png = f'{out_prefix}.{rank_col}.bin_influence_summary.png'
        plt.savefig(summary_png, dpi=180)
        plt.close()

        print('saved:', summary_csv)
        print('saved:', winners_csv)
        print('saved:', summary_png)

        # 3) Motif x bin matrix and visualizations for easier per-motif inspection.
        pivot = work.pivot_table(index='motif', columns='bin', values=rank_col, aggfunc='first')
        if len(pivot) == 0:
            continue
        for b in bins:
            if b not in pivot.columns:
                pivot[b] = np.nan
        pivot = pivot[bins]

        if motif_order:
            rank_map = {m: i for i, m in enumerate(motif_order)}
            ordered_motifs = sorted(
                pivot.index.tolist(),
                key=lambda m: rank_map.get(m, len(rank_map) + pivot.index.tolist().index(m)),
            )
        else:
            winner_map = winners.set_index('motif')['bin']
            winner_drop = winners.set_index('motif')[rank_col]
            order_df = pd.DataFrame(
                {
                    'motif': pivot.index.tolist(),
                    'winner_bin': [int(winner_map.get(m, -1)) for m in pivot.index],
                    'winner_drop': [float(winner_drop.get(m, np.nan)) for m in pivot.index],
                }
            )
            order_df = order_df.sort_values(['winner_bin', 'winner_drop'], ascending=[True, False])
            ordered_motifs = order_df['motif'].tolist()
        pivot = pivot.loc[ordered_motifs]

        matrix_csv = f'{out_prefix}.{rank_col}.motif_bin_matrix.csv'
        pivot.to_csv(matrix_csv)
        print('saved:', matrix_csv)

        # Heatmap of drops.
        arr = pivot.values.astype(float)
        vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
        if vmax <= 0:
            vmax = 1.0
        fig_w = max(6, 0.9 * len(bins) + 2)
        fig_h = max(4, 0.45 * len(pivot.index) + 1.8)
        plt.figure(figsize=(fig_w, fig_h))
        im = plt.imshow(np.nan_to_num(arr, nan=0.0), aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, fraction=0.03, pad=0.02)
        plt.xticks(np.arange(len(bins)), [f'bin{b}' for b in bins])
        plt.yticks(np.arange(len(pivot.index)), [pretty_motif_label(x) for x in pivot.index.tolist()])
        plt.xlabel('Bin')
        plt.ylabel('Motif')
        plt.title(f'Motif x Bin by {metric_display_name(rank_col)}')
        plt.tight_layout()
        heatmap_png = f'{out_prefix}.{rank_col}.motif_bin_heatmap.png'
        plt.savefig(heatmap_png, dpi=180)
        plt.close()
        print('saved:', heatmap_png)

        # Heatmap of per-motif bin ranking (1 = largest drop).
        rank_mat = pivot.rank(axis=1, ascending=False, method='min')
        rank_csv = f'{out_prefix}.{rank_col}.motif_bin_rank.csv'
        rank_mat.to_csv(rank_csv)
        print('saved:', rank_csv)

        plt.figure(figsize=(fig_w, fig_h))
        im = plt.imshow(rank_mat.values.astype(float), aspect='auto', cmap='viridis_r', vmin=1, vmax=len(bins))
        plt.colorbar(im, fraction=0.03, pad=0.02)
        plt.xticks(np.arange(len(bins)), [f'bin{b}' for b in bins])
        plt.yticks(np.arange(len(rank_mat.index)), [pretty_motif_label(x) for x in rank_mat.index.tolist()])
        plt.xlabel('Bin')
        plt.ylabel('Motif')
        plt.title(f'Motif x Bin Rank by {metric_display_name(rank_col)}')
        plt.tight_layout()
        rank_png = f'{out_prefix}.{rank_col}.motif_bin_rank_heatmap.png'
        plt.savefig(rank_png, dpi=180)
        plt.close()
        print('saved:', rank_png)

        # Small-multiples bar chart, one panel per motif.
        n_motif = len(pivot.index)
        ncol = 3
        nrow = int(np.ceil(n_motif / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, max(2.8, 2.3 * nrow)))
        axes = np.array(axes).reshape(-1)
        x = np.arange(len(bins))
        for i, motif in enumerate(pivot.index.tolist()):
            ax = axes[i]
            vals = pivot.loc[motif].values.astype(float)
            bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), color='#4C78A8')
            if np.isfinite(vals).any():
                win_i = int(np.nanargmax(vals))
                bars[win_i].set_color('#E45756')
            ax.set_xticks(x, [f'b{b}' for b in bins], fontsize=8)
            ax.set_title(pretty_motif_label(motif), fontsize=9)
            ax.axhline(0, color='black', linewidth=0.6)
        for j in range(n_motif, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f'Per-motif Bin Impact by {metric_display_name(rank_col)}', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        panel_png = f'{out_prefix}.{rank_col}.motif_bin_panels.png'
        fig.savefig(panel_png, dpi=180)
        plt.close(fig)
        print('saved:', panel_png)


def shorten_component(name, max_len=140):
    s = str(name)
    if len(s) <= max_len:
        return s
    digest = hashlib.sha1(s.encode('utf-8')).hexdigest()[:10]
    keep = max(1, max_len - 11)
    return f'{s[:keep]}.{digest}'


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print('device:', device)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    cfg = infer_moformerp_config(state_dict)
    print('inferred config:', cfg)

    task = 'cls' if (args.task == 'auto' and '.cls.' in Path(args.checkpoint).name) else args.task
    if task == 'auto':
        task = 'reg'
    print('task:', task)

    split_df = pd.read_csv(args.split_table, index_col=0)
    fold_col = f'fold_{args.fold}'
    if fold_col not in split_df.columns:
        raise KeyError(f'{fold_col} not found in split table')
    train_ids = split_df[split_df[fold_col] == 'train'].index.tolist()
    split_ids = split_df.index.tolist() if args.split == 'all' else split_df[split_df[fold_col] == args.split].index.tolist()

    motif_df_raw = pd.read_csv(args.motif_path, sep='\t', comment='#', index_col=0, engine='python')
    motif_df_raw = motif_df_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    motif_df = preprocess_motif_features(
        motif_df_raw,
        train_ids=train_ids,
        use_log1p=args.motif_log1p,
        use_zscore=args.motif_zscore,
        svd_dim=args.motif_svd_dim,
    )

    expr_df = pd.read_csv(args.expr_table, index_col='gene_id')
    if cfg['use_rna_feats']:
        gene_ids = [gid for gid in split_ids if gid in motif_df.index and gid in expr_df.index]
    else:
        gene_ids = [gid for gid in split_ids if gid in motif_df.index]
    if args.max_genes > 0:
        gene_ids = gene_ids[: args.max_genes]
    if len(gene_ids) == 0:
        raise ValueError('No genes left after filtering.')
    print('n_genes:', len(gene_ids))

    motif_df_raw = motif_df_raw.loc[gene_ids]
    motif_df = motif_df.loc[gene_ids]
    motif_cols = list(motif_df.columns)
    motif_values = motif_df.values.astype(np.float32)
    motif_raw_values = motif_df_raw.values.astype(np.float32)
    if motif_values.shape[1] != cfg['motif_feat_dim']:
        raise ValueError(f'motif dim mismatch: data={motif_values.shape[1]} ckpt={cfg["motif_feat_dim"]}')

    rna_values = None
    if cfg['use_rna_feats']:
        rna_values = expr_df.loc[gene_ids, RNA_FEAT_COLS].astype(float).values.astype(np.float32)

    y_true = None
    if task == 'cls':
        y_true = (expr_df.loc[gene_ids, f'Actual_{args.cell}'].astype(float).values > float(args.expr_threshold)).astype(int)

    motif_token_masks = None
    if cfg['has_multitoken']:
        motif_token_masks = state_dict['motif_token_masks'].detach().cpu().numpy().astype(np.float32)
        if motif_token_masks.shape[1] != motif_values.shape[1]:
            motif_token_masks = build_motif_token_masks(motif_cols)
            if motif_token_masks is None:
                raise ValueError('Multitoken checkpoint but cannot rebuild motif_token_masks from columns.')

    model = Moformer_P(
        out_dim=cfg['out_dim'],
        n_encoder=cfg['n_encoder'],
        head=args.head,
        n_enhancer=60,
        useBN=False,
        usePromoterSignal=False,
        useFeat=cfg['use_rna_feats'],
        motif_feat_dim=cfg['motif_feat_dim'],
        motif_hidden_dim=cfg['motif_hidden_dim'],
        motif_token_masks=motif_token_masks,
        device=device,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    base_logits = predict_batches(
        model=model,
        motif_values=motif_values,
        rna_values=rna_values,
        batch_size=args.batch_size,
        device=device,
    )
    base_metrics = evaluate_logits(
        y_true=y_true,
        logits=base_logits,
        task=task,
        cls_threshold=float(args.cls_prob_threshold),
    )
    print('base_metrics:', base_metrics)

    bin_to_cols = get_bin_to_cols(motif_cols)
    if len(bin_to_cols) == 0:
        raise ValueError('No __bin* columns found in motif table.')
    print('n_bins:', len(bin_to_cols), 'bins:', list(bin_to_cols.keys()))
    print('active_only:', bool(args.active_only))
    print('eval_on_active_subset:', bool(args.eval_on_active_subset))

    rows = []
    x = motif_values.copy()

    if args.mode in ('all_bins', 'both'):
        for b in sorted(bin_to_cols.keys()):
            col_set = sorted(set(bin_to_cols[b]))
            row_idxs = get_active_rows(motif_raw_values, col_set) if args.active_only else None
            backup = apply_mask_inplace(x, col_set, args.mask_value, row_idxs=row_idxs)
            logits = predict_batches(model, x, rna_values, args.batch_size, device)
            restore_inplace(x, backup)

            eval_rows = get_active_rows(motif_raw_values, col_set) if args.eval_on_active_subset else None
            base_eval, cur_eval = evaluate_pair(
                base_logits=base_logits,
                masked_logits=logits,
                y_true=y_true,
                task=task,
                cls_threshold=float(args.cls_prob_threshold),
                eval_rows=eval_rows,
            )
            if base_eval is None:
                print(f'all_bins skip: bin{b} has 0 eval genes under eval_on_active_subset')
                continue
            sc = score_row(base_eval, cur_eval, task=task)
            n_active = int(len(row_idxs)) if row_idxs is not None else int(len(gene_ids))
            n_eval = int(len(eval_rows)) if eval_rows is not None else int(len(gene_ids))
            rows.append(
                {
                    'analysis': 'all_motifs_in_bin',
                    'bin': b,
                    'label': f'bin{b}',
                    'n_cols_masked': len(col_set),
                    'active_genes_masked': n_active,
                    'active_frac_masked': float(n_active / max(1, len(gene_ids))),
                    'eval_genes': n_eval,
                    'eval_frac': float(n_eval / max(1, len(gene_ids))),
                    **sc,
                }
            )
            print(
                f'all_bins done: bin{b}, n_cols={len(col_set)}, active_genes_masked={n_active}, eval_genes={n_eval}'
            )

    if args.mode in ('top_motifs_by_bin', 'both'):
        top_motifs = get_top_motifs(args)
        if len(top_motifs) == 0:
            raise ValueError('No top motifs found. Please provide --top_motif_csv or --top_motif_list.')
        print(
            f'n_top_motifs: {len(top_motifs)}, top_mask_strategy: {args.top_mask_strategy}, '
            f'top_active_scope: {args.top_active_scope}'
        )

        bin_motif_map = build_bin_motif_map(motif_cols, family_level=args.family_level)
        motif_to_all_cols = build_motif_to_all_cols(bin_motif_map)
        for b in sorted(bin_to_cols.keys()):
            if args.top_mask_strategy == 'combined':
                selected_cols = []
                matched_motifs = []
                for m in top_motifs:
                    key = motif_to_family(m) if args.family_level else m
                    if args.exclude_unknown and 'unknown' in key.lower():
                        continue
                    idxs = bin_motif_map.get(b, {}).get(key, [])
                    if len(idxs) > 0:
                        selected_cols.extend(idxs)
                        matched_motifs.append(m)

                col_set = sorted(set(selected_cols))
                if len(col_set) == 0:
                    print(f'top_motifs_by_bin skip: bin{b} no matched motif columns')
                    continue

                row_idxs = get_active_rows(motif_raw_values, col_set) if args.active_only else None
                backup = apply_mask_inplace(x, col_set, args.mask_value, row_idxs=row_idxs)
                logits = predict_batches(model, x, rna_values, args.batch_size, device)
                restore_inplace(x, backup)

                eval_rows = get_active_rows(motif_raw_values, col_set) if args.eval_on_active_subset else None
                base_eval, cur_eval = evaluate_pair(
                    base_logits=base_logits,
                    masked_logits=logits,
                    y_true=y_true,
                    task=task,
                    cls_threshold=float(args.cls_prob_threshold),
                    eval_rows=eval_rows,
                )
                if base_eval is None:
                    print(f'top_motifs_by_bin combined skip: bin{b} has 0 eval genes')
                    continue
                sc = score_row(base_eval, cur_eval, task=task)
                n_active = int(len(row_idxs)) if row_idxs is not None else int(len(gene_ids))
                n_eval = int(len(eval_rows)) if eval_rows is not None else int(len(gene_ids))
                rows.append(
                    {
                        'analysis': 'top_motifs_in_bin_combined',
                        'bin': b,
                        'label': f'bin{b}',
                        'motif': 'ALL_TOP_MOTIFS',
                        'n_top_motifs_input': len(top_motifs),
                        'n_top_motifs_matched': len(set(matched_motifs)),
                        'matched_motifs': '|'.join(sorted(set(matched_motifs))),
                        'n_cols_masked': len(col_set),
                        'active_genes_masked': n_active,
                        'active_frac_masked': float(n_active / max(1, len(gene_ids))),
                        'eval_genes': n_eval,
                        'eval_frac': float(n_eval / max(1, len(gene_ids))),
                        **sc,
                    }
                )
                print(
                    f'top_motifs_by_bin combined done: bin{b}, matched_motifs={len(set(matched_motifs))}, '
                    f'n_cols={len(col_set)}, active_genes_masked={n_active}, eval_genes={n_eval}'
                )
            else:
                n_matched = 0
                for m in top_motifs:
                    key = motif_to_family(m) if args.family_level else m
                    if args.exclude_unknown and 'unknown' in key.lower():
                        continue
                    col_set = sorted(set(bin_motif_map.get(b, {}).get(key, [])))
                    if len(col_set) == 0:
                        continue
                    n_matched += 1

                    if args.top_active_scope == 'motif_any_bin':
                        ref_cols = motif_to_all_cols.get(key, [])
                    else:
                        ref_cols = col_set
                    row_idxs = get_active_rows(motif_raw_values, ref_cols) if args.active_only else None
                    backup = apply_mask_inplace(x, col_set, args.mask_value, row_idxs=row_idxs)
                    logits = predict_batches(model, x, rna_values, args.batch_size, device)
                    restore_inplace(x, backup)

                    eval_rows = get_active_rows(motif_raw_values, ref_cols) if args.eval_on_active_subset else None
                    base_eval, cur_eval = evaluate_pair(
                        base_logits=base_logits,
                        masked_logits=logits,
                        y_true=y_true,
                        task=task,
                        cls_threshold=float(args.cls_prob_threshold),
                        eval_rows=eval_rows,
                    )
                    if base_eval is None:
                        continue
                    sc = score_row(base_eval, cur_eval, task=task)
                    n_active = int(len(row_idxs)) if row_idxs is not None else int(len(gene_ids))
                    n_eval = int(len(eval_rows)) if eval_rows is not None else int(len(gene_ids))
                    rows.append(
                        {
                            'analysis': 'top_motifs_in_bin_per_motif',
                            'bin': b,
                            'label': f'bin{b}:{pretty_motif_label(m)}',
                            'motif': m,
                            'n_top_motifs_input': len(top_motifs),
                            'n_top_motifs_matched': 1,
                            'matched_motifs': m,
                            'n_cols_masked': len(col_set),
                            'active_scope': args.top_active_scope,
                            'active_genes_masked': n_active,
                            'active_frac_masked': float(n_active / max(1, len(gene_ids))),
                            'eval_genes': n_eval,
                            'eval_frac': float(n_eval / max(1, len(gene_ids))),
                            **sc,
                        }
                    )
                print(
                    f'top_motifs_by_bin per_motif done: bin{b}, matched_motifs={n_matched}'
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError('No analysis rows produced.')

    rank_col = 'acc_drop' if task == 'cls' else 'mean_pred_drop'
    out = out.sort_values(['analysis', rank_col], ascending=[True, False]).reset_index(drop=True)

    stem = Path(args.checkpoint).stem
    motif_stem = Path(args.motif_path).stem
    active_tag = '.activeOnly' if args.active_only else ''
    eval_tag = '.evalActive' if args.eval_on_active_subset else ''
    tag = f'{stem}.{motif_stem}.fold_{args.fold}.{args.split}.{args.mode}{active_tag}{eval_tag}'
    safe_tag = shorten_component(tag, max_len=120)
    if safe_tag != tag:
        print('note: shortened output tag to avoid path-length issues:', safe_tag)

    out_csv = os.path.join(args.output_dir, f'{safe_tag}.summary.csv')
    out.to_csv(out_csv, index=False)
    print('saved:', out_csv)

    for ana in out['analysis'].unique().tolist():
        sub = out[out['analysis'] == ana].copy()
        if task == 'cls':
            metric_cols = ['acc_drop', 'auroc_drop', 'auprc_drop']
            for metric_col in metric_cols:
                metric_png = os.path.join(args.output_dir, f'{safe_tag}.{ana}.{metric_col}.png')
                plot_bin_drops(sub, metric_png, metric_col=metric_col)
                if os.path.exists(metric_png):
                    print('saved:', metric_png)
            # Backward-compatible alias: keep old filename for acc_drop
            legacy_png = os.path.join(args.output_dir, f'{safe_tag}.{ana}.png')
            plot_bin_drops(sub, legacy_png, metric_col='acc_drop')
            if os.path.exists(legacy_png):
                print('saved:', legacy_png)
        else:
            out_png = os.path.join(args.output_dir, f'{safe_tag}.{ana}.png')
            plot_bin_drops(sub, out_png, metric_col='mean_pred_drop')
            if os.path.exists(out_png):
                print('saved:', out_png)

    if 'top_motifs_in_bin_per_motif' in out['analysis'].unique().tolist():
        sub = out[out['analysis'] == 'top_motifs_in_bin_per_motif'].copy()
        top_motif_order = [motif_to_family(m) if args.family_level else m for m in get_top_motifs(args)]
        long_prefix = f'{safe_tag}.top_motifs_in_bin_per_motif'
        short_prefix = shorten_component(long_prefix, max_len=120)
        out_prefix = os.path.join(args.output_dir, short_prefix)
        if short_prefix != long_prefix:
            print('note: shortened bin summary prefix to avoid path-length issues:', short_prefix)
        summarize_top_per_motif_bins(sub, task=task, out_prefix=out_prefix, motif_order=top_motif_order)

    print('done. top rows:')
    print(out.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
