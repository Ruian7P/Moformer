import argparse
import itertools
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
        description='Motif combination occlusion for Moformer-P checkpoints.'
    )
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--motif_path', type=str, required=True)
    p.add_argument('--cell', type=str, default='K562')
    p.add_argument('--expr_table', type=str, default='./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv')
    p.add_argument('--split_table', type=str, default='./data/leave_chrom_out_crossvalidation_split_18377genes.csv')
    p.add_argument('--fold', type=str, default='enformer')
    p.add_argument('--split', type=str, default='test', choices=['all', 'train', 'valid', 'test'])
    p.add_argument('--output_dir', type=str, default='./results/motif_combo_occlusion')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--max_genes', type=int, default=0)

    p.add_argument('--group_mode', type=str, default='motif', choices=['motif', 'column'])
    p.add_argument(
        '--family_level',
        action='store_true',
        help='When group_mode=motif, collapse motifs into families (e.g., GM.5.0.Ets.0013 -> GM.5.0.Ets).',
    )
    p.add_argument('--mask_value', type=float, default=0.0)
    p.add_argument(
        '--exclude_unknown',
        action='store_true',
        help='Exclude motif groups whose name contains "Unknown" (case-insensitive).',
    )
    p.add_argument(
        '--exclude_mixed',
        action='store_true',
        help='Exclude motif groups whose name contains "Mixed" (case-insensitive).',
    )
    p.add_argument('--task', type=str, default='auto', choices=['auto', 'reg', 'cls'])
    p.add_argument('--expr_threshold', type=float, default=0.0)
    p.add_argument(
        '--expressed_only',
        action='store_true',
        help='Only evaluate genes with Actual_<cell> > expressed_threshold.',
    )
    p.add_argument(
        '--expressed_threshold',
        type=float,
        default=0.0,
        help='Threshold used with --expressed_only on Actual_<cell>.',
    )
    p.add_argument('--cls_prob_threshold', type=float, default=0.5)
    p.add_argument('--head', type=int, default=4)
    p.add_argument(
        '--active_sample_n',
        type=int,
        default=0,
        help='If >0 (motif_count=1), sample this many promoters from each motif active set and compute sampled drop.',
    )
    p.add_argument(
        '--active_sample_trials',
        type=int,
        default=100,
        help='Number of random samplings per motif when --active_sample_n > 0.',
    )
    p.add_argument(
        '--active_sample_seed',
        type=int,
        default=42,
        help='Random seed for --active_sample_* sampling.',
    )
    p.add_argument(
        '--min_active_genes',
        type=int,
        default=0,
        help='Drop motif groups with active_genes < this threshold in the selected split.',
    )

    p.add_argument('--motif_count', type=int, default=1, help='Number of motif groups masked together.')
    p.add_argument('--candidate_top_n', type=int, default=30, help='For motif_count>=2, choose combos from top-N single motifs.')
    p.add_argument('--single_summary', type=str, default=None, help='Optional precomputed single-mask summary csv.')
    p.add_argument('--max_combos', type=int, default=200000)
    p.add_argument('--topk', type=int, default=10)

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
        if '__bin' in str(col):
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
    # Example: GM.5.0.Ets.0013 -> GM.5.0.Ets
    # Keep untouched if no numeric suffix.
    return re.sub(r'\.\d+$', '', str(motif_name))


def pretty_motif_label(label: str) -> str:
    parts = [re.sub(r'^GM\.5\.0\.', '', p.strip()) for p in str(label).split('+')]
    return ' + '.join(parts)


def metric_display_name(metric_col: str) -> str:
    return {
        'acc_drop': 'ACC Drop',
        'sampled_acc_drop_mean': 'ACC Drop',
        'active_acc_drop': 'ACC Drop',
        'auroc_drop': 'AUROC Drop',
        'sampled_auroc_drop_mean': 'AUROC Drop',
        'active_auroc_drop': 'AUROC Drop',
        'auprc_drop': 'AUPRC Drop',
        'sampled_auprc_drop_mean': 'AUPRC Drop',
        'active_auprc_drop': 'AUPRC Drop',
        'mean_pred_drop': 'Mean Prediction Drop',
    }.get(metric_col, metric_col)


def build_groups(columns, mode='motif', family_level=False):
    groups = OrderedDict()
    if mode == 'column':
        for i, c in enumerate(columns):
            groups[str(c)] = [i]
        return groups
    for i, c in enumerate(columns):
        col = str(c)
        base = re.sub(r'__bin\d+$', '', col)
        base = re.sub(r'__global$', '', base)
        if family_level:
            base = motif_to_family(base)
        if base not in groups:
            groups[base] = []
        groups[base].append(i)
    return groups


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


def score_row(base_metrics, cur_metrics, task='cls'):
    if task == 'cls':
        acc_drop = float(base_metrics['acc'] - cur_metrics['acc'])
        auroc_drop = float(base_metrics['auroc'] - cur_metrics['auroc']) if not np.isnan(base_metrics['auroc']) else float('nan')
        auprc_drop = float(base_metrics['auprc'] - cur_metrics['auprc']) if not np.isnan(base_metrics['auprc']) else float('nan')
        return {
            'acc_mask': cur_metrics['acc'],
            'auroc_mask': cur_metrics['auroc'],
            'auprc_mask': cur_metrics['auprc'],
            'acc_drop': acc_drop,
            'auroc_drop': auroc_drop,
            'auprc_drop': auprc_drop,
        }
    return {
        'mean_pred_mask': cur_metrics['mean_pred'],
        'mean_pred_drop': float(base_metrics['mean_pred'] - cur_metrics['mean_pred']),
    }


def apply_mask_inplace(x, col_idxs, mask_value):
    backup = {}
    for c in col_idxs:
        backup[c] = x[:, c].copy()
        x[:, c] = mask_value
    return backup


def restore_inplace(x, backup):
    for c, v in backup.items():
        x[:, c] = v


def _safe_div(num, den):
    if den == 0 or np.isnan(den):
        return float('nan')
    return float(num / den)


def plot_topk_metric(df, combo_col, metric_col, topk, k, out_png):
    if metric_col not in df.columns:
        return False
    sub = df.dropna(subset=[metric_col]).copy()
    if len(sub) == 0:
        return False
    topk = min(int(topk), len(sub))
    top = sub.sort_values(metric_col, ascending=False).head(topk).iloc[::-1]
    plt.figure(figsize=(12, max(5, 0.5 * topk + 1)))
    labels = [pretty_motif_label(x) for x in top[combo_col].astype(str).values]
    metric_name = metric_display_name(metric_col)
    item_name = 'Motif' if int(k) == 1 else 'Motif Combos'
    plt.barh(labels, top[metric_col].values)
    plt.xlabel(metric_name)
    plt.title(f'Top {topk} {item_name} by {metric_name}')
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.motif_count < 1:
        raise ValueError('--motif_count must be >= 1')

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print('device:', device)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    cfg = infer_moformerp_config(state_dict)
    print('inferred config:', cfg)

    if args.task == 'auto':
        task = 'cls' if '.cls.' in Path(args.checkpoint).name else 'reg'
    else:
        task = args.task
    print('task:', task)

    split_df = pd.read_csv(args.split_table, index_col=0)
    fold_col = f'fold_{args.fold}'
    if fold_col not in split_df.columns:
        raise KeyError(f'{fold_col} not found in split table')
    train_ids = split_df[split_df[fold_col] == 'train'].index.tolist()
    split_ids = split_df.index.tolist() if args.split == 'all' else split_df[split_df[fold_col] == args.split].index.tolist()

    motif_df = pd.read_csv(args.motif_path, sep='\t', comment='#', index_col=0, engine='python')
    motif_df = motif_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    motif_df_raw = motif_df.copy()
    motif_df = preprocess_motif_features(
        motif_df,
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
    if args.expressed_only:
        expr_col = f'Actual_{args.cell}'
        if expr_col not in expr_df.columns:
            raise KeyError(f'{expr_col} not found in expression table')
        thr = float(args.expressed_threshold)
        gene_ids = [gid for gid in gene_ids if gid in expr_df.index and float(expr_df.loc[gid, expr_col]) > thr]
        print(f'expressed_only enabled: Actual_{args.cell} > {thr}')
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
        raise ValueError(
            f'motif dim mismatch: data={motif_values.shape[1]} ckpt={cfg["motif_feat_dim"]}'
        )

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
            # fallback: rebuild from columns if saved mask dim mismatches
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

    groups = build_groups(motif_cols, mode=args.group_mode, family_level=args.family_level)
    if args.exclude_unknown:
        before_n = len(groups)
        groups = OrderedDict(
            (g, idxs) for g, idxs in groups.items() if 'unknown' not in str(g).lower()
        )
        print(f'exclude_unknown enabled: {before_n} -> {len(groups)} groups')
    if args.exclude_mixed:
        before_n = len(groups)
        groups = OrderedDict(
            (g, idxs) for g, idxs in groups.items() if 'mixed' not in str(g).lower()
        )
        print(f'exclude_mixed enabled: {before_n} -> {len(groups)} groups')
    if int(args.min_active_genes) > 0:
        min_n = int(args.min_active_genes)
        before_n = len(groups)
        groups = OrderedDict(
            (g, idxs)
            for g, idxs in groups.items()
            if int((motif_raw_values[:, idxs] > 0).any(axis=1).sum()) >= min_n
        )
        print(f'min_active_genes enabled ({min_n}): {before_n} -> {len(groups)} groups')
    group_names = list(groups.keys())
    print('n_groups:', len(group_names), 'group_mode:', args.group_mode, 'family_level:', args.family_level)
    if int(args.active_sample_n) > 0:
        if task != 'cls':
            raise ValueError('--active_sample_n currently supports only --task cls')
        if int(args.motif_count) != 1:
            raise ValueError('--active_sample_n currently supports only --motif_count 1')
        print(
            'active_sample enabled:',
            'n=', int(args.active_sample_n),
            'trials=', int(args.active_sample_trials),
            'seed=', int(args.active_sample_seed),
        )

    # candidate group selection
    if args.motif_count == 1:
        candidates = group_names
    else:
        if args.single_summary is not None and os.path.exists(args.single_summary):
            s = pd.read_csv(args.single_summary)
            rank_col = 'acc_drop' if task == 'cls' and 'acc_drop' in s.columns else (
                'mean_abs_delta' if 'mean_abs_delta' in s.columns else s.columns[-1]
            )
            candidates = [g for g in s.sort_values(rank_col, ascending=False)['group'].tolist() if g in groups]
        else:
            print('single_summary not provided; computing single-mask ranking for candidate selection...')
            rows = []
            x = motif_values.copy()
            for i, g in enumerate(group_names):
                backup = apply_mask_inplace(x, groups[g], args.mask_value)
                logits = predict_batches(model, x, rna_values, args.batch_size, device)
                restore_inplace(x, backup)
                cur = evaluate_logits(y_true, logits, task, float(args.cls_prob_threshold))
                sc = score_row(base_metrics, cur, task=task)
                rows.append({'group': g, **sc})
                if (i + 1) % 100 == 0 or (i + 1) == len(group_names):
                    print(f'single ranking {i + 1}/{len(group_names)}')
            s = pd.DataFrame(rows)
            rank_col = 'acc_drop' if task == 'cls' else 'mean_pred_drop'
            s = s.sort_values(rank_col, ascending=False).reset_index(drop=True)
            single_path = os.path.join(args.output_dir, f'{Path(args.checkpoint).stem}.single_for_combo.csv')
            s.to_csv(single_path, index=False)
            print('saved single ranking:', single_path)
            candidates = s['group'].tolist()

        candidates = candidates[: int(args.candidate_top_n)]
        print(f'candidate groups for combos: {len(candidates)} (top-{args.candidate_top_n})')

    # build combos
    k = int(args.motif_count)
    if len(candidates) < k:
        raise ValueError(f'Not enough candidates ({len(candidates)}) for motif_count={k}')
    combos = list(itertools.combinations(candidates, k))
    if len(combos) > int(args.max_combos):
        raise ValueError(
            f'Number of combos={len(combos)} exceeds --max_combos={args.max_combos}. '
            f'Reduce candidate_top_n or motif_count.'
        )
    print('n_combos:', len(combos), 'motif_count:', k)

    x = motif_values.copy()
    rows = []
    rng = np.random.default_rng(int(args.active_sample_seed))
    n_total = len(gene_ids)
    for i, combo in enumerate(combos):
        col_set = sorted({c for g in combo for c in groups[g]})
        backup = apply_mask_inplace(x, col_set, args.mask_value)
        logits = predict_batches(model, x, rna_values, args.batch_size, device)
        restore_inplace(x, backup)

        cur = evaluate_logits(y_true, logits, task, float(args.cls_prob_threshold))
        sc = score_row(base_metrics, cur, task=task)
        extra = {}
        if int(args.active_sample_n) > 0:
            active_mask = (motif_raw_values[:, col_set] > 0).any(axis=1)
            active_idx = np.where(active_mask)[0]
            active_n = int(active_idx.size)
            sample_n = int(args.active_sample_n)
            extra['active_genes'] = active_n
            extra['active_frac'] = float(active_n / max(1, n_total))
            extra['active_sample_n'] = sample_n
            if active_n >= sample_n and sample_n > 0:
                base_active = evaluate_logits(
                    y_true[active_idx],
                    base_logits[active_idx],
                    task='cls',
                    cls_threshold=float(args.cls_prob_threshold),
                )
                cur_active = evaluate_logits(
                    y_true[active_idx],
                    logits[active_idx],
                    task='cls',
                    cls_threshold=float(args.cls_prob_threshold),
                )
                active_drop = float(base_active['acc'] - cur_active['acc'])
                active_auroc_drop = float(base_active['auroc'] - cur_active['auroc'])
                active_auprc_drop = float(base_active['auprc'] - cur_active['auprc'])
                extra['base_acc_active'] = float(base_active['acc'])
                extra['base_auroc_active'] = float(base_active['auroc'])
                extra['base_auprc_active'] = float(base_active['auprc'])
                extra['acc_mask_active'] = float(cur_active['acc'])
                extra['auroc_mask_active'] = float(cur_active['auroc'])
                extra['auprc_mask_active'] = float(cur_active['auprc'])
                extra['active_acc_drop'] = active_drop
                extra['active_auroc_drop'] = active_auroc_drop
                extra['active_auprc_drop'] = active_auprc_drop

                sampled_acc_drops = []
                sampled_auroc_drops = []
                sampled_auprc_drops = []
                for _ in range(int(args.active_sample_trials)):
                    sampled_idx = rng.choice(active_idx, size=sample_n, replace=False)
                    base_s = evaluate_logits(
                        y_true[sampled_idx],
                        base_logits[sampled_idx],
                        task='cls',
                        cls_threshold=float(args.cls_prob_threshold),
                    )
                    cur_s = evaluate_logits(
                        y_true[sampled_idx],
                        logits[sampled_idx],
                        task='cls',
                        cls_threshold=float(args.cls_prob_threshold),
                    )
                    sampled_acc_drops.append(float(base_s['acc'] - cur_s['acc']))
                    sampled_auroc_drops.append(float(base_s['auroc'] - cur_s['auroc']))
                    sampled_auprc_drops.append(float(base_s['auprc'] - cur_s['auprc']))

                for metric_name, drops in [
                    ('acc', sampled_acc_drops),
                    ('auroc', sampled_auroc_drops),
                    ('auprc', sampled_auprc_drops),
                ]:
                    drops = np.asarray(drops, dtype=float)
                    mean = float(np.nanmean(drops))
                    std = float(np.nanstd(drops, ddof=1)) if drops.size > 1 else float('nan')
                    extra[f'sampled_{metric_name}_drop_mean'] = mean
                    extra[f'sampled_{metric_name}_drop_std'] = std
                    extra[f'sampled_{metric_name}_drop_se'] = _safe_div(std, np.sqrt(max(1, drops.size)))
            else:
                extra['base_acc_active'] = float('nan')
                extra['base_auroc_active'] = float('nan')
                extra['base_auprc_active'] = float('nan')
                extra['acc_mask_active'] = float('nan')
                extra['auroc_mask_active'] = float('nan')
                extra['auprc_mask_active'] = float('nan')
                extra['active_acc_drop'] = float('nan')
                extra['active_auroc_drop'] = float('nan')
                extra['active_auprc_drop'] = float('nan')
                for metric_name in ['acc', 'auroc', 'auprc']:
                    extra[f'sampled_{metric_name}_drop_mean'] = float('nan')
                    extra[f'sampled_{metric_name}_drop_std'] = float('nan')
                    extra[f'sampled_{metric_name}_drop_se'] = float('nan')
        rows.append(
            {
                'combo': ' + '.join(combo),
                'motif_count': k,
                'n_groups_masked': len(combo),
                'n_cols_masked': len(col_set),
                **sc,
                **extra,
            }
        )
        if (i + 1) % 100 == 0 or (i + 1) == len(combos):
            print(f'combo {i + 1}/{len(combos)}')

    out = pd.DataFrame(rows)
    if int(args.active_sample_n) > 0:
        rank_col = 'sampled_acc_drop_mean'
    else:
        rank_col = 'acc_drop' if task == 'cls' else 'mean_pred_drop'
    out = out.sort_values(rank_col, ascending=False).reset_index(drop=True)

    stem = Path(args.checkpoint).stem
    expr_tag = f'.expr_gt{args.expressed_threshold:g}' if args.expressed_only else ''
    family_tag = '.family' if args.family_level else ''
    tag = f'{stem}.fold_{args.fold}.{args.split}{expr_tag}{family_tag}.k{k}.combo_{args.group_mode}'
    out_csv = os.path.join(args.output_dir, f'{tag}.summary.csv')
    out.to_csv(out_csv, index=False)
    print('saved:', out_csv)

    topk = min(int(args.topk), len(out))
    out_png = os.path.join(args.output_dir, f'{tag}.top{topk}.png')
    ok = plot_topk_metric(out, 'combo', rank_col, topk, k, out_png)
    if ok:
        print('saved:', out_png)

    if task == 'cls':
        metric_cols = ['acc_drop', 'auroc_drop', 'auprc_drop']
        if int(args.active_sample_n) > 0:
            metric_cols = [
                'sampled_acc_drop_mean',
                'sampled_auroc_drop_mean',
                'sampled_auprc_drop_mean',
                *metric_cols,
            ]
        for metric_col in metric_cols:
            if metric_col == rank_col:
                continue
            if metric_col not in out.columns:
                continue
            metric_png = os.path.join(args.output_dir, f'{tag}.top{topk}.{metric_col}.png')
            ok = plot_topk_metric(out, 'combo', metric_col, topk, k, metric_png)
            if ok:
                print('saved:', metric_png)

    print(f'top {topk}:')
    print(out.head(topk).to_string(index=False))


if __name__ == '__main__':
    main()
