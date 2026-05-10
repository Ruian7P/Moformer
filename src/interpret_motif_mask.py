import argparse
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
    p = argparse.ArgumentParser(description='Motif occlusion analysis for Moformer-P checkpoints.')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--motif_path', type=str, required=True)
    p.add_argument('--cell', type=str, default='K562')
    p.add_argument('--expr_table', type=str, default='./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv')
    p.add_argument('--split_table', type=str, default='./data/leave_chrom_out_crossvalidation_split_18377genes.csv')
    p.add_argument('--fold', type=str, default='enformer')
    p.add_argument('--split', type=str, default='test', choices=['all', 'train', 'valid', 'test'])
    p.add_argument('--output_dir', type=str, default='./results/motif_occlusion')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--max_genes', type=int, default=0, help='0 means use all genes in selected split.')
    p.add_argument('--group_mode', type=str, default='motif', choices=['motif', 'column'])
    p.add_argument('--mask_value', type=float, default=0.0)
    p.add_argument('--save_delta_matrix', action='store_true')
    p.add_argument('--task', type=str, default='auto', choices=['auto', 'reg', 'cls'])
    p.add_argument('--expr_threshold', type=float, default=0.0)
    p.add_argument('--cls_prob_threshold', type=float, default=0.5)
    p.add_argument('--topk', type=int, default=10)

    p.add_argument('--motif_log1p', action='store_true')
    p.add_argument('--motif_zscore', action='store_true')
    p.add_argument('--motif_svd_dim', type=int, default=0)
    p.add_argument('--motif_multitoken_include_global', action='store_true')
    p.add_argument('--head', type=int, default=4, help='MultiheadAttention nhead; training default is 4.')
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


def build_motif_token_masks(motif_columns, include_global=True):
    bin_to_idx = {}
    global_idx = []
    for i, col in enumerate(motif_columns):
        if '__bin' in col:
            try:
                b = int(str(col).rsplit('__bin', 1)[1])
            except Exception:
                global_idx.append(i)
                continue
            bin_to_idx.setdefault(b, []).append(i)
        else:
            global_idx.append(i)
    if len(bin_to_idx) == 0:
        return None
    token_indices = [bin_to_idx[b] for b in sorted(bin_to_idx.keys())]
    if include_global and len(global_idx) > 0:
        token_indices.append(global_idx)
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


def build_groups(columns, mode='motif'):
    groups = OrderedDict()
    if mode == 'column':
        for i, c in enumerate(columns):
            groups[str(c)] = [i]
        return groups
    for i, c in enumerate(columns):
        col = str(c)
        base = re.sub(r'__bin\d+$', '', col)
        base = re.sub(r'__global$', '', base)
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
            if rna_values is None:
                rna_t = None
            else:
                rna_t = torch.from_numpy(rna_values[i:j]).float().to(device)
            p, _ = model(rna_feats=rna_t, motif_feats=motif_t)
            preds.append(p.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def maybe_plot_topk(summary_df, out_png, task='reg', topk=10):
    topk = max(1, min(int(topk), len(summary_df)))
    if task == 'cls':
        plot_df = summary_df.sort_values('acc_drop', ascending=False).head(topk).iloc[::-1]
        vals = plot_df['acc_drop'].values
        xlab = 'ACC drop (acc_base - acc_mask)'
        title = f'Top {topk} Motif Groups by ACC Drop'
    else:
        plot_df = summary_df.sort_values('mean_abs_delta', ascending=False).head(topk).iloc[::-1]
        vals = plot_df['mean_abs_delta'].values
        xlab = 'Mean |delta prediction|'
        title = f'Top {topk} Motif Groups by Mean |Delta|'

    labels = plot_df['group'].astype(str).tolist()
    fig_h = max(5, 0.45 * topk + 1.2)
    plt.figure(figsize=(10, fig_h))
    plt.barh(labels, vals)
    plt.xlabel(xlab)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


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
    if args.task == 'auto':
        task = 'cls' if '.cls.' in Path(args.checkpoint).name else 'reg'
    else:
        task = args.task
    print('task:', task)

    split_df = pd.read_csv(args.split_table, index_col=0)
    fold_col = f'fold_{args.fold}'
    if fold_col not in split_df.columns:
        raise ValueError(f'Fold column not found: {fold_col}')
    train_ids = split_df[split_df[fold_col] == 'train'].index.tolist()

    motif_df = pd.read_csv(args.motif_path, sep='\t', comment='#', index_col=0, engine='python')
    motif_df = motif_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    motif_df = preprocess_motif_features(
        motif_df,
        train_ids=train_ids,
        use_log1p=args.motif_log1p,
        use_zscore=args.motif_zscore,
        svd_dim=args.motif_svd_dim,
    )

    expr_df = pd.read_csv(args.expr_table, index_col='gene_id')
    if args.split == 'all':
        split_ids = split_df.index.tolist()
    else:
        split_ids = split_df[split_df[fold_col] == args.split].index.tolist()

    if cfg['use_rna_feats']:
        gene_ids = [gid for gid in split_ids if gid in motif_df.index and gid in expr_df.index]
    else:
        gene_ids = [gid for gid in split_ids if gid in motif_df.index]
    if args.max_genes > 0:
        gene_ids = gene_ids[: args.max_genes]
    if len(gene_ids) == 0:
        raise ValueError('No genes left after filtering. Check fold/split and input tables.')
    print('n_genes:', len(gene_ids), 'n_motif_cols:', motif_df.shape[1])

    motif_df = motif_df.loc[gene_ids]
    motif_cols = list(motif_df.columns)
    motif_values = motif_df.values.astype(np.float32)
    rna_values = None
    if cfg['use_rna_feats']:
        rna_values = (
            expr_df.loc[gene_ids, RNA_FEAT_COLS]
            .astype(float)
            .values.astype(np.float32)
        )
    y_true = None
    if task == 'cls':
        y_true = (
            expr_df.loc[gene_ids, f'Actual_{args.cell}'].astype(float).values > float(args.expr_threshold)
        ).astype(int)

    motif_token_masks = None
    if cfg['has_multitoken']:
        if 'motif_token_masks' in state_dict:
            motif_token_masks = state_dict['motif_token_masks'].detach().cpu().numpy().astype(np.float32)
        else:
            motif_token_masks = build_motif_token_masks(
                motif_cols,
                include_global=args.motif_multitoken_include_global,
            )
            if motif_token_masks is None:
                raise ValueError(
                    'Checkpoint expects multi-token motif encoder, but motif columns do not contain __bin structure.'
                )

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

    if motif_values.shape[1] != cfg['motif_feat_dim']:
        raise ValueError(
            f'Motif feature dimension mismatch: table={motif_values.shape[1]} checkpoint={cfg["motif_feat_dim"]}'
        )

    base_pred = predict_batches(
        model=model,
        motif_values=motif_values,
        rna_values=rna_values,
        batch_size=args.batch_size,
        device=device,
    )
    base_df = pd.DataFrame({'gene_id': gene_ids, 'pred_base': base_pred})
    if task == 'cls':
        base_df['prob_base'] = 1.0 / (1.0 + np.exp(-base_df['pred_base'].values))
        base_metrics = compute_binary_metrics(y_true, base_pred, threshold=float(args.cls_prob_threshold))
        base_acc = float(base_metrics['acc'])
        print(
            f"baseline metrics: acc={base_metrics['acc']:.6f} "
            f"auroc={base_metrics['auroc']:.6f} auprc={base_metrics['auprc']:.6f}"
        )
    else:
        base_metrics = None
        base_acc = None

    groups = build_groups(motif_cols, mode=args.group_mode)
    print('n_groups:', len(groups), 'group_mode:', args.group_mode)

    rows = []
    delta_matrix = np.zeros((len(gene_ids), len(groups)), dtype=np.float32)
    for gi, (gname, idxs) in enumerate(groups.items()):
        x_masked = motif_values.copy()
        x_masked[:, idxs] = args.mask_value
        pred_mask = predict_batches(
            model=model,
            motif_values=x_masked,
            rna_values=rna_values,
            batch_size=args.batch_size,
            device=device,
        )
        delta = pred_mask - base_pred
        delta_matrix[:, gi] = delta.astype(np.float32)
        row = {
            'group': gname,
            'n_cols_masked': len(idxs),
            'mean_delta': float(np.mean(delta)),
            'mean_abs_delta': float(np.mean(np.abs(delta))),
            'median_delta': float(np.median(delta)),
            'p10_delta': float(np.quantile(delta, 0.10)),
            'p90_delta': float(np.quantile(delta, 0.90)),
        }
        if task == 'cls':
            m = compute_binary_metrics(y_true, pred_mask, threshold=float(args.cls_prob_threshold))
            row['acc_mask'] = float(m['acc'])
            row['delta_acc'] = float(m['acc'] - base_acc)
            row['acc_drop'] = float(base_acc - m['acc'])
        rows.append(row)
        if (gi + 1) % 50 == 0 or (gi + 1) == len(groups):
            print(f'processed {gi + 1}/{len(groups)} groups')

    if task == 'cls':
        summary_df = pd.DataFrame(rows).sort_values('acc_drop', ascending=False).reset_index(drop=True)
    else:
        summary_df = pd.DataFrame(rows).sort_values('mean_abs_delta', ascending=False).reset_index(drop=True)
    stem = Path(args.checkpoint).stem
    out_prefix = f'{stem}.fold_{args.fold}.{args.split}.mask_{args.group_mode}'
    summary_path = os.path.join(args.output_dir, f'{out_prefix}.summary.csv')
    base_path = os.path.join(args.output_dir, f'{out_prefix}.baseline.csv')
    top10_png = os.path.join(args.output_dir, f'{out_prefix}.top{args.topk}.png')

    base_df.to_csv(base_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    maybe_plot_topk(summary_df, top10_png, task=task, topk=args.topk)

    if args.save_delta_matrix:
        delta_df = pd.DataFrame(delta_matrix, index=gene_ids, columns=list(groups.keys()))
        delta_df.index.name = 'gene_id'
        delta_path = os.path.join(args.output_dir, f'{out_prefix}.delta_matrix.csv')
        delta_df.to_csv(delta_path)
        print('saved:', delta_path)

    print('saved:', base_path)
    print('saved:', summary_path)
    print('saved:', top10_png)
    if task == 'cls':
        print(f'top {args.topk} groups by ACC drop:')
    else:
        print(f'top {args.topk} groups by mean_abs_delta:')
    print(summary_df.head(args.topk).to_string(index=False))


if __name__ == '__main__':
    main()
