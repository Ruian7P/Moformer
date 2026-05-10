import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from EPInformer.models_abc import Moformer_P
from scripts.classification_utils import (
    EarlyStoppingConfig,
    EarlyStoppingMetric,
    compute_binary_metrics,
    resolve_fold_ids,
    seed_everything,
)


def preprocess_motif_features(motif_df, train_ids, use_log1p=False, use_zscore=False, svd_dim=0, eps=1e-6):
    if motif_df is None:
        return None
    out = motif_df.copy()
    if use_log1p:
        out = np.log1p(np.maximum(out.values, 0.0))
        out = pd.DataFrame(out, index=motif_df.index, columns=motif_df.columns)
    if use_zscore:
        train_ids = [eid for eid in train_ids if eid in out.index]
        if len(train_ids) > 0:
            train_block = out.loc[train_ids]
            mu = train_block.mean(axis=0)
            sigma = train_block.std(axis=0).replace(0, np.nan)
            out = (out - mu) / (sigma + eps)
            out = out.fillna(0.0)
    if int(svd_dim) > 0 and out.shape[1] > int(svd_dim):
        train_ids = [eid for eid in train_ids if eid in out.index]
        if len(train_ids) > 1:
            n_comp = min(int(svd_dim), max(1, len(train_ids) - 1), max(1, out.shape[1] - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd.fit(out.loc[train_ids].values)
            transformed = svd.transform(out.values)
            cols = [f'motif_svd_{i}' for i in range(n_comp)]
            out = pd.DataFrame(transformed, index=out.index, columns=cols)
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
        return None, None
    token_names = []
    token_indices = []
    for b in sorted(bin_to_idx.keys()):
        token_names.append(f'bin{b}')
        token_indices.append(bin_to_idx[b])
    if include_global and len(global_idx) > 0:
        token_names.append('global')
        token_indices.append(global_idx)
    d = len(motif_columns)
    masks = np.zeros((len(token_indices), d), dtype=np.float32)
    for t, idxs in enumerate(token_indices):
        masks[t, idxs] = 1.0
    return masks, token_names


class MoformerPClsDataset(Dataset):
    def __init__(self, expr_df, motif_df, cell='K562', expr_threshold=0.0, use_rna_feats=True):
        common_ids = expr_df.index.intersection(motif_df.index)
        self.expr_df = expr_df.loc[common_ids]
        self.motif_df = motif_df.loc[common_ids]
        self.cell = cell
        self.expr_threshold = float(expr_threshold)
        self.use_rna_feats = bool(use_rna_feats)
        self.gene_ids = common_ids.tolist()

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        gid = self.gene_ids[idx]
        if self.use_rna_feats:
            rna_feats = np.array(
                self.expr_df.loc[gid][
                    [
                        'UTR5LEN_log10zscore',
                        'CDSLEN_log10zscore',
                        'INTRONLEN_log10zscore',
                        'UTR3LEN_log10zscore',
                        'UTR5GC',
                        'CDSGC',
                        'UTR3GC',
                        'ORFEXONDENSITY',
                    ]
                ].values.astype(float)
            ).flatten()
        else:
            rna_feats = np.zeros((8,), dtype=np.float32)
        motif_feats = self.motif_df.loc[gid].values.astype(np.float32)
        expr = float(self.expr_df.loc[gid, f'Actual_{self.cell}'])
        y = 1.0 if expr > self.expr_threshold else 0.0
        return rna_feats.astype(np.float32), motif_feats, np.float32(y), gid


def evaluate(model, ds, device='cuda', batch_size=64, use_rna_feats=True):
    loader = data_utils.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    model.eval()
    y_true, logits, gids = [], [], []
    with torch.no_grad():
        for rna_feats, motif_feats, y, gid in loader:
            motif_feats = motif_feats.float().to(device)
            if use_rna_feats:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            logit, _ = model(rna_feats=rna_feats, motif_feats=motif_feats)
            y_true.extend(y.numpy().tolist())
            logits.extend(logit.detach().cpu().numpy().tolist())
            gids.extend(gid)
    metrics = compute_binary_metrics(np.asarray(y_true), np.asarray(logits))
    return metrics, np.asarray(y_true), np.asarray(logits), gids


def train_one_fold(
    model,
    fold_name,
    train_ds,
    valid_ds,
    test_ds,
    saved_model_path,
    model_name,
    device='cuda',
    epochs=50,
    lr=5e-4,
    batch_size=64,
    early_stop_patience=10,
    use_rna_feats=True,
):
    train_loader = data_utils.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, drop_last=True
    )

    ys = []
    for _, _, yb, _ in data_utils.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0):
        ys.extend(yb.numpy().tolist())
    ys = np.asarray(ys)
    pos = float((ys == 1).sum())
    neg = float((ys == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    print(f'train positives: {int(pos)} negatives: {int(neg)} pos_weight: {float(pos_weight.item()):.4f}')

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    stopper = EarlyStoppingMetric(
        EarlyStoppingConfig(
            patience=early_stop_patience,
            mode='max',
            delta=0.0,
            path=os.path.join(saved_model_path, f'fold_{fold_name}_best_{model_name}_checkpoint.pt'),
            verbose=True,
        )
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_logits, train_targets = [], []
        for rna_feats, motif_feats, y, _ in tqdm(train_loader):
            motif_feats = motif_feats.float().to(device)
            y = y.float().to(device)
            if use_rna_feats:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            optimizer.zero_grad()
            logit, _ = model(rna_feats=rna_feats, motif_feats=motif_feats)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            train_logits.extend(logit.detach().cpu().numpy().tolist())
            train_targets.extend(y.detach().cpu().numpy().tolist())

        train_metrics = compute_binary_metrics(np.asarray(train_targets), np.asarray(train_logits))
        val_metrics, _, _, _ = evaluate(model, valid_ds, device=device, batch_size=batch_size, use_rna_feats=use_rna_feats)
        print(
            f"[Epoch {epoch + 1}] loss: {running_loss / len(train_loader):.6f} "
            f"train_auprc: {train_metrics['auprc']:.6f} train_auroc: {train_metrics['auroc']:.6f} "
            f"train_acc: {train_metrics['acc']:.6f} "
            f"val_auprc: {val_metrics['auprc']:.6f} val_auroc: {val_metrics['auroc']:.6f} "
            f"val_acc: {val_metrics['acc']:.6f}"
        )
        stopper.step(val_metrics['auprc'], model, epoch)
        if stopper.early_stop:
            print('Early stopping')
            break

    ckpt = torch.load(os.path.join(saved_model_path, f'fold_{fold_name}_best_{model_name}_checkpoint.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics, y_true, logits, gids = evaluate(model, test_ds, device=device, batch_size=batch_size, use_rna_feats=use_rna_feats)
    print(
        f"test_auprc: {test_metrics['auprc']:.6f} test_auroc: {test_metrics['auroc']:.6f} "
        f"test_acc: {test_metrics['acc']:.6f}"
    )
    out = pd.DataFrame(
        {
            'gene_id': gids,
            'y_true': y_true,
            'logit': logits,
            'prob': 1.0 / (1.0 + np.exp(-logits)),
            'fold': fold_name,
        }
    )
    out.to_csv(os.path.join(saved_model_path, f'fold_{fold_name}_{model_name}_cls_predictions.csv'), index=False)
    return test_metrics, out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cuda_id', type=int, default=0)
    p.add_argument('--cell', type=str, default='K562', choices=['K562', 'GM12878', 'HepG2'])
    p.add_argument('--fold', type=str, default='enformer')
    p.add_argument(
        '--model_type',
        type=str,
        default='Moformer-P',
        choices=['Moformer-P', 'Moformer-P-rna'],
    )
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--expr_threshold', type=float, default=0.0)
    p.add_argument('--motif_path', type=str, required=True)
    # Backward-compatible switch; overridden by --model_type when needed.
    p.add_argument('--use_rna_feats', action='store_true')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--early_stop_patience', type=int, default=10)
    p.add_argument('--motif_log1p', action='store_true')
    p.add_argument('--motif_zscore', action='store_true')
    p.add_argument('--motif_svd_dim', type=int, default=0)
    p.add_argument('--motif_multitoken', action='store_true')
    p.add_argument('--motif_multitoken_include_global', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    seed_everything(args.seed)
    print('seed:', args.seed)
    use_rna_feats = (args.model_type == 'Moformer-P-rna') or args.use_rna_feats
    print('model_type:', args.model_type, 'use_rna_feats:', use_rna_feats)
    device = 'cuda'

    split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
    fold_ids = resolve_fold_ids(split_df, args.fold)
    expr_df = pd.read_csv('./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv', index_col='gene_id')

    motif_df = pd.read_csv(args.motif_path, sep='\t', index_col=0).apply(pd.to_numeric, errors='coerce').fillna(0.0)

    all_results = []
    for fi in fold_ids:
        fold_col = f'fold_{fi}'
        train_ids = split_df[split_df[fold_col] == 'train'].index
        valid_ids = split_df[split_df[fold_col] == 'valid'].index
        test_ids = split_df[split_df[fold_col] == 'test'].index

        motif_fold = motif_df.copy()
        if args.motif_log1p or args.motif_zscore or args.motif_svd_dim > 0:
            motif_fold = preprocess_motif_features(
                motif_fold,
                train_ids=list(train_ids),
                use_log1p=args.motif_log1p,
                use_zscore=args.motif_zscore,
                svd_dim=args.motif_svd_dim,
            )

        motif_token_masks = None
        if args.motif_multitoken:
            if args.motif_svd_dim > 0:
                print('Warning: --motif_multitoken disabled because --motif_svd_dim > 0.')
            else:
                motif_token_masks, token_names = build_motif_token_masks(
                    list(motif_fold.columns),
                    include_global=args.motif_multitoken_include_global,
                )
                if motif_token_masks is None:
                    print('Warning: no __bin columns found; fallback to single-token motif encoding.')
                else:
                    token_sizes = motif_token_masks.sum(axis=1).astype(int).tolist()
                    print('motif multitoken enabled:', ', '.join([f'{n}:{s}' for n, s in zip(token_names, token_sizes)]))

        ds = MoformerPClsDataset(
            expr_df=expr_df,
            motif_df=motif_fold,
            cell=args.cell,
            expr_threshold=args.expr_threshold,
            use_rna_feats=use_rna_feats,
        )
        id_to_idx = {g: i for i, g in enumerate(ds.gene_ids)}
        train_ds = Subset(ds, [id_to_idx[g] for g in train_ids if g in id_to_idx])
        valid_ds = Subset(ds, [id_to_idx[g] for g in valid_ids if g in id_to_idx])
        test_ds = Subset(ds, [id_to_idx[g] for g in test_ids if g in id_to_idx])

        z_tr = float(np.mean([np.all(ds.motif_df.loc[g].values == 0) for g in train_ids if g in ds.motif_df.index]) * 100) if len(train_ds) > 0 else float('nan')
        z_va = float(np.mean([np.all(ds.motif_df.loc[g].values == 0) for g in valid_ids if g in ds.motif_df.index]) * 100) if len(valid_ds) > 0 else float('nan')
        z_te = float(np.mean([np.all(ds.motif_df.loc[g].values == 0) for g in test_ids if g in ds.motif_df.index]) * 100) if len(test_ds) > 0 else float('nan')
        print('motif_feat_dim:', ds.motif_df.shape[1], 'motif_zero% train/valid/test:', f'{z_tr:.2f}/{z_va:.2f}/{z_te:.2f}')

        model = Moformer_P(
            out_dim=64,
            n_enhancer=60,
            useBN=False,
            usePromoterSignal=False,
            useFeat=use_rna_feats,
            motif_feat_dim=ds.motif_df.shape[1],
            motif_token_masks=motif_token_masks,
        ).to(device)

        motif_tag = Path(args.motif_path).stem
        if args.motif_log1p:
            motif_tag += '.log1p'
        if args.motif_zscore:
            motif_tag += '.zscore'
        if args.motif_svd_dim > 0:
            motif_tag += f'.svd{args.motif_svd_dim}'
        if args.motif_multitoken and motif_token_masks is not None:
            motif_tag += f'.mtok{motif_token_masks.shape[0]}'
        model.name = f"{model.name}.{args.cell}.cls.thr{args.expr_threshold}.seed{args.seed}.{motif_tag}"

        saved_model_path = './results/Moformer-P-cls/'
        os.makedirs(saved_model_path, exist_ok=True)
        metrics, pred_df = train_one_fold(
            model=model,
            fold_name=fi,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            saved_model_path=saved_model_path,
            model_name=model.name,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            early_stop_patience=args.early_stop_patience,
            use_rna_feats=use_rna_feats,
        )
        metrics_row = {'fold': fi, **metrics}
        all_results.append(metrics_row)
        pred_df['model_name'] = model.name

    res_df = pd.DataFrame(all_results)
    avg_row = {'fold': 'avg'}
    for metric in ('auprc', 'auroc', 'acc'):
        avg_row[metric] = float(res_df[metric].mean())
    res_df = pd.concat([res_df, pd.DataFrame([avg_row])], ignore_index=True)
    print('Summary:')
    print(res_df)
    print(f"avg acc: {avg_row['acc']:.6f}")
    tag = f"{args.cell}.thr{args.expr_threshold}.seed{args.seed}"
    res_df.to_csv(f'./results/Moformer-P-cls/{args.model_type}_cls_summary.{tag}.csv', index=False)


if __name__ == '__main__':
    main()
