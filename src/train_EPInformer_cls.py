import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from EPInformer.models_abc import EPInformer_promoter_v2, EPInformer_v2
from scripts.classification_utils import (
    EarlyStoppingConfig,
    EarlyStoppingMetric,
    compute_binary_metrics,
    resolve_fold_ids,
    seed_everything,
)


def one_hot_encode_dna(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    lut = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, ch in enumerate(seq.upper()):
        j = lut.get(ch, None)
        if j is not None:
            arr[i, j] = 1.0
    return arr


class PromoterClsDataset(Dataset):
    def __init__(
        self,
        expr_df: pd.DataFrame,
        cell: str = 'K562',
        expr_threshold: float = 0.0,
        use_rna_feats: bool = False,
    ):
        self.expr_df = expr_df
        self.cell = cell
        self.expr_threshold = float(expr_threshold)
        self.use_rna_feats = bool(use_rna_feats)
        self.gene_ids = expr_df.index.tolist()

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        gid = self.gene_ids[idx]
        seq = self.expr_df.loc[gid, 'promoter_2k']
        pe_seq = one_hot_encode_dna(seq)[np.newaxis, :, :]  # [1, 2000, 4]
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
        expr = float(self.expr_df.loc[gid, f'Actual_{self.cell}'])
        y = 1.0 if expr > self.expr_threshold else 0.0
        return pe_seq, rna_feats.astype(np.float32), np.float32(y), gid


def evaluate(model, ds, device='cuda', batch_size=64, use_rna_feats=False):
    loader = data_utils.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    model.eval()
    y_true, logits, gids = [], [], []
    with torch.no_grad():
        for pe_seq, rna_feats, y, gid in loader:
            pe_seq = pe_seq.float().to(device)
            if use_rna_feats:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            logit, _ = model(pe_seq, rna_feats=rna_feats, enh_feats=None)
            y_true.extend(y.numpy().tolist())
            logits.extend(logit.detach().cpu().numpy().tolist())
            gids.extend(gid)
    metrics = compute_binary_metrics(np.asarray(y_true), np.asarray(logits))
    return metrics, np.asarray(y_true), np.asarray(logits), gids


def train_one_fold(
    model,
    fold_name: str,
    train_ds,
    valid_ds,
    test_ds,
    saved_model_path: str,
    model_name: str,
    device='cuda',
    epochs=50,
    lr=5e-4,
    batch_size=64,
    early_stop_patience=10,
    use_rna_feats=False,
):
    train_loader = data_utils.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, drop_last=True
    )
    # Class imbalance handling from train split.
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
        for pe_seq, rna_feats, y, _ in tqdm(train_loader):
            pe_seq = pe_seq.float().to(device)
            y = y.float().to(device)
            if use_rna_feats:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            optimizer.zero_grad()
            logit, _ = model(pe_seq, rna_feats=rna_feats, enh_feats=None)
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
    test_metrics, y_true, logits, gids = evaluate(
        model, test_ds, device=device, batch_size=batch_size, use_rna_feats=use_rna_feats
    )
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
        default='EPInformer-promoter-v2',
        choices=['EPInformer-promoter-v2', 'EPInformer-v2'],
    )
    p.add_argument('--use_rna_feats', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--expr_threshold', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--early_stop_patience', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    seed_everything(args.seed)
    print('seed:', args.seed)
    use_rna_feats = (args.model_type == 'EPInformer-v2') or args.use_rna_feats
    print('model_type:', args.model_type, 'use_rna_feats:', use_rna_feats)
    device = 'cuda'

    split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
    fold_ids = resolve_fold_ids(split_df, args.fold)
    expr_df = pd.read_csv('./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv', index_col='gene_id')

    all_results = []
    for fi in fold_ids:
        fold_col = f'fold_{fi}'
        train_ids = split_df[split_df[fold_col] == 'train'].index.intersection(expr_df.index)
        valid_ids = split_df[split_df[fold_col] == 'valid'].index.intersection(expr_df.index)
        test_ids = split_df[split_df[fold_col] == 'test'].index.intersection(expr_df.index)

        fold_df = expr_df.loc[train_ids.union(valid_ids).union(test_ids)].copy()
        ds = PromoterClsDataset(
            fold_df,
            cell=args.cell,
            expr_threshold=args.expr_threshold,
            use_rna_feats=use_rna_feats,
        )
        id_to_idx = {g: i for i, g in enumerate(ds.gene_ids)}
        train_ds = Subset(ds, [id_to_idx[g] for g in train_ids if g in id_to_idx])
        valid_ds = Subset(ds, [id_to_idx[g] for g in valid_ids if g in id_to_idx])
        test_ds = Subset(ds, [id_to_idx[g] for g in test_ids if g in id_to_idx])

        if args.model_type == 'EPInformer-promoter-v2':
            model = EPInformer_promoter_v2(
                out_dim=64,
                n_enhancer=60,
                useBN=False,
                usePromoterSignal=False,
                useFeat=False,
            ).to(device)
        elif args.model_type == 'EPInformer-v2':
            model = EPInformer_v2(
                out_dim=64,
                n_enhancer=0,
                useBN=False,
                usePromoterSignal=False,
                useFeat=True,
                n_extraFeat=0,
            ).to(device)
        else:
            raise ValueError(f'Unsupported model_type: {args.model_type}')
        model.name = f"{model.name}.{args.cell}.cls.thr{args.expr_threshold}.seed{args.seed}"
        saved_model_path = './results/EPInformer-promoter-cls/'
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
    res_df.to_csv(
        f'./results/EPInformer-promoter-cls/{args.model_type}_cls_summary.{args.cell}.thr{args.expr_threshold}.seed{args.seed}.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
