import os
import sys
import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from EPInformer.models_abc import Moformer, Moformer_P, enhancer_predictor_256bp


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class promoter_enhancer_dataset_moformer(Dataset):
    """Dataset for Moformer.

    Key difference from EPInformer/MoPInformer:
    - No promoter sequence is returned.
    - Promoter information comes from motif_feats only.
    """

    def __init__(
        self,
        cell_type='K562',
        expr_type='RNA',
        n_enh_feats=3,
        disable_enh=False,
        distance_thr=None,
        max_n_enh=200,
        use_prm_signal=False,
        rm_prm_seq=False,
        motif_feat_path=None,
    ):
        self.data_h5 = h5py.File(f'/dev/shm/data/{cell_type}_200CREs-gene_RPM_4feats.hdf5', 'r')
        self.rm_prm_seq = rm_prm_seq
        self.cell_type = cell_type
        self.n_enh_feats = n_enh_feats
        self.expr_type = expr_type
        self.disable_enh = disable_enh
        self.distance_thr = distance_thr
        self.max_n_enh = max_n_enh
        self.use_prm_signal = use_prm_signal
        self.expr_df = pd.read_csv(
            './data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv',
            index_col='gene_id',
        )
        if cell_type == 'K562':
            promoter_df = pd.read_csv(
                './data/K562_ABC_EGLinks/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt',
                sep='\t',
                index_col='name',
            )
        elif cell_type == 'GM12878':
            promoter_df = pd.read_csv(
                './epinformer_data_20250503/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt',
                sep='\t',
                index_col='name',
            )
        elif cell_type == 'HepG2':
            promoter_df = pd.read_csv(
                './epinformer_data_20250503/HepG2/DNase_ENCFF691HJY_Neighborhoods/GeneList.txt',
                sep='\t',
                index_col='name',
            )
        elif cell_type == 'NHEK':
            promoter_df = pd.read_csv(
                './data/NHEK/DNase_ENCFF862NDZ_Neighborhoods/GeneList.txt',
                sep='\t',
                index_col='name',
            )
        elif cell_type == 'HUVEC':
            promoter_df = pd.read_csv(
                './data/HUVEC/DNase_ENCFF091KTX_Neighborhoods/GeneList.txt',
                sep='\t',
                index_col='name',
            )
        elif cell_type == 'H1':
            promoter_df = pd.read_csv(
                './data/H1/DNase_ENCFF761ZRE_Neighborhoods/GeneList.txt',
                sep='\t',
                index_col='name',
            )
        else:
            raise ValueError(f'Cell not found: {cell_type}')
        promoter_df['promoter_activity'] = np.sqrt(
            promoter_df['DHS.RPKM.TSS1Kb'] * promoter_df['H3K27ac.RPKM.TSS1Kb']
        )
        self.promoter_df = promoter_df

        self.motif_df = None
        self.motif_feat_dim = 0
        if motif_feat_path is not None and os.path.exists(motif_feat_path):
            try:
                motif_df = pd.read_csv(motif_feat_path, sep='\t', comment='#', index_col=0, engine='python')
            except Exception:
                motif_df = pd.read_csv(motif_feat_path, sep='\t', index_col=0, engine='python')
            motif_df = motif_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            self.motif_df = motif_df
            self.motif_feat_dim = motif_df.shape[1]

    def __len__(self):
        return len(self.data_h5['ensid'])

    def __getitem__(self, idx):
        sample_ensid = self.data_h5['ensid'][idx].decode()
        enh_ohe = self.data_h5['enhancers_ohe'][idx]
        enh_feats = self.data_h5['enhancers_feat'][idx][:, :]
        prm_signal = np.log(1 + np.array([self.promoter_df.loc[sample_ensid, 'promoter_activity']]))

        if self.n_enh_feats == 0:
            enh_feats = np.zeros_like(
                np.concatenate([abs(enh_feats[:, [0]]), enh_feats[:, [3]], enh_feats[:, [-1]]], axis=1)[:, :1]
            )
        else:
            enh_feats = np.concatenate([abs(enh_feats[:, [0]]), enh_feats[:, [3]], enh_feats[:, [-1]]], axis=1)[
                :, : self.n_enh_feats
            ]

        rna_feats = np.array(
            self.expr_df.loc[sample_ensid][
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
        if self.use_prm_signal:
            rna_feats = np.concatenate([rna_feats, prm_signal])

        if self.distance_thr is not None:
            enh_ohe_new = np.zeros((self.max_n_enh, 2000, 4), dtype=enh_ohe.dtype)
            enh_feats_new = np.zeros((self.max_n_enh, enh_feats.shape[-1]), dtype=enh_feats.dtype)
            new_i = 0
            for i in range(enh_ohe.shape[0]):
                if not self.rm_prm_seq:
                    keep = abs(enh_feats[i][0]) <= self.distance_thr
                else:
                    keep = abs(enh_feats[i][0]) <= self.distance_thr and abs(enh_feats[i][0]) >= 1000
                if keep:
                    enh_ohe_new[new_i] = enh_ohe[i]
                    enh_feats_new[new_i] = enh_feats[i]
                    new_i += 1
                if new_i >= self.max_n_enh:
                    break
            enh_ohe = enh_ohe_new
            enh_feats = enh_feats_new

        if self.disable_enh:
            enh_ohe = np.zeros_like(enh_ohe)
            enh_feats = np.zeros_like(enh_feats)

        if self.expr_type == 'CAGE':
            expr = np.log10(self.expr_df.loc[sample_ensid, self.cell_type + '_CAGE_128*3_sum'] + 1)
        else:
            expr = self.expr_df.loc[sample_ensid, 'Actual_' + self.cell_type]

        prm_feats = np.ones_like(enh_feats[[0]])
        if self.use_prm_signal and self.n_enh_feats == 3:
            prm_feats[0, 1] = prm_signal
        pe_feats = np.concatenate([prm_feats, enh_feats], axis=0)

        if self.motif_df is not None and sample_ensid in self.motif_df.index:
            motif_feats = self.motif_df.loc[sample_ensid].values.astype(np.float32)
        else:
            motif_feats = np.zeros((self.motif_feat_dim,), dtype=np.float32)
        return enh_ohe, rna_feats, pe_feats, motif_feats, expr, sample_ensid


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


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch_i):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_i)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}',
                'best_score',
                self.best_score,
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_i)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch_i):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(
            {'epoch': epoch_i, 'model_state_dict': model.state_dict(), 'loss': val_loss},
            self.path,
        )
        print('Saving ckpt at', self.path)
        self.val_loss_min = val_loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(
    net,
    training_dataset,
    fold_i,
    saved_model_path='./models/',
    learning_rate=1e-4,
    fixed_encoder=False,
    valid_dataset=None,
    model_name='',
    batch_size=64,
    device='cuda',
    stratify=None,
    EPOCHS=100,
    valid_size=1000,
    early_stop_patience=5,
):
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    if valid_dataset is not None:
        train_ds = training_dataset
        valid_ds = valid_dataset
    else:
        train_idx, val_idx = train_test_split(
            list(range(len(training_dataset))),
            test_size=valid_size,
            shuffle=True,
            random_state=66,
            stratify=stratify,
        )
        train_ds = Subset(training_dataset, train_idx)
        valid_ds = Subset(training_dataset, val_idx)

    if fixed_encoder:
        print('fixed parameter of encoder')
        for name, value in net.named_parameters():
            if name.startswith('seq_encoder'):
                value.requires_grad = False

    print("fold", fold_i, "training data:", len(train_ds), "validated data:", len(valid_ds), 'total data:', len(training_dataset))
    trainloader = data_utils.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    early_stopping = EarlyStopping(
        patience=early_stop_patience,
        verbose=True,
        path=saved_model_path + "/fold_" + str(fold_i) + "_best_" + model_name + "_checkpoint.pt",
    )
    L_expr = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-6)

    for epoch in range(EPOCHS):
        net.train()
        print('learning rate:', get_lr(optimizer))
        running_loss = 0
        loss_e = 0
        for data in tqdm(trainloader):
            optimizer.zero_grad()
            enh_seqs, rna_feats, enh_feats, motif_feats, y_expr, _ = data
            enh_seqs = enh_seqs.float().to(device)
            if net.useFeat:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            enh_feats = enh_feats.float().to(device)
            motif_feats = motif_feats.float().to(device)
            y_expr = y_expr.float().to(device)

            pred_expr, _ = net(enh_seqs, enh_feats=enh_feats, rna_feats=rna_feats, motif_feats=motif_feats)
            loss_expr = L_expr(pred_expr, y_expr)
            loss = loss_expr
            loss.backward()
            optimizer.step()
            loss_e += loss_expr.item()
            running_loss += loss.item()

        print('[Epoch %d] loss: %.9f' % (epoch + 1, running_loss / len(trainloader)))
        print('Training Loss: expression loss:', loss_e / len(trainloader))
        _, val_r2_all, _ = validate(net, valid_ds, device=device)
        print('Valdaition R square all:', val_r2_all)
        early_stopping(-val_r2_all, net, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def validate(net, valid_ds, batch_size=16, device='cuda'):
    validloader = data_utils.DataLoader(valid_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    net.eval()
    L_expr = nn.SmoothL1Loss()
    with torch.no_grad():
        preds = []
        actual = []
        loss_e = 0
        for data in tqdm(validloader):
            enh_seqs, rna_feats, enh_feats, motif_feats, y_expr, _ = data
            enh_seqs = enh_seqs.float().to(device)
            if net.useFeat:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            enh_feats = enh_feats.float().to(device)
            motif_feats = motif_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(enh_seqs, enh_feats=enh_feats, rna_feats=rna_feats, motif_feats=motif_feats)
            preds += list(pred_expr.flatten().cpu().detach().numpy())
            actual += list(y_expr.flatten().cpu().detach().numpy())
            loss_e += L_expr(pred_expr, y_expr).item()
    try:
        _, _, r_value, _, _ = stats.linregress(preds, actual)
        peasonr, _ = stats.pearsonr(preds, actual)
    except Exception:
        peasonr = 0
        r_value = 0
    mse = mean_squared_error(preds, actual)
    print('Validation loss expression loss:', loss_e / len(validloader))
    print("valid: mse", mse, "R_sqaure", r_value**2, 'peasonr', peasonr)
    return mse, r_value**2, peasonr


def test(net, test_ds, fold_i, model_name=None, saved_model_path=None, batch_size=64, device='cuda'):
    testloader = data_utils.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    if saved_model_path is not None:
        checkpoint = torch.load(
            saved_model_path + "/fold_" + str(fold_i) + "_best_" + model_name + "_checkpoint.pt",
            weights_only=False,
        )
        net.load_state_dict(checkpoint['model_state_dict'])
        print(model_name, 'loaded!')
    net.eval()
    with torch.no_grad():
        preds = []
        actual = []
        ensid_list = []
        for data in testloader:
            enh_seqs, rna_feats, enh_feats, motif_feats, y_expr, eid = data
            enh_seqs = enh_seqs.float().to(device)
            if net.useFeat:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            enh_feats = enh_feats.float().to(device)
            motif_feats = motif_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(enh_seqs, enh_feats=enh_feats, rna_feats=rna_feats, motif_feats=motif_feats)
            preds += list(pred_expr.flatten().cpu().detach().numpy())
            actual += list(y_expr.flatten().cpu().detach().numpy())
            ensid_list += eid

    _, _, r_value, _, _ = stats.linregress(preds, actual)
    peasonr, _ = stats.pearsonr(preds, actual)
    _ = mean_squared_error(preds, actual)
    print('\nPearson R:', peasonr)
    print('R_square:', r_value**2)
    sys.stdout.flush()
    df = pd.DataFrame(index=np.array(ensid_list).flatten())
    df['Pred'] = preds
    df['actual'] = actual
    df['fold_idx'] = fold_i
    pearsonr_we, _ = stats.pearsonr(df['Pred'], df['actual'])
    print('PearsonR:', pearsonr_we)
    if saved_model_path is not None:
        df.to_csv(saved_model_path + "/fold_" + str(fold_i) + "_" + model_name + "_predictions.csv")
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='model type', default='Moformer', choices=['Moformer', 'Moformer-P'])
    parser.add_argument('--cuda_id', type=int, help='cuda id', default=0)
    parser.add_argument('--expr_type', type=str, help='expression type', default='RNA', choices=['CAGE', 'RNA'])
    parser.add_argument('--n_enh_feats', type=int, help='number of enhancer features', default=3, choices=[1, 2, 3])
    parser.add_argument('--cell', type=str, help='cell type', default='K562', choices=['K562', 'GM12878', 'HepG2'])
    parser.add_argument('--use_prm_signal', type=bool, help='use promoter signal', default=False)
    parser.add_argument('--use_pretrained_encoder', type=bool, help='use pretrained encoder', default=False)
    parser.add_argument('--rm_prm_seq', type=bool, help='remove promoter-near enhancers', default=False)
    parser.add_argument('--fold', type=str, help='fold name without prefix, e.g. "1", "enformer", or "all"', default='enformer')
    parser.add_argument('--motif_path', type=str, help='path to promoter motif feature table', required=True)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
    parser.add_argument('--epochs', type=int, help='max number of epochs', default=50)
    parser.add_argument('--early_stop_patience', type=int, help='early stopping patience', default=5)
    parser.add_argument('--motif_log1p', action='store_true', help='apply log1p transform to motif features')
    parser.add_argument('--motif_zscore', action='store_true', help='z-score motif features using train-fold statistics')
    parser.add_argument('--motif_svd_dim', type=int, help='optional SVD dimension for motif features (0 to disable)', default=0)
    parser.add_argument('--seed', type=int, help='global random seed for reproducible training', default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    seed_everything(args.seed)
    print('seed:', args.seed)
    device = 'cuda'

    split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
    available_fold_cols = [c for c in split_df.columns if c.startswith('fold_')]
    if len(available_fold_cols) == 0:
        raise ValueError('No fold columns found in split file. Expected columns like fold_1 / fold_enformer.')
    if args.fold == 'all':
        fold_ids = [c.replace('fold_', '') for c in available_fold_cols]
    else:
        requested_col = f'fold_{args.fold}'
        if requested_col in split_df.columns:
            fold_ids = [args.fold]
        elif args.fold == 'borzoi':
            fallback = '1' if 'fold_1' in split_df.columns else available_fold_cols[0].replace('fold_', '')
            print(f'Warning: fold_borzoi not found. Falling back to fold_{fallback}.')
            fold_ids = [fallback]
        else:
            raise KeyError(f'{requested_col} not found. Available: {available_fold_cols}')

    os.makedirs('/dev/shm/data/', exist_ok=True)
    results = []
    expr_type = args.expr_type
    batch_size = 64
    max_n_enh = 60
    dist_thr = 100_000
    lr = args.lr
    cell_type = args.cell
    use_prm_signal = args.use_prm_signal
    model_dist = {'Moformer': Moformer, 'Moformer-P': Moformer_P}
    print('use_prm_signal:', use_prm_signal)

    for fi in fold_ids:
        fold_i = f'fold_{fi}'
        for use_rna_feats, rm_prm_seq in [(True, args.rm_prm_seq)]:
            for cell in [cell_type]:
                shm_path = f'/dev/shm/data/{cell}_200CREs-gene_RPM_4feats.hdf5'
                if not os.path.exists(shm_path):
                    print('copying data into /dev/shm/')
                    os.system(f'cp ./data/{cell}_200CREs-gene_RPM_4feats.hdf5 /dev/shm/data/')
                    print(os.path.exists(shm_path))

                ds = promoter_enhancer_dataset_moformer(
                    cell_type=cell,
                    expr_type=expr_type,
                    n_enh_feats=args.n_enh_feats,
                    distance_thr=dist_thr,
                    max_n_enh=max_n_enh,
                    use_prm_signal=use_prm_signal,
                    rm_prm_seq=rm_prm_seq,
                    motif_feat_path=args.motif_path,
                )
                train_ensid = split_df[split_df[fold_i] == 'train'].index
                valid_ensid = split_df[split_df[fold_i] == 'valid'].index
                test_ensid = split_df[split_df[fold_i] == 'test'].index
                if ds.motif_df is not None and (args.motif_log1p or args.motif_zscore or args.motif_svd_dim > 0):
                    ds.motif_df = preprocess_motif_features(
                        ds.motif_df,
                        train_ids=list(train_ensid),
                        use_log1p=args.motif_log1p,
                        use_zscore=args.motif_zscore,
                        svd_dim=args.motif_svd_dim,
                    )
                    ds.motif_feat_dim = ds.motif_df.shape[1]
                if ds.motif_df is not None:
                    def _zero_pct(ids):
                        ids = [eid for eid in ids if eid in ds.motif_df.index]
                        if len(ids) == 0:
                            return float('nan')
                        x = ds.motif_df.loc[ids].values
                        return float((x.sum(axis=1) == 0).mean() * 100)
                    z_tr = _zero_pct(train_ensid)
                    z_va = _zero_pct(valid_ensid)
                    z_te = _zero_pct(test_ensid)
                    print(
                        'motif_feat_dim:',
                        ds.motif_feat_dim,
                        'motif_zero% train/valid/test:',
                        f'{z_tr:.2f}/{z_va:.2f}/{z_te:.2f}',
                    )

                ensid_list = [eid.decode('utf-8') for eid in ds.data_h5['ensid'][:]]
                ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
                ensid_df['idx'] = np.arange(len(ensid_list))
                ensid_df = ensid_df.set_index('ensid')
                train_idx = ensid_df.loc[train_ensid]['idx']
                valid_idx = ensid_df.loc[valid_ensid]['idx']
                test_idx = ensid_df.loc[test_ensid]['idx']
                train_ds = Subset(ds, train_idx)
                valid_ds = Subset(ds, valid_idx)
                test_ds = Subset(ds, test_idx)

                if args.use_pretrained_encoder and args.model_type == 'Moformer':
                    print('Using pre-trained enhancer encoder')
                    pt_model_name = f'./pretrained_seqencoder_h3k27ac/fold_{fi}_best_enhancer_predictor_H3K27ac_256bp_{cell}_checkpoint.pt'
                    checkpoint = torch.load(pt_model_name, weights_only=False)
                    pretrained_convNet = enhancer_predictor_256bp()
                    pretrained_convNet.load_state_dict(checkpoint['model_state_dict'])
                    model = model_dist[args.model_type](
                        n_extraFeat=args.n_enh_feats,
                        pre_trained_encoder=pretrained_convNet.encoder,
                        useFeat=use_rna_feats,
                        out_dim=64,
                        n_enhancer=max_n_enh,
                        useBN=False,
                        usePromoterSignal=use_prm_signal,
                        motif_feat_dim=ds.motif_feat_dim,
                    ).to(device)
                    print('freezing the enhancer encoder parameters')
                    for name, value in model.named_parameters():
                        if name.startswith('seq_encoder'):
                            value.requires_grad = False
                else:
                    if args.use_pretrained_encoder and args.model_type == 'Moformer-P':
                        print('Warning: Moformer-P has no enhancer seq encoder; --use_pretrained_encoder is ignored.')
                    model = model_dist[args.model_type](
                        n_extraFeat=args.n_enh_feats,
                        pre_trained_encoder=None,
                        useFeat=use_rna_feats,
                        out_dim=64,
                        n_enhancer=max_n_enh,
                        useBN=False,
                        usePromoterSignal=use_prm_signal,
                        motif_feat_dim=ds.motif_feat_dim,
                    ).to(device)

                use_rna_feats_flag = 'rnafeats' if use_rna_feats else 'nornafeats'
                use_prm_signal_flag = 'prmsig' if use_prm_signal else 'nonprmsig'
                rm_prm_signal_flag = 'rmprmseq' if rm_prm_seq else 'nonrmprmseq'
                motif_tag = Path(args.motif_path).stem
                if args.motif_log1p:
                    motif_tag = motif_tag + '.log1p'
                if args.motif_zscore:
                    motif_tag = motif_tag + '.zscore'
                if args.motif_svd_dim > 0:
                    motif_tag = motif_tag + f'.svd{args.motif_svd_dim}'
                model.name = model.name + '.{}.{}.{}enhs.{}feats.{}.{}.{}.{}kb2TSS.{}'.format(
                    cell,
                    expr_type,
                    max_n_enh,
                    args.n_enh_feats,
                    use_rna_feats_flag,
                    use_prm_signal_flag,
                    rm_prm_signal_flag,
                    str(int(dist_thr / 1000)),
                    motif_tag,
                )

                model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                total_params = sum(np.prod(p.size()) for p in model_parameters)
                print(cell, 'fold', fi, 'total', total_params / 1_000_000, 'M params')
                print(model.name)
                saved_model_path = f'./results/{args.model_type}/'
                train(
                    model,
                    train_ds,
                    valid_dataset=valid_ds,
                    learning_rate=lr,
                    EPOCHS=args.epochs,
                    model_name=model.name,
                    fold_i=fi,
                    batch_size=batch_size,
                    device=device,
                    saved_model_path=saved_model_path,
                    early_stop_patience=args.early_stop_patience,
                )
                test_df = test(
                    model,
                    test_ds,
                    model_name=model.name,
                    saved_model_path=saved_model_path,
                    fold_i=fi,
                    batch_size=batch_size,
                    device=device,
                )
                test_df['cell'] = cell
                test_df['fold'] = fi
                test_df['use_rna_feats'] = use_rna_feats
                test_df['use_prm_signal'] = use_prm_signal_flag
                test_df['rm_prm_seq'] = rm_prm_signal_flag
                test_df['n_enh_feats'] = args.n_enh_feats
                test_df['motif_feat_dim'] = ds.motif_feat_dim
                results.append(test_df)

    results_df = pd.concat(results)
    result_path = f'./results/{args.model_type}/'
    os.makedirs(result_path, exist_ok=True)
    results_df.to_csv(f'{result_path}{model.name}_results.csv', index=False)


if __name__ == '__main__':
    main()
