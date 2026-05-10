import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


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


def resolve_fold_ids(split_df: pd.DataFrame, fold_arg: str) -> list[str]:
    available_fold_cols = [c for c in split_df.columns if c.startswith('fold_')]
    if len(available_fold_cols) == 0:
        raise ValueError('No fold columns found in split file. Expected columns like fold_1 / fold_enformer.')
    if fold_arg == 'all':
        return [c.replace('fold_', '') for c in available_fold_cols]

    requested_col = f'fold_{fold_arg}'
    if requested_col in split_df.columns:
        return [fold_arg]

    if fold_arg == 'borzoi':
        fallback = '1' if 'fold_1' in split_df.columns else available_fold_cols[0].replace('fold_', '')
        print(f'Warning: fold_borzoi not found. Falling back to fold_{fallback}.')
        return [fallback]

    raise KeyError(f'{requested_col} not found. Available: {available_fold_cols}')


def compute_binary_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    logits = np.asarray(logits).astype(float)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    out = {
        'acc': float(accuracy_score(y_true, preds)),
        'auroc': float('nan'),
        'auprc': float('nan'),
        'pos_rate': float(y_true.mean()) if len(y_true) > 0 else float('nan'),
    }
    # Metrics requiring both classes present.
    if len(np.unique(y_true)) > 1:
        out['auroc'] = float(roc_auc_score(y_true, probs))
        out['auprc'] = float(average_precision_score(y_true, probs))
    return out


@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    mode: str = 'max'
    delta: float = 0.0
    path: str = 'checkpoint.pt'
    verbose: bool = True


class EarlyStoppingMetric:
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.cfg.mode == 'max':
            return score > self.best_score + self.cfg.delta
        return score < self.best_score - self.cfg.delta

    def step(self, score: float, model: torch.nn.Module, epoch_i: int):
        if self._is_better(score):
            old = self.best_score
            self.best_score = score
            self.counter = 0
            self._save(score, old, model, epoch_i)
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.cfg.patience}', 'best_score', self.best_score)
            if self.counter >= self.cfg.patience:
                self.early_stop = True

    def _save(self, score, old, model, epoch_i):
        if self.cfg.verbose:
            print(f'Validation metric improved ({old} -> {score}). Saving model ...')
        torch.save(
            {
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'metric': score,
            },
            self.cfg.path,
        )
        print('Saving ckpt at', self.cfg.path)
