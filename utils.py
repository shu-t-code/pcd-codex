from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics using sklearn."""
    pred_bin = (y_pred > 0.5).astype(np.int32)
    metrics = {
        'accuracy': accuracy_score(y_true, pred_bin),
        'precision': precision_score(y_true, pred_bin, zero_division=0),
        'recall': recall_score(y_true, pred_bin, zero_division=0),
        'f1': f1_score(y_true, pred_bin, zero_division=0),
    }
    return metrics
