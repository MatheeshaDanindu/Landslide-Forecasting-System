from __future__ import annotations

import numpy as np


def _apply_valid(y_true, y_pred, valid_mask):
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).ravel()
        y_true, y_pred = y_true[valid], y_pred[valid]
    return y_true, y_pred


def confusion_matrix_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5,
                                  valid_mask: np.ndarray | None = None):
    """valid_mask: optional bool array, True where the pixel is a real (non-cloud/nodata)
    observation. Excluded pixels never enter the counts, matching the masked-loss principle."""
    y_true, y_pred = _apply_valid(y_true, y_pred, valid_mask)
    true_mask = y_true > 0.5
    pred_mask = y_pred >= threshold

    tp = np.sum(true_mask & pred_mask)
    fp = np.sum((~true_mask) & pred_mask)
    tn = np.sum((~true_mask) & (~pred_mask))
    fn = np.sum(true_mask & (~pred_mask))
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def average_precision(y_true: np.ndarray, y_pred: np.ndarray, valid_mask: np.ndarray | None = None):
    """PR-AUC (average precision) over continuous probabilities. Returns NaN for a
    single-class fold (no positives or no negatives), since AP is undefined there."""
    from sklearn.metrics import average_precision_score

    y_true, y_pred = _apply_valid(y_true, y_pred, valid_mask)
    true_mask = y_true > 0.5
    if true_mask.all() or not true_mask.any():
        return float("nan")
    return float(average_precision_score(true_mask, y_pred))


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray, valid_mask: np.ndarray | None = None):
    """ROC-AUC over continuous probabilities. NaN for a single-class fold (same reason as above)."""
    from sklearn.metrics import roc_auc_score

    y_true, y_pred = _apply_valid(y_true, y_pred, valid_mask)
    true_mask = y_true > 0.5
    if true_mask.all() or not true_mask.any():
        return float("nan")
    return float(roc_auc_score(true_mask, y_pred))


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5,
                          valid_mask: np.ndarray | None = None):
    counts = confusion_matrix_from_arrays(y_true, y_pred, threshold, valid_mask)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    pr_auc_val = average_precision(y_true, y_pred, valid_mask)
    roc_auc_val = roc_auc(y_true, y_pred, valid_mask)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
