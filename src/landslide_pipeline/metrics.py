from __future__ import annotations

import numpy as np


def confusion_matrix_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    true_mask = y_true > 0.5
    pred_mask = y_pred >= threshold

    tp = np.sum(true_mask & pred_mask)
    fp = np.sum((~true_mask) & pred_mask)
    tn = np.sum((~true_mask) & (~pred_mask))
    fn = np.sum(true_mask & (~pred_mask))
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    counts = confusion_matrix_from_arrays(y_true, y_pred, threshold)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
