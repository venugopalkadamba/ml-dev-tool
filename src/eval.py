import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    log_loss as sk_log_loss,
    confusion_matrix as sk_confusion_matrix,
    classification_report as sk_classification_report,
)


# Core scalar metrics
def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_precision_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_recall_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(sk_log_loss(y_true, y_proba))


# AUC metrics
def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_proba))


def compute_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    # Average precision is a standard PR AUC summary
    return float(average_precision_score(y_true, y_proba))


# Curves for plotting
def compute_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    return fpr, tpr, thresholds, float(auc_val)


def compute_pr_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    return precision, recall, thresholds, float(ap)


# Reports and matrices
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred)


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return sk_classification_report(y_true, y_pred, output_dict=True, zero_division=0)


# Fairness metrics
def compute_demographic_parity_ratio(y_pred: np.ndarray, protected_attribute: pd.Series) -> float:
    """
    Demographic Parity Ratio (a.k.a. Disparity):
    P(ŷ=1 | A=1) / P(ŷ=1 | A=0)
    Returns inf if the denominator is 0.
    """
    # Normalize boolean-like or string categories to two groups
    series = protected_attribute.copy()
    # If boolean, map to 0/1
    if series.dtype == bool:
        series = series.astype(int)
    # If object/categorical with two unique values, map first to 0, second to 1 consistently
    uniques = pd.Series(series.unique()).dropna().tolist()
    if len(uniques) == 2 and not set(uniques) <= {0, 1}:
        mapping = {uniques[0]: 0, uniques[1]: 1}
        series = series.map(mapping)

    mask0 = series == 0
    mask1 = series == 1
    # Ensure index alignment for pandas Series
    if hasattr(y_pred, 'index'):
        yp = y_pred
    else:
        # y_pred is numpy array; align by boolean array lengths
        yp = y_pred
    p_y1_g0 = float((yp[mask0] == 1).mean()) if mask0.any() else 0.0
    p_y1_g1 = float((yp[mask1] == 1).mean()) if mask1.any() else 0.0
    if p_y1_g0 == 0:
        return float("inf")
    return p_y1_g1 / p_y1_g0


