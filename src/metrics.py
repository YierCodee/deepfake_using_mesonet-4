import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .config import THRESHOLD

def compute_metrics(y_true, y_pred_proba):
    """Compute all classification metrics for binary (Real=0, Fake=1) labels."""
    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_proba) >= THRESHOLD).astype(int)

    # 1. Hitung AUC-ROC secara aman (butuh minimal 2 kelas unik)
    if len(np.unique(y_true)) < 2:
        auc_roc = 0.5  # Nilai default jika data hanya berisi 1 kelas
    else:
        auc_roc = roc_auc_score(y_true, y_pred_proba)

    # 2. Paksa confusion matrix selalu berukuran 2x2 menggunakan labels=[0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 3. Paksa classification report membaca kedua kelas menggunakan labels=[0, 1]
    report = classification_report(
        y_true, 
        y_pred, 
        labels=[0, 1], 
        target_names=["Real", "Fake"], 
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": np.array(y_pred_proba),
    }
