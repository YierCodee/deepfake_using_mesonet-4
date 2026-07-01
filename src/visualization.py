import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import auc, roc_curve


def plot_gradcam_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Overlay a Grad-CAM heatmap (2D array) on a PIL image.

    Returns an RGB numpy array in [0, 1].
    """
    heatmap_resized = np.array(
        Image.fromarray(heatmap).resize(image.size, Image.LANCZOS)
    )
    colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    base = np.array(image.convert("RGB")) / 255.0
    overlay = colored * alpha + base * (1 - alpha)
    return np.clip(overlay, 0, 1)


def plot_confusion_matrix(cm, class_names=("Real", "Fake")):
    """Render a confusion matrix as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba):
    """Render an ROC curve as a matplotlib figure."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confidence_histogram(real_probas, fake_probas):
    """Plot histogram of confidence scores split by ground-truth class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    if len(real_probas):
        ax.hist(real_probas, bins=bins, alpha=0.7, label="REAL", color="green")
    if len(fake_probas):
        ax.hist(fake_probas, bins=bins, alpha=0.7, label="FAKE", color="red")
    ax.axvline(0.5, color="gray", linestyle="--", label="Threshold")
    ax.set_xlabel("FAKE probability (confidence score)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    """Convert a matplotlib figure to PNG bytes (for download / display)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
