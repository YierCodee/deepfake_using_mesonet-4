import csv
import io

import streamlit as st

from src.config import THRESHOLD
from src.visualization import (
    plot_confidence_histogram,
)

st.set_page_config(page_title="Analisis", page_icon="📉", layout="wide")

st.title("📉 Analisis & Export Hasil")

# --- Confidence Histogram ---
st.subheader("Distribusi Confidence Score")
st.markdown(
    "Histogram di bawah menunjukkan sebaran confidence score (probabilitas FAKE) "
    "berdasarkan ground-truth label. Jika tersedia, data dari evaluasi In-Dataset "
    "dan Out-of-Dataset akan ditampilkan."
)

in_metrics = st.session_state.get("in_metrics")
out_metrics = st.session_state.get("out_metrics")

real_probas = []
fake_probas = []

for metrics, label in [(in_metrics, "In-Dataset"), (out_metrics, "Out-of-Dataset")]:
    if metrics is not None:
        y_true = metrics["y_true"]
        y_proba = metrics["y_pred_proba"]
        for t, p in zip(y_true, y_proba):
            if t == 0:
                real_probas.append(p)
            else:
                fake_probas.append(p)

if not real_probas and not fake_probas:
    st.info(
        "Belum ada data evaluasi. Jalankan evaluasi di halaman "
        "**Evaluasi In-Dataset** atau **Evaluasi Out-of-Dataset** terlebih dahulu."
    )
else:
    fig = plot_confidence_histogram(real_probas, fake_probas)
    st.pyplot(fig)

    total_real = len(real_probas)
    total_fake = len(fake_probas)
    overconfident_real = sum(1 for p in real_probas if p >= THRESHOLD)
    overconfident_fake = sum(1 for p in fake_probas if p < THRESHOLD)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Real gambar terprediksi Fake",
            f"{overconfident_real}/{total_real} ({overconfident_real / total_real:.1%})" if total_real else "N/A",
        )
    with col2:
        st.metric(
            "Fake gambar terprediksi Real",
            f"{overconfident_fake}/{total_fake} ({overconfident_fake / total_fake:.1%})" if total_fake else "N/A",
        )

# --- Export CSV ---
st.subheader("Export Hasil ke CSV")

if not in_metrics and not out_metrics:
    st.info("Tidak ada hasil evaluasi untuk diexport.")
    st.stop()

csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow([
    "Dataset", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC",
    "Total", "Real", "Fake", "Misclassified",
])

for name, metrics in [("In-Dataset", in_metrics), ("Out-of-Dataset", out_metrics)]:
    if metrics is None:
        continue
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    misclassified_count = int(sum(y_true != y_pred))
    writer.writerow([
        name,
        f"{metrics['accuracy']:.4f}",
        f"{metrics['precision']:.4f}",
        f"{metrics['recall']:.4f}",
        f"{metrics['f1']:.4f}",
        f"{metrics['auc_roc']:.4f}",
        len(y_true),
        int(sum(y_true == 0)),
        int(sum(y_true == 1)),
        misclassified_count,
    ])

# Add detailed per-sample results
writer.writerow([])
writer.writerow(["Dataset", "Filename", "True Label", "Predicted Label", "Confidence"])

for name, metrics in [("In-Dataset", in_metrics), ("Out-of-Dataset", out_metrics)]:
    if metrics is None:
        continue
    # If we don't have per-sample filenames, use indices
    for i, (t, p, proba) in enumerate(
        zip(metrics["y_true"], metrics["y_pred"], metrics["y_pred_proba"])
    ):
        writer.writerow([
            name, f"sample_{i:04d}.jpg",
            "Fake" if t else "Real",
            "Fake" if p else "Real",
            f"{proba:.4f}",
        ])

st.download_button(
    label="📥 Download CSV",
    data=csv_buffer.getvalue(),
    file_name="evaluasi_mesonet.csv",
    mime="text/csv",
    type="primary",
)
