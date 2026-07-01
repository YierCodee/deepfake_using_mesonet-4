import os
import tempfile
import zipfile

import numpy as np
import streamlit as st
from PIL import Image

from src.config import MAX_BATCH, THRESHOLD
from src.metrics import compute_metrics
from src.model_utils import load_model, predict, predict_batch
from src.preprocessing import preprocess_image
from src.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
)

st.set_page_config(page_title="Evaluasi Out-of-Dataset", page_icon="📈", layout="wide")

model = load_model()
if model is None:
    st.stop()

st.title("📈 Evaluasi Out-of-Dataset (Data Eksternal)")

mode = st.radio(
    "Metode Upload",
    ["📁 Upload ZIP (folder real/ & fake/)", "🖼️ Upload Gambar Langsung"],
    horizontal=True,
)

images = []
labels = []
filenames = []
misclassified = []
ready = False

if mode == "📁 Upload ZIP (folder real/ & fake/)":
    st.markdown("""
    Upload file **ZIP** yang berisi folder `real/` dan `fake/` dengan gambar wajah.

    Struktur ZIP yang diharapkan:
    ```
    dataset.zip
    ├── real/
    │   ├── img001.jpg
    │   └── ...
    └── fake/
        ├── img001.jpg
        └── ...
    ```
    """)

    uploaded_zip = st.file_uploader(
        "Upload ZIP dataset", type=["zip"], key="out_zip"
    )
    ready = uploaded_zip is not None

else:
    col_real, col_fake = st.columns(2)
    with col_real:
        real_files = st.file_uploader(
            "Pilih gambar REAL",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="out_real",
        )
    with col_fake:
        fake_files = st.file_uploader(
            "Pilih gambar FAKE",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="out_fake",
        )
    ready = bool(real_files) or bool(fake_files)

if not ready:
    st.info("Upload data untuk memulai evaluasi.")
    st.stop()

if st.button("🚀 Jalankan Evaluasi", type="primary"):
    if mode == "📁 Upload ZIP (folder real/ & fake/)":
        progress = st.progress(0, text="Membaca ZIP...")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(uploaded_zip) as z:
                members = [m for m in z.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
                total = len(members)
                if total == 0:
                    st.error("Tidak ditemukan gambar dalam ZIP.")
                    st.stop()

                for i, member in enumerate(members):
                    progress.progress(
                        (i + 1) / total,
                        text=f"Membaca {member} ({i + 1}/{total})...",
                    )
                    z.extract(member, tmpdir)
                    path = os.path.join(tmpdir, member)

                    parts = os.path.normpath(member).split(os.sep)
                    if "fake" in parts:
                        label = 1
                    elif "real" in parts:
                        label = 0
                    else:
                        continue

                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        labels.append(label)
                        filenames.append(os.path.basename(member))
                    except Exception as e:
                        st.warning(f"Gagal membaca {member}: {e}")

    else:
        all_uploads = []
        if real_files:
            all_uploads.extend((f, 0) for f in real_files)
        if fake_files:
            all_uploads.extend((f, 1) for f in fake_files)

        if len(all_uploads) > MAX_BATCH:
            st.warning(f"Maksimal {MAX_BATCH} gambar. {MAX_BATCH} pertama diproses.")
            all_uploads = all_uploads[:MAX_BATCH]

        progress = st.progress(0, text="Memproses...")
        for i, (f, label) in enumerate(all_uploads):
            progress.progress(
                (i + 1) / len(all_uploads),
                text=f"Memproses {f.name} ({i + 1}/{len(all_uploads)})...",
            )
            try:
                f.seek(0)
                img = Image.open(f).copy().convert("RGB")
                images.append(img)
                labels.append(label)
                filenames.append(f.name)
            except Exception as e:
                st.warning(f"Gagal memproses {f.name}: {e}")

    if len(images) == 0:
        st.error("Tidak ada gambar yang berhasil diproses.")
        st.stop()

    progress.progress(0, text="Menjalankan prediksi...")

    if mode == "📁 Upload ZIP (folder real/ & fake/)":
        batch_arrays = [preprocess_image(img) for img in images]
        pred_probas = list(predict_batch(model, np.vstack(batch_arrays)))
    else:
        pred_probas = []
        for i, img in enumerate(images):
            progress.progress(
                (i + 1) / len(images),
                text=f"Prediksi {filenames[i]} ({i + 1}/{len(images)})...",
            )
            pred_probas.append(predict(model, preprocess_image(img)))

    progress.empty()

    for i in range(len(images)):
        pred_label = 1 if pred_probas[i] >= THRESHOLD else 0
        if labels[i] != pred_label:
            misclassified.append({
                "name": filenames[i],
                "img": images[i],
                "true": labels[i],
                "pred": pred_label,
                "proba": pred_probas[i],
            })

    metrics = compute_metrics(labels, pred_probas)
    st.session_state.out_metrics = metrics

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1']:.2%}")
    col5.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")

    if "in_metrics" in st.session_state:
        st.subheader("Perbandingan In-Dataset vs Out-of-Dataset")
        in_m = st.session_state.in_metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Akurasi In-Dataset", f"{in_m['accuracy']:.2%}",
            delta=f"{metrics['accuracy'] - in_m['accuracy']:.2%}",
            delta_color="off",
        )
        col2.metric(
            "Akurasi Out-of-Dataset", f"{metrics['accuracy']:.2%}"
        )
        col3.metric(
            "AUC In-Dataset", f"{in_m['auc_roc']:.3f}",
            delta=f"{metrics['auc_roc'] - in_m['auc_roc']:.3f}",
            delta_color="off",
        )

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]))
    with col_right:
        st.subheader("ROC Curve")
        st.pyplot(plot_roc_curve(metrics["y_true"], metrics["y_pred_proba"]))

    st.subheader("Galeri Misclassified")
    if misclassified:
        for idx, m in enumerate(misclassified):
            if idx % 4 == 0:
                cols = st.columns(4)
            with cols[idx % 4]:
                st.image(m["img"], use_container_width=True)
                st.error(f"True: {'FAKE' if m['true'] else 'REAL'} → Pred: {'FAKE' if m['pred'] else 'REAL'}")
                conf = m["proba"] if m["pred"] == 1 else 1.0 - m["proba"]
                st.caption(f"Confidence: {conf:.2%}")
    else:
        st.success("🎉 Tidak ada gambar yang salah klasifikasi!")

    st.subheader("Rangkuman")
    st.info(
        f"Dataset: **{len(labels)} gambar** "
        f"({sum(1 for l in labels if l == 0)} Real, {sum(1 for l in labels if l == 1)} Fake)  |  "
        f"Akurasi: **{metrics['accuracy']:.2%}**  |  "
        f"AUC-ROC: **{metrics['auc_roc']:.3f}**  |  "
        f"Misclassified: **{len(misclassified)}**"
    )
