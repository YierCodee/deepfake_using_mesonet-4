import streamlit as st
from PIL import Image

from src.config import MAX_BATCH, THRESHOLD
from src.model_utils import load_model, predict, compute_gradcam
from src.preprocessing import preprocess_image
from src.visualization import plot_gradcam_overlay

st.set_page_config(page_title="Inference", page_icon="🔍", layout="wide")

model = load_model()
if model is None:
    st.stop()

st.title("🔍 Inference — Deteksi Deepfake")
st.markdown("Upload satu atau beberapa gambar wajah untuk dianalisis.")

uploaded_files = st.file_uploader(
    "Pilih gambar (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Silakan upload gambar untuk memulai.")
    st.stop()

if len(uploaded_files) > MAX_BATCH:
    st.warning(f"Maksimal {MAX_BATCH} gambar per batch. {MAX_BATCH} pertama diproses.")
    uploaded_files = uploaded_files[:MAX_BATCH]

st.subheader("Preview Gambar")
cols = st.columns(4)
for i, f in enumerate(uploaded_files):
    with cols[i % 4]:
        st.image(f, caption=f.name, use_container_width=True)

if "inf_results" not in st.session_state:
    st.session_state.inf_results = None

if st.button("🔍 Analisis Gambar", type="primary"):
    progress = st.progress(0, text="Memproses...")
    results = []

    for i, f in enumerate(uploaded_files):
        progress.progress(
            (i + 1) / len(uploaded_files),
            text=f"Memproses {f.name} ({i + 1}/{len(uploaded_files)})...",
        )
        try:
            f.seek(0)
            img = Image.open(f).copy()
            processed = preprocess_image(img)
            proba = predict(model, processed)

            results.append({
                "name": f.name,
                "img": img,
                "array": processed,
                "proba": proba,
                "label": "FAKE" if proba >= THRESHOLD else "REAL",
            })
        except Exception as e:
            st.warning(f"Gagal memproses {f.name}: {e}")

    progress.empty()
    st.session_state.inf_results = results

if st.session_state.inf_results is None:
    st.stop()

results = st.session_state.inf_results
st.subheader("Hasil Analisis")

for idx, r in enumerate(results):
    if idx % 4 == 0:
        cols = st.columns(4)
    with cols[idx % 4]:
        st.image(r["img"], use_container_width=True)
        confidence = r["proba"] if r["label"] == "FAKE" else 1.0 - r["proba"]
        if r["label"] == "FAKE":
            st.error(f"🚨 **FAKE**")
        else:
            st.success(f"✅ **REAL**")
        st.caption(f"Confidence: {confidence:.2%}")

        with st.expander(f"🔥 Grad-CAM — {r['name']}"):
            heatmap = compute_gradcam(model, r["array"])
            if heatmap is not None:
                overlay = plot_gradcam_overlay(r["img"], heatmap)
                st.image(overlay, use_container_width=True)
                st.caption("Area merah menunjukkan region yang paling memengaruhi keputusan model.")
            else:
                st.warning("Tidak dapat menghasilkan Grad-CAM (tidak ditemukan layer konvolusi).")
