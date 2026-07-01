import streamlit as st

st.set_page_config(
    page_title="Deteksi Deepfake — MesoNet", page_icon="🕵️", layout="wide"
)

st.title("🕵️ Deteksi Deepfake dengan MesoNet")
st.markdown("""
Aplikasi ini menggunakan model **MesoNet-4** (akurasi ~92% pada "140k Real and Fake Faces" dari Kaggle) untuk mendeteksi deepfake pada gambar wajah.

Gunakan sidebar untuk navigasi ke halaman-halaman berikut:

| Halaman | Fungsi |
|---|---|
| 🔍 **Inference** | Upload gambar → prediksi Real/Fake + confidence score + Grad-CAM heatmap |
| 📊 **Evaluasi In-Dataset** | Evaluasi pada test set terstruktur (folder `real/` & `fake/`) |
| 📈 **Evaluasi Out-of-Dataset** | Evaluasi pada data eksternal dengan label dari nama file |
| 📉 **Analisis** | Histogram confidence score & export hasil ke CSV |

### Spesifikasi Model
- **Input**: 256×256 RGB, normalisasi [0, 1]
- **Threshold**: ≥ 0.5 → **FAKE**, < 0.5 → **REAL**
- **Backend**: TensorFlow (CPU inference)
""")
