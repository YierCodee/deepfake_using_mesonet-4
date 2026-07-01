# 🕵️ Deteksi Deepfake dengan MesoNet

Aplikasi **Streamlit** untuk mendeteksi gambar wajah deepfake menggunakan model **MesoNet-4** yang telah dilatih pada dataset *"140k Real and Fake Faces"* dari Kaggle (akurasi ~92%).

Model mengklasifikasikan gambar wajah sebagai **REAL** atau **FAKE**, dilengkapi visualisasi **Grad-CAM** yang menunjukkan area gambar yang paling memengaruhi keputusan model, serta halaman evaluasi komprehensif dengan metrik klasifikasi dan kurva ROC.

## Fitur

| Halaman | Deskripsi |
|---|---|
| 🔍 **Inference** | Upload gambar (single / batch, maks. 100) → prediksi Real/Fake + confidence score + heatmap Grad-CAM |
| 📊 **Evaluasi In-Dataset** | Upload ZIP test set terstruktur (`real/` & `fake/`), hitung akurasi, precision, recall, F1, AUC-ROC, confusion matrix, ROC curve |
| 📈 **Evaluasi Out-of-Dataset** | Upload gambar eksternal, parse label dari prefix nama file (`real_`/`fake_`), metrik + galeri misclassified + perbandingan side-by-side dengan in-dataset |
| 📉 **Analisis** | Histogram distribusi confidence score, metrik overconfidence, export semua hasil ke CSV |

## Teknologi

| Teknologi | Versi (terdeteksi) | Kegunaan |
|---|---|---|
| Python | 3.13 | Runtime |
| TensorFlow / Keras | 2.21 | Model MesoNet, inference, Grad-CAM |
| Streamlit | 1.58 | UI web interaktif |
| scikit-learn | 1.9 | Metrik evaluasi (akurasi, AUC, dll.) |
| Matplotlib | 3.11 | Plot confusion matrix, ROC curve, histogram |
| Pillow | – | Load & preprocessing gambar |
| NumPy | – | Operasi array |

## Struktur Folder

```
├── app.py                         # Landing page utama (multi-page entry point)
├── debug_model.py                 # Script testing model langsung (tanpa UI)
├── requirements.txt               # Dependensi Python
├── .gitignore                     # File/folder yang diabaikan git
├── AGENTS.md                      # Instruksi untuk OpenCode / Claude Code
│
├── pages/                         # Halaman Streamlit (auto-discovered)
│   ├── 1_Inference.py             # Halaman inference + Grad-CAM
│   ├── 2_Evaluasi_InDataset.py    # Evaluasi test set terstruktur
│   ├── 3_Evaluasi_OutDataset.py   # Evaluasi data eksternal
│   └── 4_Analisis.py              # Histogram & export CSV
│
├── src/                           # Modul Python inti
│   ├── config.py                  # Konstanta & flag konfigurasi
│   ├── preprocessing.py           # Load & preprocessing gambar
│   ├── model_utils.py             # Load model, predict, Grad-CAM
│   ├── metrics.py                 # Metrik evaluasi klasifikasi
│   └── visualization.py           # Plotting (Grad-CAM, CM, ROC, histogram)
│
├── model/                         # Model pre-trained
│   └── mesonet_deepfake.h5        # MesoNet-4 (Sequential, 4 Conv2D)
│
├── fake1.jpg                      # Sample gambar FAKE
├── fake2.jpg
├── fake3.jpg
├── real1.jpg                      # Sample gambar REAL
├── real2.jpg
└── real3.jpg
```

## Instalasi

### 1. Clone / masuk ke direktori project

```bash
cd direktori-project
```

### 2. Aktifkan virtual environment

```bash
source env313/bin/activate
```

> Virtual environment **env313/** sudah tersedia dengan Python 3.13 dan semua dependensi terinstal.

### 3. (Opsional) Install dependensi dari `requirements.txt`

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`. Gunakan sidebar untuk navigasi antar halaman.

### Testing Model Langsung (tanpa UI)

```bash
python debug_model.py
```

Script ini memuat model dan menjalankan prediksi pada `real2.jpg`, menampilkan output mentah model.

## Spesifikasi Model

| Parameter | Nilai |
|---|---|
| Arsitektur | MesoNet-4 (4 layer Conv2D) |
| Input | 256×256 RGB |
| Normalisasi | [0, 1] (÷255.0) |
| Threshold | ≥ 0.5 → **FAKE**, < 0.5 → **REAL** |
| Label training asli | REAL = 1, FAKE = 0 (terbalik) |
| Koreksi | `INVERT_PROBABILITY = True` di `src/config.py` |
| Last conv layer | `conv2d_3` (auto-detected untuk Grad-CAM) |
| Batch maksimal | 100 gambar (mencegah OOM di RAM 8 GB) |

## Preprocessing Pipeline

```
Input image → Convert RGB → [Face detection + crop — TODO] 
→ Resize 256×256 → [Mean subtraction — TODO] 
→ Normalize /255.0 → Expand dims → Model input (1, 256, 256, 3)
```

Fitur preprocessing lanjutan (face detection, mean subtraction, TTA) tersedia sebagai placeholder di `src/preprocessing.py` dan dapat diaktifkan melalui flag di `src/config.py`.

## Catatan Penting

- **CPU only**: Mesin pengembangan tidak memiliki GPU. Semua inference berjalan di CPU.
- **RAM 8 GB**: Batch processing dibatasi 100 gambar untuk menghindari out-of-memory.
- **Label terbalik**: Model dilatih dengan label REAL=1, FAKE=0. Output di-invert otomatis oleh `INVERT_PROBABILITY = True`.
- **Model tidak di-ignore git**: File `.h5` sengaja tidak masuk `.gitignore`.
- **Tidak ada test set** bawaan — gunakan halaman Evaluasi untuk upload ZIP test set sendiri.
