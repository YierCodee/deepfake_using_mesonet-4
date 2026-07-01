# Evaluasi Model Deepfake — Panduan Fitur

Dokumen ini menjelaskan dua halaman evaluasi dalam aplikasi Deteksi Deepfake: **Evaluasi In-Dataset (Test Set)** dan **Evaluasi Out-of-Dataset (Data Eksternal)**. Kedua halaman mengukur performa model MesoNet-4 pada dataset uji, tetapi memiliki tujuan dan pendekatan yang berbeda.

---

## 1. Evaluasi In-Dataset (Test Set)

**Tujuan:** Mengukur performa model pada dataset terstruktur yang labelnya sudah diketahui. Dataset ini sebaiknya representatif terhadap distribusi data training (in-distribution), sehingga hasilnya mencerminkan akurasi dasar model.

### Cara Upload

Pengguna dapat memilih salah satu dari dua metode upload melalui radio button:

#### Metode A: Upload ZIP (folder real/ & fake/)

Upload file **ZIP** yang berisi folder `real/` dan `fake/`. Label gambar ditentukan secara otomatis dari struktur folder.

```
test_set.zip
├── real/
│   ├── img001.jpg
│   └── ...
└── fake/
│   ├── img001.jpg
    └── ...
```

- Gambar di dalam folder `real/` diberi label **REAL** (0)
- Gambar di dalam folder `fake/` diberi label **FAKE** (1)
- Gambar di luar kedua folder akan dilewati

#### Metode B: Upload Gambar Langsung

Upload file gambar satu per satu (JPG/PNG) tanpa perlu ZIP. Dua area upload disediakan:

| Kolom | Label | Label Numerik |
|---|---|---|
| Kiri | **Pilih gambar REAL** | 0 |
| Kanan | **Pilih gambar FAKE** | 1 |

Pengguna tinggal menyeret gambar ke kolom yang sesuai. Nama file tidak diperiksa — label murni ditentukan oleh kolom upload yang dipilih.

### Proses

1. **Ekstraksi/loading** — gambar dibaca dari ZIP atau dari upload langsung
2. **Preprocessing** — setiap gambar di-resize ke 256×256, dikonversi ke RGB, dinormalisasi ÷255.0
3. **Prediksi batch** — semua gambar diproses sekaligus dengan `predict_batch()` untuk efisiensi
4. **Koreksi label** — output model di-invert (`INVERT_PROBABILITY = True`) karena model dilatih dengan label terbalik (REAL=1, FAKE=0)
5. **Klasifikasi** — probability ≥ 0.5 → FAKE, < 0.5 → REAL

### Output

#### Metrik Utama (5 kolom)

| Metrik | Rentang | Interpretasi |
|---|---|---|
| **Accuracy** | 0–100% | Proporsi prediksi benar dari total gambar |
| **Precision** | 0–100% | Dari semua yang diprediksi FAKE, berapa yang benar-benar FAKE |
| **Recall** | 0–100% | Dari semua FAKE sebenarnya, berapa yang berhasil terdeteksi |
| **F1-Score** | 0–100% | Harmonic mean precision & recall |
| **AUC-ROC** | 0–1 | Area Under Curve ROC; semakin mendekati 1 semakin baik |

#### Visualisasi

- **Confusion Matrix** — tabel 2×2 (Real vs Fake) yang menunjukkan True Positive, True Negative, False Positive, False Negative
- **ROC Curve** — kurva Trade-off antara True Positive Rate dan False Positive Rate, lengkap dengan nilai AUC

#### Classification Report

Tabel per kelas yang menampilkan Precision, Recall, F1-Score, dan Support (jumlah sampel) untuk kelas Real dan Fake.

---

## 2. Evaluasi Out-of-Dataset (Data Eksternal)

**Tujuan:** Mengukur performa model pada gambar-gambar di luar distribusi data training. Ini penting untuk menguji seberapa baik model **menggeneralisasi** ke data baru yang belum pernah dilihat sebelumnya.

### Cara Upload

Sama seperti In-Dataset, dua metode upload tersedia:

#### Metode A: Upload ZIP (folder real/ & fake/)

Struktur ZIP identik dengan In-Dataset:

```
dataset.zip
├── real/
│   ├── img001.jpg
│   └── ...
└── fake/
    ├── img001.jpg
    └── ...
```

Bedanya: untuk ZIP mode, prediksi dilakukan secara **batch** (`predict_batch`) sama seperti In-Dataset, sehingga lebih cepat untuk dataset besar.

#### Metode B: Upload Gambar Langsung

Dua area upload: REAL dan FAKE. **Tidak ada batasan nama file** — pengguna bebas memberi nama file apa pun. Label ditentukan semata-mata oleh kolom upload yang dipilih.

### Proses

- Sama seperti In-Dataset: loading → preprocessing → prediksi → kalkulasi metrik
- Untuk mode Gambar Langsung: prediksi dilakukan **satu per satu** dengan progress bar per gambar
- Untuk mode ZIP: prediksi batch seperti di In-Dataset

### Output

#### Metrik Utama

Sama seperti In-Dataset: Accuracy, Precision, Recall, F1-Score, AUC-ROC — ditampilkan dalam 5 kolom.

#### Visualisasi

- Confusion Matrix
- ROC Curve

#### Galeri Misclassified

Fitur **hanya ada di halaman Out-of-Dataset**. Menampilkan semua gambar yang salah diklasifikasikan dalam grid 4 kolom:

- Gambar asli
- Label sebenarnya (True) vs label prediksi (Pred)
- Confidence score prediksi

Jika tidak ada gambar yang salah klasifikasi, akan muncul pesan sukses.

#### Perbandingan In-Dataset vs Out-of-Dataset

Fitur **unik** yang muncul hanya jika pengguna sudah menjalankan Evaluasi In-Dataset sebelumnya dalam sesi yang sama (data disimpan di `st.session_state.in_metrics`).

Menampilkan side-by-side:

| Metrik | Sumber |
|---|---|
| Akurasi In-Dataset | Dari sesi In-Dataset sebelumnya |
| Akurasi Out-of-Dataset | Dari evaluasi saat ini |
| Delta Akurasi | Selisih (Out − In), bisa positif atau negatif |
| AUC In-Dataset | Dari sesi In-Dataset sebelumnya |
| Delta AUC | Selisih AUC (Out − In) |

Delta yang besar (terutama negatif) mengindikasikan bahwa model tidak bisa menggeneralisasi dengan baik ke data baru (overfitting).

---

## Perbandingan Langsung

| Aspek | In-Dataset | Out-of-Dataset |
|---|---|---|
| **Tujuan** | Mengukur akurasi dasar pada data representatif | Menguji generalisasi pada data baru |
| **Metode Upload** | ZIP (folder real/fake) atau Gambar Langsung | ZIP (folder real/fake) atau Gambar Langsung |
| **Prediksi** | Batch (`predict_batch`) | ZIP: batch, Langsung: per gambar |
| **Classification Report** | ✅ Ya | ❌ Tidak |
| **Galeri Misclassified** | ❌ Tidak | ✅ Ya |
| **Perbandingan In vs Out** | ❌ Tidak | ✅ Ya (jika In-Dataset sudah dijalankan) |
| **Batasan** | Maks. 100 gambar per sesi | Maks. 100 gambar per sesi |
| **Label** | Otomatis dari folder/kolom upload | Otomatis dari folder/kolom upload |

---

## Catatan Teknis

- **Label convention**: REAL = 0, FAKE = 1 (setelah koreksi `INVERT_PROBABILITY`)
- **Threshold**: ≥ 0.5 → FAKE
- **Input preprocessing**: RGB → resize 256×256 → ÷255.0
- **Max batch**: 100 gambar (pencegahan OOM di RAM 8 GB)
- **Semua metrik** dihitung oleh `src.metrics.compute_metrics()` menggunakan scikit-learn
