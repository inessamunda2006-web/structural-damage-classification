# Structural Damage Classification

Prediksi tingkat kerusakan bangunan (`damage_grade`: Grade 1 sampai Grade 5) setelah gempa, berdasarkan lokasi dan karakteristik struktur bangunan. Tugas 1 Supervised Machine Learning 2026 (Kaggle community competition).

## Struktur Repository

```
structural-damage-classification/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── .gitkeep
│   ├── train.csv              (tidak di-commit)
│   ├── test.csv               (tidak di-commit)
│   ├── sample_submission.csv  (tidak di-commit)
│   └── metadata.csv           (tidak di-commit)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
│
├── scripts/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── tune.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── pipeline_model_lgbm.pkl
│   ├── best_params.json
│   └── cv_scores.csv
│
├── submissions/
│   └── submission.csv
│
└── reports/
    └── figures/
        ├── confusion_matrix.png
        └── feature_importance.png
```

## Cara Menggunakan Repository

1. Clone repo dan install dependency:
   ```bash
   git clone https://github.com/inessamunda2006-web/structural-damage-classification.git
   cd structural-damage-classification
   pip install -r requirements.txt
   ```
2. Unduh data dari halaman Data kompetisi Kaggle (tombol **Download All**) atau lewat CLI, lalu taruh `train.csv`, `test.csv`, `sample_submission.csv`, `metadata.csv` di folder `data/`:
   ```bash
   kaggle competitions download -c structural-damage-classification
   ```
   Pastikan sudah klik "I Understand and Accept" di halaman Rules.
3. (Opsional) tuning hyperparameter, hasilnya disimpan ke `models/best_params.json`:
   ```bash
   python -m scripts.tune
   ```
4. Training, evaluasi, dan simpan model (tambahkan `--cv` untuk Stratified 5-fold CV):
   ```bash
   python -m scripts.train --cv
   ```
5. Buat file submission:
   ```bash
   python -m scripts.predict
   ```
   Output: `submissions/submission.csv`. Upload ke Kaggle atau lewat CLI:
   ```bash
   kaggle competitions submit -c structural-damage-classification -f submissions/submission.csv -m "LightGBM pipeline"
   ```

Notebook di `notebooks/` dijalankan dari foldernya sendiri dan memakai modul dari `scripts/`.

## Workflow Analisis

1. **EDA** (`01_eda.ipynb`): distribusi target, missing value, anomali, pengaruh lokasi dan material.
2. **Feature engineering** (`BuildingFeatureAdder`): `age_building = 999` dianggap placeholder (jadi NaN + flag), rasio tinggi per lantai, luas per lantai, proxy volume, slenderness, log luas.
3. **Preprocessing** (`ColumnTransformer`): numerik di-impute median, kategorik `OrdinalEncoder`, kolom lokasi (district, vdcmun, ward) dipakai sebagai angka asli dan juga `TargetEncoder` multikelas (cross-fitting internal supaya tidak bocor).
4. **Model**: `Pipeline` = feature engineering, ColumnTransformer, `LGBMClassifier`. Tidak memakai Neural Network. Perbandingan awal dengan Logistic Regression (OvR dan OvO), Random Forest, HistGradientBoosting ada di `02_modeling.ipynb`.
5. **Tuning**: `RandomizedSearchCV` (10 kombinasi, 3-fold, 25% data), lalu jumlah pohon final dipilih dari kurva validasi holdout.
6. **Evaluasi**: holdout 80/20 dan Stratified 5-fold CV pada seluruh train.
7. **Prediksi**: model dilatih ulang di seluruh train, lalu `predict.py` menulis `submission.csv` sesuai format `sample_submission.csv`.

## Hasil

Stratified 5-fold CV pada seluruh data train (rata-rata, simpangan baku):

| Metrik | Rata-rata | Std |
|---|---|---|
| Accuracy (= micro-F1) | 0.5877 | 0.0013 |
| Macro-F1 | 0.5472 | 0.0012 |
| Quadratic Weighted Kappa | 0.7320 | 0.0022 |

Per kelas pada holdout: Grade 5 dan Grade 1 paling mudah diprediksi (F1 sekitar 0.74 dan 0.67), sedangkan Grade 2 dan Grade 3 paling sulit (F1 sekitar 0.38 dan 0.42).

## Interpretasi

- **Lokasi adalah sinyal terkuat.** Fitur target encoding ward menyumbang sekitar 57% total gain model, ditambah vdcmun. Tingkat kerusakan sangat berbeda antar wilayah (proporsi Grade 5 per district berkisar 9% sampai 81%), kemungkinan terkait kedekatan dengan pusat gempa dan kondisi tanah, tetapi penyebab pastinya tidak ada di data. Ini bekerja karena semua ward di test juga ada di train. Kalau dipakai untuk wilayah baru yang tidak ada di data latih, performanya akan turun.
- **Struktur bangunan** (usia, jenis atap, lantai dasar, luas dan tinggi) menjadi sinyal kedua. Fitur turunan seperti `slenderness` dan `volume_proxy` masuk 12 besar importance.
- **Kesalahan umumnya antar kelas bertetangga** (misal Grade 3 tertukar dengan Grade 4), terlihat dari QWK 0.73 yang jauh lebih tinggi dari akurasi. Kelas ordinal ini memang bersebelahan dan fitur yang ada belum cukup membedakannya.
- **Batasan**: akurasi sekitar 0.59 menunjukkan sebagian besar variasi kerusakan tidak terjelaskan oleh fitur yang tersedia (tidak ada data gempa seperti jarak ke episentrum atau jenis tanah). Menambah kompleksitas model memberi peningkatan yang sangat kecil (tuning hanya menaikkan akurasi sekitar 0.5 poin).
- **Public vs Private score**: keduanya berasal dari test yang sama, jadi pemilihan model berdasarkan CV (bukan berdasarkan public score) lebih aman agar tidak overfit ke public leaderboard.

## Catatan Aturan

- Dataset tidak di-commit (`data/*.csv` ada di `.gitignore`), cara unduh ada di atas.
- Semua function dan class di `scripts/` memiliki docstring.
- Tidak menggunakan model Neural Network.
