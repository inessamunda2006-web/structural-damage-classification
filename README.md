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
│   ├── neighbor_features.py
│   ├── postprocess.py
│   ├── models.py
│   ├── tune.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── pipeline_model_lgbm.pkl
│   ├── best_params.json
│   ├── class_weights.json
│   └── cv_scores.csv          (CV model awal, tanpa fitur tetangga)
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
   git clone <url-repo-kamu>
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
4. Training, evaluasi holdout, dan simpan model (tambahkan `--cv` untuk Stratified 5-fold CV, lebih lambat):
   ```bash
   python -m scripts.train
   ```
5. Buat file submission (label angka 1-5, sesuai yang diterima evaluator Kaggle):
   ```bash
   python -m scripts.predict              # argmax biasa (default)
   python -m scripts.predict --weighted   # opsional: bobot kelas untuk macro-F1
   ```
   Output: `submissions/submission.csv`. Upload ke Kaggle atau lewat CLI:
   ```bash
   kaggle competitions submit -c structural-damage-classification -f submissions/submission.csv -m "LightGBM pipeline"
   ```

Notebook di `notebooks/` dijalankan dari foldernya sendiri dan memakai modul dari `scripts/`.

## Workflow Analisis

1. **EDA** (`01_eda.ipynb`): distribusi target, missing value, anomali, pengaruh lokasi dan material.
2. **Fitur tetangga** (`NeighborTargetFeatures`): `building_id` berurutan mengikuti urutan survei, jadi label bangunan train dengan ID berdekatan (jendela 3, 10, 30) dipakai sebagai fitur, mirip KNN. Pada data train dihitung leave-one-out supaya label baris itu sendiri tidak bocor ke fiturnya.
3. **Feature engineering** (`BuildingFeatureAdder`): `age_building = 999` dianggap placeholder (jadi NaN + flag), rasio tinggi per lantai, luas per lantai, proxy volume, slenderness, log luas.
4. **Preprocessing** (`ColumnTransformer`): numerik di-impute median, kategorik `OrdinalEncoder`, kolom lokasi (district, vdcmun, ward) dipakai sebagai angka asli dan juga `TargetEncoder` multikelas (cross-fitting internal supaya tidak bocor).
5. **Model**: `Pipeline` = NeighborTargetFeatures, BuildingFeatureAdder, ColumnTransformer, `LGBMClassifier`. Tidak memakai Neural Network. Perbandingan awal dengan Logistic Regression (OvR dan OvO), Random Forest, HistGradientBoosting ada di `02_modeling.ipynb`.
6. **Tuning**: `RandomizedSearchCV` (10 kombinasi, 3-fold, 25% data), lalu jumlah pohon final dipilih dari kurva validasi holdout. Tuning dilakukan sebelum fitur tetangga ditambahkan dan tidak diulang, jadi hyperparameter model final mungkin belum optimal.
7. **Evaluasi**: holdout 80/20 (stratified) untuk model final, dan Stratified 5-fold CV untuk model awal.
8. **Prediksi**: model dilatih ulang di seluruh train, lalu `predict.py` menulis `submission.csv` (kolom sama dengan `sample_submission.csv`, label angka 1-5).
9. **Post-processing opsional** (`postprocess.py`): bobot per kelas yang dicari pada data holdout untuk macro-F1. Efeknya kecil (macro-F1 +0.3 poin, akurasi -0.4 poin), jadi tidak dipakai secara default.

## Hasil

Holdout 80/20 (stratified, split yang sama untuk kedua model):

| Model | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| Awal (tanpa fitur tetangga) | 0.5845 | 0.5434 | 0.7314 |
| **Final (dengan fitur tetangga)** | **0.6224** | **0.5869** | **0.7613** |

Model awal juga dievaluasi dengan Stratified 5-fold CV: accuracy 0.5877 (std 0.0013), macro-F1 0.5472, QWK 0.7320 (`models/cv_scores.csv`). Public score submission pertama (model awal) di Kaggle: 0.55031.

Per kelas pada holdout (model final, tanpa bobot kelas): Grade 5 dan Grade 1 paling mudah diprediksi, sedangkan Grade 2 dan Grade 3 paling sulit.

## Interpretasi

Dari hasil di atas, ini yang bisa disimpulkan:

- **Fitur tetangga paling berpengaruh** (sekitar 63% dari total gain model). Setelah dicek, `building_id` ternyata berurutan sesuai urutan survei, dan bangunan yang ID-nya bersebelahan punya grade yang sama sekitar 52% dari waktu, sedangkan kalau acak cuma sekitar 25%. ID bangunan di test juga terselip di antara ID train (jarak median ke ID train terdekat 10), jadi label train di sekitarnya bisa jadi petunjuk. Dengan fitur ini akurasi holdout naik sekitar 3.8 poin.
- **Lokasi jadi petunjuk kedua** (target encoding ward dan vdcmun, ward sekitar 10% gain). Persentase Grade 5 antar district beda jauh, dari 9% sampai 81%. Mungkin karena jarak ke pusat gempa atau kondisi tanahnya, tapi data ini tidak punya informasi itu, jadi penyebabnya tidak bisa dipastikan.
- **Struktur bangunan** (usia, jenis atap, lantai dasar, luas, dan tinggi) ikut membantu, tapi pengaruhnya lebih kecil.
- **Salah prediksi kebanyakan ke kelas yang bersebelahan**, misalnya Grade 3 tertukar dengan Grade 4. Ini kelihatan dari QWK (0.76) yang jauh lebih tinggi daripada akurasi (0.62), artinya kalau salah biasanya meleset cuma satu tingkat.
- **Keterbatasan**: fitur tetangga dan target encoding cuma bisa dipakai karena test diambil acak dari survei yang sama dengan train (semua ward test ada di train, dan ID-nya terselip di antara ID train). Kalau dipakai untuk wilayah atau survei yang tidak ada di data latih, performanya akan turun ke level model tanpa fitur tersebut, sekitar 0.58.
- **Soal public dan private score**: keduanya dihitung dari data test yang sama, jadi model dipilih berdasarkan hasil holdout/CV, bukan public score, supaya tidak overfit ke public leaderboard.

## Catatan

- Dataset tidak ikut di-commit karena ukurannya besar (`data/*.csv` sudah masuk `.gitignore`). Cara download-nya ada di bagian "Cara Menggunakan Repository".
- Semua function dan class di folder `scripts/` sudah ada docstring-nya.
- Model yang dipakai LightGBM (boosting dari pohon keputusan), jadi tidak pakai Neural Network sesuai aturan tugas.
- Tuning hyperparameter dilakukan sebelum fitur tetangga ditambahkan dan belum diulang, jadi masih mungkin bisa lebih bagus kalau dituning ulang.
- Bobot kelas (`postprocess.py`) cuma opsional karena efeknya kecil, jadi submission utama pakai prediksi biasa.
- Kalau mau menjalankan ulang, urutannya: download data, `python -m scripts.train`, lalu `python -m scripts.predict`.
