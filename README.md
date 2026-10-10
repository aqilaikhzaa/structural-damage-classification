# Structural Damage Classification (Kaggle) - Supervised Machine Learning

Klasifikasi multi-kelas tingkat kerusakan bangunan pasca gempa (`damage_grade`: Grade 1-5) dari fitur struktural dan lokasi bangunan.
Kompetisi: <https://www.kaggle.com/competitions/structural-damage-classification>

**Model akhir:** LightGBM (gradient boosting, bukan neural network) di dalam `Pipeline` + `ColumnTransformer`.
Akurasi holdout 20%: **0,5975** | Macro-F1: **0,5597**.

## Struktur Repository
```
structural-damage-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                      # CSV kompetisi (tidak di-commit, lihat cara unduh)
├── notebooks/
│   ├── 01_eda.ipynb           # eksplorasi data
│   ├── 02_modeling.ipynb      # splitting, pipeline, tuning, evaluasi, simpan model
│   └── 03_submission.ipynb    # prediksi test dan pembuatan submission.csv
├── scripts/
│   ├── __init__.py
│   ├── feature_engineering.py # fitur turunan + daftar kolom
│   ├── preprocess.py          # load data, ColumnTransformer
│   ├── train.py               # pipeline, tuning, pemilihan n_estimators (bisa lewat CLI)
│   └── predict.py             # membuat submission (bisa lewat CLI)
├── models/pipeline_lgbm_structural_damage.pkl
├── submissions/submission.csv
└── reports/
    ├── metrics.json
    ├── feature_importance.csv
    └── figures/
```

## Cara Menggunakan
1. **Install dependency**
   ```bash
   python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Unduh data** (harus sudah klik *Join Competition* dan *I Understand and Accept* di web Kaggle)
   ```bash
   kaggle competitions download -c structural-damage-classification
   unzip structural-damage-classification.zip -d data/
   ```
   Atau unduh manual dari tab *Data* dan taruh `train.csv`, `test.csv`, `sample_submission.csv` di folder `data/`.
3. **Jalankan notebook berurutan** dari folder `notebooks/`: `01_eda` -> `02_modeling` -> `03_submission`.
   Notebook 02 memakan waktu sekitar 25 menit pada 1 CPU (tuning + training 700 pohon + refit penuh).
4. **Alternatif lewat script** (dijalankan dari root repo):
   ```bash
   python -m scripts.train      # tuning, training, simpan model ke models/
   python -m scripts.predict    # membuat submissions/submission.csv
   ```
5. **Submit**: unggah `submissions/submission.csv` di tab *Submit Predictions*, atau
   `kaggle competitions submit -c structural-damage-classification -f submissions/submission.csv -m "LightGBM pipeline"`.

## Workflow Analisis
1. **Read dataset**: train 533.465 baris x 17 kolom, test 228.629 baris.
2. **Quick EDA**: cek duplikasi, missing value, statistik deskriptif, distribusi, imbalance target, hubungan lokasi dan material dengan target.
3. **Splitting**: 80:20 `stratify` berdasarkan target (`random_state=42`).
4. **Preprocessing** (`Pipeline` + `ColumnTransformer`):
   - `FunctionTransformer(add_features)`: clip `age_building` di 100, rasio ukuran (luas/lantai, tinggi/lantai, volume), potongan digit `building_id`, ID lokasi diubah menjadi string.
   - Numerik: `SimpleImputer(median)`. Scaling tidak dipakai karena model pohon tidak sensitif skala.
   - Kategorik (8 kolom + `district_id`, `vdcmun_id`, `ward_id`): `SimpleImputer(most_frequent)` + `OrdinalEncoder`, lalu dibaca LightGBM sebagai fitur kategorik native.
5. **Training**: baseline Decision Tree (seperti contoh asisten), lalu LightGBM.
6. **Tuning**: `RandomizedSearchCV` (6 kombinasi, 3-fold, scoring accuracy) pada subsampel 100.000 baris; jumlah pohon dipilih dari kurva skor vs jumlah pohon.
7. **Evaluasi**: classification report, confusion matrix, feature importance.
8. **Final fit**: dilatih ulang pada seluruh data train (660 pohon), disimpan ke `models/`.

## Hasil (holdout 20%)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Decision Tree (baseline) | 0,5526 | 0,5089 |
| LightGBM awal (200 pohon) | 0,5929 | 0,5539 |
| LightGBM tuned (600 pohon) | **0,5975** | **0,5597** |

Parameter terbaik: `num_leaves=63`, `min_child_samples=100`, `colsample_bytree=0.7`, `reg_lambda=5`, `cat_smooth=10`, `learning_rate=0.1`.

Per kelas (model tuned):

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Grade 1 | 0,65 | 0,73 | 0,69 |
| Grade 2 | 0,45 | 0,37 | 0,40 |
| Grade 3 | 0,46 | 0,42 | 0,44 |
| Grade 4 | 0,53 | 0,52 | 0,53 |
| Grade 5 | 0,72 | 0,77 | 0,74 |

## Laporan Interpretasi
**Data.** Tidak ada missing value maupun `building_id` ganda. Target tidak seimbang (Grade 5 ~36%, Grade 1 ~10%). `age_building` memiliki outlier hingga 999 sehingga di-clip pada 100. Kolom `superstructure` berupa satu kolom kategorik (11 nilai), berbeda dari flag boolean di `metadata.csv`.

**Lokasi adalah faktor paling dominan.** Proporsi Grade 5 sangat bervariasi antar district (sekitar 9% di district 31 hingga sekitar 81% di district 23). Pada feature importance (gain), `ward_id` menyumbang 37,3% dan `vdcmun_id` 21,6%, disusul `superstructure` 11,3%. Fitur ukuran (luas, tinggi, volume) dan umur masing-masing hanya 2-4%. Interpretasinya: kerusakan sangat bergantung pada kedekatan dengan pusat guncangan dan kondisi lokal, lalu pada material utama bangunan. Ini juga sebabnya lokasi harus diperlakukan sebagai kategori; sebagai angka urut, akurasi turun dari 0,58 menjadi 0,556 pada eksperimen subsampel, dan tanpa fitur lokasi hanya 0,459.

**Mengapa LightGBM.** Model harus menangkap interaksi antara ratusan ward dan jenis material. Decision Tree tunggal hanya mencapai akurasi 0,5526, sedangkan boosting dengan penanganan kategorik native menghindari One-Hot pada kolom `ward_id` (~945 nilai) dan naik ke 0,5975. Tuning memberi tambahan kecil (+0,5 poin) dibanding parameter awal.

**Confusion matrix dan kelas.** Kelas ekstrem (Grade 1 dan Grade 5) paling mudah diprediksi (F1 0,69 dan 0,74). Kelas menengah (Grade 2 dan 3, F1 0,40 dan 0,44) paling sulit karena tingkat kerusakan berurutan dan batas antar-grade bersifat subjektif (penilaian surveyor), sehingga sebagian besar kesalahan jatuh ke grade bersebelahan. Lihat `reports/figures/confusion_matrix.png`.

**Catatan dan keterbatasan.**
- Fitur dari potongan `building_id` (`bid_seq`, `bid_prefix`, `bid_mid`) ikut dipakai karena menambah sekitar 1 poin akurasi pada eksperimen. Ini fitur dari ID, bukan sifat fisik bangunan, kemungkinan menangkap urutan/area survei. Kontribusinya (sekitar 8% gain) perlu diinterpretasi hati-hati.
- Jumlah pohon dipilih memakai data holdout yang sama dengan pelaporan, sehingga skor holdout sedikit optimis.
- Tuning terbatas (6 kombinasi, subsampel 100 ribu) karena keterbatasan komputasi; skor dapat naik dengan pencarian lebih luas, `learning_rate` lebih kecil, atau ensembling.
- Format submission: kolom `damage_grade` ditulis sebagai angka 1-5, karena evaluator Kaggle menolak teks "Grade 3" (error "input string 'Grade' was not in a correct format"), walaupun `sample_submission.csv` memakai teks. Gunakan `--as-text` pada `scripts.predict` jika perlu format teks.
- Metrik resmi kompetisi diasumsikan accuracy/micro-F1; macro-F1 dilaporkan sebagai pembanding karena kelas tidak seimbang.
