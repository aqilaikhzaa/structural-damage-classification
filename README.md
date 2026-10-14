# Structural Damage Classification (Kaggle) - Supervised Machine Learning

Klasifikasi multi-kelas tingkat kerusakan bangunan pasca gempa (`damage_grade`: Grade 1-5) dari fitur struktural dan lokasi bangunan.
Kompetisi: <https://www.kaggle.com/competitions/structural-damage-classification>

**Model akhir:** LightGBM berbasis gradient boosting yang diintegrasikan dalam `Pipeline` dengan `ColumnTransformer`.
Akurasi holdout 20%: **0,5975** | Macro-F1: **0,5597**.

## Struktur Repository
```
structural-damage-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                      # CSV kompetisi 
├── notebooks/
│   ├── 01_eda.ipynb           # eksplorasi data
│   ├── 02_modeling.ipynb      # splitting, pipeline, tuning, evaluasi, simpan model
│   └── 03_submission.ipynb    # prediksi test dan pembuatan submission.csv
├── scripts/
│   ├── __init__.py
│   ├── feature_engineering.py # fitur turunan + daftar kolom
│   ├── preprocess.py          # load data, ColumnTransformer
│   ├── train.py               # pipeline, tuning, pemilihan n_estimators 
│   └── predict.py             # membuat submission 
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

## Laporan Interpretasi Model

### Analisis Data

Hasil eksplorasi menunjukkan bahwa dataset tidak memiliki nilai kosong (*missing value*) dan tidak ditemukan duplikasi pada variabel identitas bangunan (`building_id`). Distribusi target menunjukkan adanya ketidakseimbangan kelas (*class imbalance*), dengan Grade 5 sebagai kelas dominan sekitar 36%, sedangkan Grade 1 memiliki proporsi sekitar 10%.

Pada fitur numerik, variabel `age_building` memiliki nilai ekstrem hingga 999 tahun. Untuk mengurangi pengaruh pencilan terhadap proses pembelajaran model, dilakukan proses *clipping* dengan membatasi nilai maksimum umur bangunan menjadi 100 tahun.

Selain itu, fitur `superstructure` diperlakukan sebagai satu variabel kategorikal dengan 11 kategori berdasarkan struktur data aktual pada dataset. Pendekatan ini berbeda dengan representasi *boolean flag* pada `metadata.csv`, sehingga preprocessing disesuaikan dengan karakteristik data yang tersedia.

---

### Faktor Dominan terhadap Prediksi Kerusakan

Hasil analisis menunjukkan bahwa faktor lokasi merupakan variabel yang paling berpengaruh terhadap prediksi tingkat kerusakan bangunan. Perbedaan kondisi wilayah menyebabkan proporsi tingkat kerusakan yang berbeda antar area, misalnya persentase Grade 5 berkisar sekitar 9% pada district 31 hingga sekitar 81% pada district 23.

Berdasarkan *feature importance* menggunakan metode gain, fitur `ward_id` memberikan kontribusi terbesar sebesar 37,3%, diikuti oleh `vdcmun_id` sebesar 21,6%, dan `superstructure` sebesar 11,3%. Sementara itu, fitur karakteristik fisik bangunan seperti luas, tinggi, volume, dan umur bangunan memberikan kontribusi yang relatif lebih kecil, yaitu sekitar 2-4%.

Hasil tersebut menunjukkan bahwa tingkat kerusakan bangunan sangat dipengaruhi oleh kondisi wilayah dan faktor lokal, selain karakteristik material bangunan. Oleh karena itu, fitur lokasi dipertahankan sebagai variabel kategorikal agar model dapat mempelajari pola antarwilayah secara lebih optimal.

Eksperimen tambahan menunjukkan bahwa penggunaan fitur lokasi memberikan peningkatan performa model. Ketika fitur lokasi tidak digunakan, akurasi mengalami penurunan hingga sekitar 0,459. Selain itu, perlakuan fitur lokasi sebagai kategori menghasilkan performa lebih baik dibandingkan menganggapnya sebagai nilai numerik berurutan.

---

### Alasan Pemilihan LightGBM

LightGBM dipilih sebagai model utama karena mampu menangani dataset berukuran besar dengan kombinasi fitur numerik dan kategorikal serta mampu menangkap hubungan kompleks antarvariabel.

Sebagai model pembanding awal, Decision Tree menghasilkan akurasi sebesar 0,5526. Namun, model pohon tunggal memiliki keterbatasan dalam mempelajari interaksi kompleks antara ratusan kategori lokasi dan jenis material bangunan.

Penggunaan metode *gradient boosting* pada LightGBM memungkinkan model membangun kombinasi beberapa pohon keputusan secara bertahap sehingga menghasilkan prediksi yang lebih baik. Selain itu, LightGBM mampu mengatasi fitur kategorikal dengan jumlah kategori tinggi seperti `ward_id` yang memiliki sekitar 945 kategori tanpa harus melakukan One-Hot Encoding.

Hasil tuning parameter meningkatkan performa model menjadi akurasi 0,5975. Peningkatan tersebut menunjukkan bahwa konfigurasi parameter yang disesuaikan mampu memberikan performa lebih baik dibandingkan model awal.

---

### Analisis Performa Berdasarkan Kelas

Berdasarkan hasil evaluasi model, performa prediksi berbeda pada setiap kategori tingkat kerusakan.

Kelas dengan tingkat kerusakan ekstrem, yaitu Grade 1 dan Grade 5, memiliki performa terbaik dengan nilai F1-score masing-masing sebesar 0,69 dan 0,74. Hal ini menunjukkan bahwa model lebih mudah mengenali bangunan dengan kondisi kerusakan yang sangat rendah maupun sangat tinggi.

Sebaliknya, Grade 2 dan Grade 3 menjadi kelas yang paling sulit diprediksi dengan nilai F1-score masing-masing sebesar 0,40 dan 0,44. Kesalahan klasifikasi sebagian besar terjadi antar kelas yang berdekatan karena tingkat kerusakan bangunan memiliki karakteristik bertahap dan batas antar kategori dapat memiliki kemiripan berdasarkan hasil penilaian surveyor.

Evaluasi detail performa model tersedia pada notebook:

`notebooks/02_modeling.ipynb`


---

### Catatan dan Keterbatasan Model

Meskipun LightGBM memberikan performa terbaik pada eksperimen ini, terdapat beberapa keterbatasan yang perlu diperhatikan:

- Fitur turunan dari `building_id`, yaitu `bid_seq`, `bid_prefix`, dan `bid_mid`, digunakan karena memberikan peningkatan performa sekitar 1 poin akurasi pada eksperimen. Namun, fitur tersebut bukan merupakan karakteristik fisik bangunan secara langsung sehingga kontribusinya perlu diinterpretasikan secara hati-hati karena kemungkinan menangkap pola urutan data atau area survei.

- Pemilihan jumlah pohon dilakukan menggunakan data *holdout* yang sama dengan evaluasi akhir, sehingga nilai performa yang diperoleh dapat sedikit lebih optimis dibandingkan kemampuan generalisasi sebenarnya.

- Proses tuning masih terbatas pada 6 kombinasi parameter menggunakan subsampel 100 ribu data karena mempertimbangkan keterbatasan komputasi. Performa model masih berpotensi ditingkatkan melalui pencarian parameter yang lebih luas, penggunaan *learning rate* yang lebih kecil, atau pendekatan *ensemble learning*.

- Metrik utama kompetisi diasumsikan menggunakan accuracy atau micro-F1, sedangkan Macro-F1 digunakan sebagai metrik tambahan untuk memberikan gambaran performa model pada kondisi kelas yang tidak seimbang.
