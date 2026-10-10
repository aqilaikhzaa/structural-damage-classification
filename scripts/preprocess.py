"""Preprocessing: pemuatan data dan ColumnTransformer untuk model."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from scripts.feature_engineering import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

TARGET = "damage_grade"


def load_data(data_dir: str = "data"):
    """Membaca train.csv, test.csv, dan sample_submission.csv.

    Args:
        data_dir: Folder tempat file CSV kompetisi berada.

    Returns:
        Tuple ``(train, test, sample_submission)`` berupa DataFrame.
    """
    d = Path(data_dir)
    return (
        pd.read_csv(d / "train.csv"),
        pd.read_csv(d / "test.csv"),
        pd.read_csv(d / "sample_submission.csv"),
    )


def build_preprocessor() -> ColumnTransformer:
    """Membangun ColumnTransformer (numerik + kategorik).

    - Numerik: imputer median. Scaling tidak diperlukan karena model berbasis
      pohon tidak sensitif terhadap skala fitur.
    - Kategorik: imputer modus lalu ``OrdinalEncoder``. Kode integer ini dibaca
      LightGBM sebagai fitur kategorik native (lebih baik daripada One-Hot untuk
      kolom ber-kardinalitas tinggi seperti ``ward_id`` dengan ~945 nilai).
      Kategori yang tidak dikenal saat prediksi diberi kode -1.

    Returns:
        ColumnTransformer dengan urutan keluaran: numerik dahulu, lalu kategorik.
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])
    return ColumnTransformer([
        ("numeric", numerical_pipeline, NUMERIC_COLUMNS),
        ("categoric", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


def get_fit_params() -> dict:
    """Parameter ``fit`` agar LightGBM memperlakukan kolom kategorik sebagai kategori.

    Returns:
        Dict ``{"algo__categorical_feature": [indeks kolom kategorik]}``.
    """
    start = len(NUMERIC_COLUMNS)
    return {"algo__categorical_feature": list(range(start, start + len(CATEGORICAL_COLUMNS)))}
