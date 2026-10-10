"""Membuat file submission dari model yang sudah disimpan.

Contoh::

    python -m scripts.predict --model-path models/pipeline_lgbm_structural_damage.pkl
"""
import argparse
import os
import pickle

import pandas as pd

from scripts.preprocess import load_data


def make_submission(model, test: pd.DataFrame, sample: pd.DataFrame, numeric: bool = True) -> pd.DataFrame:
    """Memprediksi data test dan menyusun DataFrame sesuai format sample_submission.

    Args:
        model: Pipeline hasil training.
        test: DataFrame test mentah (memuat ``building_id``).
        sample: ``sample_submission.csv`` sebagai acuan nama dan urutan kolom.
        numeric: Jika True, ``damage_grade`` ditulis sebagai angka 1-5 (evaluator Kaggle
            kompetisi ini gagal membaca teks "Grade 3"). Jika False, ditulis "Grade 3".

    Returns:
        DataFrame submission dengan kolom ``building_id`` dan ``damage_grade``.
    """
    submission = sample.copy()
    pred = pd.Series(model.predict(test), index=test["building_id"])
    submission["damage_grade"] = submission["building_id"].map(pred).values
    assert submission["damage_grade"].notna().all(), "Ada building_id yang tidak terprediksi"
    if numeric:
        submission["damage_grade"] = submission["damage_grade"].str.replace("Grade ", "", regex=False).astype(int)
    return submission


def main():
    """Entry point CLI untuk membuat ``submissions/submission.csv``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-path", default="models/pipeline_lgbm_structural_damage.pkl")
    ap.add_argument("--output", default="submissions/submission.csv")
    ap.add_argument("--as-text", action="store_true", help='tulis "Grade 3" alih-alih angka 3')
    args = ap.parse_args()

    _, test, sample = load_data(args.data_dir)
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    sub = make_submission(model, test, sample, numeric=not args.as_text)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(sub.shape, sub["damage_grade"].value_counts().to_dict())


if __name__ == "__main__":
    main()
