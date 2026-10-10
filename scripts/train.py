"""Training, tuning, dan pemilihan jumlah pohon untuk model LightGBM.

Dapat dijalankan sebagai script::

    python -m scripts.train --data-dir data --model-path models/pipeline_lgbm_structural_damage.pkl
"""
import argparse
import json
import os
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from scripts.feature_engineering import add_features
from scripts.preprocess import TARGET, build_preprocessor, get_fit_params, load_data

DEFAULT_PARAMS = dict(
    learning_rate=0.1,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    max_bin=127,
    cat_smooth=30,
    random_state=42,
    verbose=-1,
)

SEARCH_SPACE = {
    "algo__num_leaves": [31, 63, 127, 255],
    "algo__min_child_samples": [10, 20, 50, 100],
    "algo__colsample_bytree": [0.5, 0.7, 0.9],
    "algo__reg_lambda": [0.0, 1.0, 5.0, 10.0],
    "algo__cat_smooth": [10, 30, 100],
}


def build_pipeline(**params) -> Pipeline:
    """Membangun Pipeline lengkap: feature engineering -> preprocessing -> LightGBM.

    Args:
        **params: Hyperparameter LGBMClassifier yang menimpa ``DEFAULT_PARAMS``.

    Returns:
        Pipeline scikit-learn dengan step ``fe``, ``prep``, dan ``algo``.
    """
    cfg = {**DEFAULT_PARAMS, **params}
    return Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("algo", LGBMClassifier(**cfg)),
    ])


def tune_pipeline(X, y, n_iter=6, cv=3, sample_size=100_000, n_estimators=200, random_state=42):
    """Randomized search hyperparameter pada subsampel data.

    Subsampel dipakai agar tuning cepat; jumlah pohon final dipilih terpisah
    oleh ``select_n_estimators``.

    Args:
        X: Fitur training (DataFrame mentah).
        y: Target training.
        n_iter: Jumlah kombinasi parameter yang dicoba.
        cv: Jumlah fold cross-validation.
        sample_size: Banyak baris yang dipakai untuk tuning.
        n_estimators: Jumlah pohon selama tuning.
        random_state: Seed.

    Returns:
        Objek ``RandomizedSearchCV`` yang sudah di-fit (``refit=False``).
    """
    Xs = X.sample(min(sample_size, len(X)), random_state=random_state)
    ys = y.loc[Xs.index]
    search = RandomizedSearchCV(
        build_pipeline(n_estimators=n_estimators),
        param_distributions=SEARCH_SPACE,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        refit=False,
        random_state=random_state,
        verbose=2,
    )
    search.fit(Xs, ys, **get_fit_params())
    return search


def select_n_estimators(pipe: Pipeline, X_val, y_val, step=50):
    """Mencari jumlah pohon terbaik dari satu model yang sudah di-fit.

    Prediksi dihitung pada beberapa ``num_iteration`` sehingga tidak perlu
    melatih ulang model berkali-kali.

    Args:
        pipe: Pipeline yang sudah di-fit dengan ``n_estimators`` besar.
        X_val: Fitur validasi.
        y_val: Target validasi.
        step: Selang jumlah pohon yang dievaluasi.

    Returns:
        Tuple ``(best_n, curve)``; ``curve`` adalah DataFrame akurasi per jumlah pohon.
    """
    Xt = pipe[:-1].transform(X_val)
    algo = pipe.named_steps["algo"]
    rows = []
    for k in range(step, algo.n_estimators + 1, step):
        pred = algo.predict(Xt, num_iteration=k)
        rows.append({"n_trees": k, "accuracy": accuracy_score(y_val, pred),
                     "f1_macro": f1_score(y_val, pred, average="macro")})
    curve = pd.DataFrame(rows)
    best_n = int(curve.loc[curve["accuracy"].idxmax(), "n_trees"])
    return best_n, curve


def main():
    """Entry point CLI: tuning -> pilih jumlah pohon -> refit pada seluruh data -> simpan model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-path", default="models/pipeline_lgbm_structural_damage.pkl")
    ap.add_argument("--metrics-path", default="reports/metrics.json")
    ap.add_argument("--n-iter", type=int, default=6)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--tune-sample", type=int, default=100_000)
    ap.add_argument("--max-trees", type=int, default=700)
    ap.add_argument("--train-sample", type=int, default=None, help="subset data (untuk uji cepat)")
    args = ap.parse_args()

    train, _, _ = load_data(args.data_dir)
    if args.train_sample:
        train = train.sample(args.train_sample, random_state=42)
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    search = tune_pipeline(X_tr, y_tr, n_iter=args.n_iter, cv=args.cv, sample_size=args.tune_sample)
    best = {k.replace("algo__", ""): v for k, v in search.best_params_.items()}

    pipe = build_pipeline(n_estimators=args.max_trees, **best)
    pipe.fit(X_tr, y_tr, **get_fit_params())
    best_n, curve = select_n_estimators(pipe, X_va, y_va, step=max(args.max_trees // 14, 5))
    pred = pipe.named_steps["algo"].predict(pipe[:-1].transform(X_va), num_iteration=best_n)

    final = build_pipeline(n_estimators=int(best_n * 1.1), **best)
    final.fit(X, y, **get_fit_params())

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.model_path, "wb") as f:
        pickle.dump(final, f)
    metrics = {"best_params": best, "best_n_trees_holdout": best_n,
               "final_n_trees": int(best_n * 1.1),
               "holdout_accuracy": accuracy_score(y_va, pred),
               "holdout_f1_macro": f1_score(y_va, pred, average="macro")}
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(metrics)


if __name__ == "__main__":
    main()
