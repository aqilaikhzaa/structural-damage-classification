"""Feature engineering untuk dataset Structural Damage Classification.

Modul ini berisi daftar kolom akhir yang dipakai model serta fungsi
``add_features`` yang dibungkus ``FunctionTransformer`` di dalam Pipeline,
sehingga transformasi yang sama otomatis diterapkan pada train maupun test.
"""
import pandas as pd

AGE_CAP = 100  # umur bangunan di atas 100 tahun (maks. 999) dianggap outlier / kode khusus

GEO_COLUMNS = ["district_id", "vdcmun_id", "ward_id"]

NUMERIC_COLUMNS = [
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "height_ft_pre_eq",
    "area_per_floor",
    "height_per_floor",
    "volume",
    "bid_prefix",
    "bid_mid",
    "bid_seq",
]

CATEGORICAL_COLUMNS = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "superstructure",
] + GEO_COLUMNS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur turunan dan mengembalikan kolom yang dipakai model.

    Langkah yang dilakukan:
      1. ``age_building`` di-clip pada ``AGE_CAP``.
      2. Rasio ukuran bangunan: luas per lantai, tinggi per lantai, dan volume.
      3. Potongan digit ``building_id`` (``bid_prefix``, ``bid_mid``, ``bid_seq``).
      4. ID lokasi (district, vdc/municipality, ward) diubah ke string agar
         diperlakukan sebagai kategori, bukan angka berurutan.

    Args:
        df: DataFrame mentah (harus memuat ``building_id``).

    Returns:
        DataFrame baru dengan kolom ``NUMERIC_COLUMNS + CATEGORICAL_COLUMNS``.
    """
    out = df.copy()
    out["age_building"] = out["age_building"].clip(upper=AGE_CAP)

    floors = out["count_floors_pre_eq"].clip(lower=1)
    out["area_per_floor"] = out["plinth_area_sq_ft"] / floors
    out["height_per_floor"] = out["height_ft_pre_eq"] / floors
    out["volume"] = out["plinth_area_sq_ft"] * out["height_ft_pre_eq"]

    bid = out["building_id"].astype("int64")
    out["bid_prefix"] = bid // 10**6          # 6 digit pertama
    out["bid_mid"] = out["bid_prefix"] % 10**4  # digit ke-3 s.d. ke-6
    out["bid_seq"] = bid % 10**6              # 6 digit terakhir

    for col in GEO_COLUMNS:
        out[col] = out[col].astype(str)

    return out[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
