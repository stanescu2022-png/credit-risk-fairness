"""Shared preparation and baseline definitions for the existing research notebooks."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/default of credit card clients.xls"
DATA_PATH = ROOT / "data/processed/taiwan_credit_clean.csv"
TABLES = ROOT / "outputs/tables"
CATEGORICAL = ["EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERIC = ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = NUMERIC + CATEGORICAL


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_data(raw):
    """Keep the original cleaning decisions and preserve ID as metadata."""
    df = raw.rename(columns={"default payment next month": "default"}).copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    if df.ID.isna().any() or not df.ID.is_unique:
        raise ValueError("ID must be present and unique for traceability.")
    if df.isna().any().any():
        raise ValueError("Unexpected missing data; review before splitting.")
    return df


def split_metadata(df):
    """Repair the seeded original split by grouping exact model predictors.

    The lowest-ID member anchors each group's original assignment. This rule
    uses neither model performance nor group outcomes to choose a destination.
    Exact equality is evaluated on columns, not on potentially colliding hashes.
    """
    df = df.reset_index(drop=True)
    train, test = train_test_split(df.index, test_size=0.20, random_state=42, stratify=df["default"])
    metadata = df[["ID", "SEX", "default"]].copy()
    metadata["original_split"] = "train"
    metadata.loc[test, "original_split"] = "test"
    metadata["duplicate_group_id"] = df.groupby(FEATURES, sort=False, dropna=False)["ID"].transform("min")
    original_by_id = metadata.set_index("ID")["original_split"]
    metadata["split"] = metadata.duplicate_group_id.map(original_by_id)
    # Preserve the old within-partition order; append moved rows in source order.
    positions = list(train) + list(test)
    ordered = metadata.loc[positions].copy()
    for name in ("train", "test"):
        retained = ordered.index[(ordered.split == name) & (ordered.original_split == name)].tolist()
        moved = metadata.index[(metadata.split == name) & (metadata.original_split != name)].tolist()
        metadata.loc[retained + moved, "partition_order"] = np.arange(len(retained) + len(moved))
    metadata["partition_order"] = metadata.partition_order.astype(int)
    assert metadata.groupby("duplicate_group_id").split.nunique().max() == 1
    return metadata


def duplicate_summary(df, columns, original):
    groups = df.groupby(columns, sort=False, dropna=False).ngroup()
    duplicated = df.duplicated(columns, keep=False)
    cross = set(groups[original == "train"]) & set(groups[original == "test"])
    return {
        "excess_duplicate_rows": int(df.duplicated(columns).sum()),
        "records_in_duplicate_groups": int(duplicated.sum()),
        "duplicate_groups": int(groups[duplicated].nunique()),
        "original_crossing_groups": len(cross),
        "original_train_records_in_crossing_groups": int((groups.isin(cross) & (original == "train")).sum()),
        "original_test_records_in_crossing_groups": int((groups.isin(cross) & (original == "test")).sum()),
    }


def prepare_data():
    raw = pd.read_excel(RAW_PATH, header=1, engine="xlrd")
    df = clean_data(raw)
    metadata = split_metadata(df)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    metadata.to_csv(DATA_PATH.parent / "split_metadata.csv", index=False)
    for name in ("train", "test"):
        metadata[metadata.split == name].sort_values("partition_order").to_csv(
            DATA_PATH.parent / f"{name}_metadata.csv", index=False
        )
    original = metadata.original_split
    audit = {
        "raw_all_except_ID": duplicate_summary(raw, raw.columns.drop("ID").tolist(), original),
        "clean_all_except_ID": duplicate_summary(df, df.columns.drop("ID").tolist(), original),
        "clean_predictors_including_SEX": duplicate_summary(df, df.columns.drop(["ID", "default"]).tolist(), original),
        "model_predictors_excluding_ID_SEX_target": duplicate_summary(df, FEATURES, original),
        "predictor_groups_with_conflicting_outcomes": int(df.groupby(metadata.duplicate_group_id)["default"].nunique().gt(1).sum()),
        "new_crossing_predictor_groups": int(metadata.groupby("duplicate_group_id").split.nunique().gt(1).sum()),
        "records_retained": len(df),
        "records_moved": int((metadata.split != original).sum()),
        "train_to_test": int(((original == "train") & (metadata.split == "test")).sum()),
        "test_to_train": int(((original == "test") & (metadata.split == "train")).sum()),
        "partition_summary": {},
    }
    for split_column in ("original_split", "split"):
        audit["partition_summary"][split_column] = {
            name: {"records": int(len(part)), "defaults": int(part.default.sum()), "default_rate": float(part.default.mean())}
            for name, part in metadata.groupby(split_column)
        }
    duplicate_records = df[df.duplicated(FEATURES, keep=False)].merge(
        metadata.drop(columns=["SEX", "default"]), on="ID", validate="one_to_one"
    )
    duplicate_records.to_csv(TABLES / "duplicate_records.csv", index=False)
    metadata[metadata.split != original].to_csv(TABLES / "split_changes.csv", index=False)
    (TABLES / "duplicate_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "raw_sha256": sha256(RAW_PATH), "processed_sha256": sha256(DATA_PATH),
        "split_sha256": sha256(DATA_PATH.parent / "split_metadata.csv"),
        "random_state": 42, "initial_test_size": 0.20,
        "split_rule": "Original target-stratified split; all exact predictor-group members follow the lowest-ID member's original partition.",
        "features": FEATURES,
    }
    (DATA_PATH.parent / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return df, metadata, audit


def load_split(df):
    """Use the persisted partition and fail if data or metadata are stale."""
    manifest = json.loads((DATA_PATH.parent / "preparation_manifest.json").read_text())
    for path, key in [(DATA_PATH, "processed_sha256"), (RAW_PATH, "raw_sha256"),
                      (DATA_PATH.parent / "split_metadata.csv", "split_sha256")]:
        if sha256(path) != manifest[key]:
            raise ValueError("Preparation inputs changed; rerun script_prepare_data.py.")
    meta = pd.read_csv(DATA_PATH.parent / "split_metadata.csv")
    if not df[["ID", "SEX", "default"]].reset_index(drop=True).equals(meta[["ID", "SEX", "default"]]):
        raise ValueError("Data and split metadata are not aligned.")
    parts = [meta[meta.split == name].sort_values("partition_order") for name in ("train", "test")]
    tr, te = [p.index for p in parts]
    X = df.drop(columns=["ID", "SEX", "default"])
    if set(X.columns) != set(FEATURES):
        raise ValueError("Unexpected predictor schema.")
    return X.loc[tr], X.loc[te], df.default.loc[tr], df.default.loc[te], parts[0], parts[1]


def build_model(kind):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    if kind == "logistic":
        classifier = LogisticRegression(max_iter=2000, random_state=42)
    elif kind == "gradient_boosting":
        classifier = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(kind)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def save_pipeline(model, kind, X_test, y_pred, y_prob):
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    path = folder / f"baseline_{kind}.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict(X_test), y_pred)
    np.testing.assert_array_equal(restored.predict_proba(X_test)[:, 1], y_prob)
    manifest = {
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("pandas", "numpy", "scipy", "scikit-learn", "joblib")},
        "model_sha256": sha256(path), "processed_sha256": sha256(DATA_PATH),
        "split_sha256": sha256(DATA_PATH.parent / "split_metadata.csv"),
        "feature_columns": model.feature_names_in_.tolist(),
        "classifier_parameters": model.named_steps["classifier"].get_params(),
        "decision_rule": "Unchanged sklearn predict(), default probability boundary 0.5",
        "reload_predictions_identical": True,
    }
    (folder / f"baseline_{kind}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Saved and verified fitted pipeline:", path)
