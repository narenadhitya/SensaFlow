"""MachinaSense — industrial fault detection pipeline.

Usage:
    python pipeline.py                                   # train on AI4I, evaluate on held-out 20% split
    python pipeline.py --train data/train.csv --test data/test.csv
    python pipeline.py --predict data/unseen.csv         # inference only, using saved model + config
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
CONFIG_PATH = ROOT / "feature_config.json"
MODEL_PATH = MODEL_DIR / "model.pkl"
DEFAULT_DATA = ROOT / "datasets" / "ai4i2020.csv"
SEED = 42

RAW_SENSORS = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Canonical name -> accepted aliases (after normalisation: lowercase, units stripped, non-alnum -> "_")
ALIASES = {
    "air_temp": ["air_temperature", "air_temp", "ambient_temperature", "ambient_temp"],
    "process_temp": ["process_temperature", "process_temp"],
    "rpm": ["rotational_speed", "rpm", "speed", "rotation_speed"],
    "torque": ["torque"],
    "tool_wear": ["tool_wear", "wear"],
    "type": ["type", "product_type", "quality", "variant"],
    "id": ["udi", "id", "timestamp", "datetime", "time"],
    "label": ["machine_failure", "label", "failure", "target", "fault"],
}


# ---------------------------------------------------------------- data loading

def normalise(col: str) -> str:
    col = re.sub(r"\[.*?\]|\(.*?\)", "", col.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", col).strip("_")


def load_csv(path) -> pd.DataFrame:
    """Read a CSV (auto-detecting ',' or ';') and rename known columns to canonical names."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    lookup = {normalise(c): c for c in df.columns}
    rename = {}
    for canon, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lookup and lookup[alias] not in rename:
                rename[lookup[alias]] = canon
                break
    return df.rename(columns=rename)


# ----------------------------------------------------------- feature building

def build_features(df: pd.DataFrame, config: dict | None = None, log: list | None = None) -> pd.DataFrame:
    """Raw sensors + Type + physics-derived features.

    With a config (inference), missing sensors / NaNs are imputed with training means
    and the output is aligned to the training feature order. Extra columns are ignored.
    """
    log = log if log is not None else []
    X = pd.DataFrame(index=df.index)
    for col in RAW_SENSORS:
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    X["type"] = df["type"].map(TYPE_MAP) if "type" in df else np.nan

    if config is not None:
        means = config["train_means"]
        for col in RAW_SENSORS + ["type"]:
            n_missing = int(X[col].isna().sum())
            if n_missing == len(X):
                log.append(f"sensor '{col}' absent from input -> imputed with training mean {means[col]:.3f}")
            elif n_missing:
                log.append(f"sensor '{col}': {n_missing} missing values -> imputed with training mean")
            X[col] = X[col].fillna(means[col])

    # Layer 2 — physics features
    X["temp_delta"] = X["process_temp"] - X["air_temp"]                 # heat dissipation
    X["power"] = X["torque"] * X["rpm"] * 2 * np.pi / 60                # mechanical power [W]
    X["temp_rpm_ratio"] = X["process_temp"] / (X["rpm"] + 1)            # thermal stress
    X["wear_rate"] = X["tool_wear"] / (X["rpm"] + 1)                    # tool stress rate
    # Layer 4 — interactions
    X["strain"] = X["torque"] * X["tool_wear"]                          # overstrain indicator

    if config is not None:
        X = X[config["features"]]
    return X


def feature_stats(X: pd.DataFrame) -> tuple[dict, dict]:
    return X.mean().to_dict(), X.std().replace(0, 1).to_dict()


def evidence_strings(X: pd.DataFrame, means: dict, stds: dict, top_k: int = 2) -> list[str]:
    """Per-row evidence: the features deviating most from the training distribution."""
    z = (X - pd.Series(means)) / pd.Series(stds)
    out = []
    for _, row in z.iterrows():
        top = row.abs().sort_values(ascending=False).head(top_k)
        out.append("; ".join(f"{f} {row[f]:+.1f}σ" for f in top.index if abs(row[f]) >= 1.0))
    return out


# --------------------------------------------------------------------- models

def train_candidates(X_tr, y_tr, X_val, y_val) -> dict:
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=50,
        random_state=SEED, n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced",
                                min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)

    results = {}
    for name, model in [("XGBoost", xgb), ("RandomForest", rf)]:
        proba = model.predict_proba(X_val)[:, 1]
        thr, f1 = best_f1_threshold(y_val, proba)
        results[name] = {"model": model, "threshold": thr, "val_f1": f1,
                         "val_aucpr": average_precision_score(y_val, proba)}
    return results


def best_f1_threshold(y, proba) -> tuple[float, float]:
    """Pick the probability cut-off that maximises fault-class F1 (tuned on validation only)."""
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def severity(proba: np.ndarray, threshold: float) -> np.ndarray:
    high = max(0.8, threshold)
    return np.select([proba >= high, proba >= threshold, proba >= threshold * 0.5],
                     ["HIGH", "MEDIUM", "LOW"], default="NONE")


# ----------------------------------------------------------------- reporting

def metrics_block(y, pred, proba) -> tuple[str, dict]:
    m = {
        "f1_fault": f1_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
        "aucpr": average_precision_score(y, proba),
    }
    cm = confusion_matrix(y, pred, labels=[0, 1])
    text = (
        f"F1 Score (fault):   {m['f1_fault']:.3f}\n"
        f"F1 Score (macro):   {m['f1_macro']:.3f}\n"
        f"Precision (fault):  {m['precision']:.3f}\n"
        f"Recall (fault):     {m['recall']:.3f}\n"
        f"AUC-PR:             {m['aucpr']:.3f}\n\n"
        "Confusion Matrix:\n"
        "               Predicted Normal  Predicted Fault\n"
        f"Actual Normal  {cm[0, 0]:>16}  {cm[0, 1]:>15}\n"
        f"Actual Fault   {cm[1, 0]:>16}  {cm[1, 1]:>15}\n"
    )
    m["confusion_matrix"] = cm.tolist()
    return text, m


def predict_frame(df: pd.DataFrame, model, config: dict, log: list) -> pd.DataFrame:
    X = build_features(df, config, log)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= config["threshold"]).astype(int)
    ids = df["id"].values if "id" in df else np.arange(len(df))
    out = pd.DataFrame({
        "timestamp": ids,
        "predicted_label": pred,
        "fault_probability": proba.round(4),
        "alert_severity": severity(proba, config["threshold"]),
    })
    ev = evidence_strings(X, config["feature_means"], config["feature_stds"])
    out["evidence_summary"] = [e if p else "" for e, p in zip(ev, pred)]
    return out, X


def write_outputs(df, preds, X, label=None):
    OUT_DIR.mkdir(exist_ok=True)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    # Richer table for the dashboard: sensors + derived features + predictions (+ ground truth)
    scored = pd.concat([preds.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    if label is not None:
        scored["actual_label"] = np.asarray(label)
    scored.to_csv(OUT_DIR / "scored_data.csv", index=False)


# ---------------------------------------------------------------------- modes

def run_train(train_path, test_path=None):
    df = load_csv(train_path)
    if "label" not in df:
        raise SystemExit(f"No label column found in {train_path}")
    y_all = df["label"].astype(int)

    if test_path:   # external test set: use the whole train file for train/val
        df_trv, y_trv = df, y_all
        df_te = load_csv(test_path)
        y_te = df_te["label"].astype(int) if "label" in df_te else None
        split_desc = f"Train file: {train_path} | Test file: {test_path}"
    else:           # held-out 20% stratified test split
        df_trv, df_te, y_trv, y_te = train_test_split(df, y_all, test_size=0.20, stratify=y_all, random_state=SEED)
        split_desc = "Stratified split: 64% train / 16% validation / 20% test (seed 42)"
    df_tr, df_val, y_tr, y_val = train_test_split(df_trv, y_trv, test_size=0.20, stratify=y_trv, random_state=SEED)

    X_tr, X_val = build_features(df_tr), build_features(df_val)
    raw_means = X_tr[RAW_SENSORS + ["type"]].mean().to_dict()
    feat_means, feat_stds = feature_stats(X_tr)

    results = train_candidates(X_tr, y_tr, X_val, y_val)
    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best = results[best_name]

    config = {
        "model": best_name,
        "features": list(X_tr.columns),
        "raw_sensors": RAW_SENSORS,
        "type_map": TYPE_MAP,
        "threshold": best["threshold"],
        "train_means": raw_means,
        "feature_means": feat_means,
        "feature_stds": feat_stds,
        "n_train": len(X_tr), "n_val": len(X_val),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    log = []
    preds, X_te = predict_frame(df_te, best["model"], config, log)
    write_outputs(df_te, preds, X_te, y_te)

    lines = [
        "MACHINASENSE — EVALUATION REPORT", "=" * 34,
        f"Data: {split_desc}",
        f"Samples: train={len(X_tr)}  val={len(X_val)}  test={len(df_te)}",
        "", "MODEL SELECTION (validation split, F1-optimal threshold)", "-" * 34,
    ]
    for name, r in results.items():
        lines.append(f"{name:<13} val F1={r['val_f1']:.3f}  val AUC-PR={r['val_aucpr']:.3f}  threshold={r['threshold']:.3f}")
    lines += ["", f"Selected: {best_name}  (decision threshold {best['threshold']:.3f}, tuned on validation only)"]
    if best_name == "XGBoost":
        lines.append(f"XGBoost best iteration: {best['model'].best_iteration}")

    if y_te is not None:
        text, m = metrics_block(y_te, preds["predicted_label"], preds["fault_probability"])
        lines += ["", "RESULTS ON TEST SET", "-" * 34, text]
        (OUT_DIR / "metrics.json").write_text(json.dumps({"model": best_name, **m}, indent=2))
    if log:
        lines += ["", "INPUT ALIGNMENT LOG"] + [f"  - {l}" for l in log]

    report = "\n".join(lines)
    (OUT_DIR / "evaluation_metrics.txt").write_text(report, encoding="utf-8")
    print(report)


def run_predict(path):
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found — run `python pipeline.py` first.")
    model, config = joblib.load(MODEL_PATH), json.loads(CONFIG_PATH.read_text())
    df = load_csv(path)
    log = []
    preds, X = predict_frame(df, model, config, log)
    write_outputs(df, preds, X, df["label"] if "label" in df else None)
    for l in log:
        print("[align]", l)
    print(f"{len(preds)} rows scored, {int(preds['predicted_label'].sum())} faults -> {OUT_DIR / 'predictions.csv'}")
    if "label" in df:
        print(metrics_block(df["label"].astype(int), preds["predicted_label"], preds["fault_probability"])[0])


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="MachinaSense fault detection pipeline")
    ap.add_argument("--train", default=str(DEFAULT_DATA), help="labelled training CSV")
    ap.add_argument("--test", help="optional separate test CSV (labels optional)")
    ap.add_argument("--predict", help="score a CSV with the saved model (no training)")
    args = ap.parse_args()
    if args.predict:
        run_predict(args.predict)
    else:
        run_train(args.train, args.test)


if __name__ == "__main__":
    main()
