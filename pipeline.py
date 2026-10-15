"""SensaFlow — industrial fault detection pipeline.

Usage:
    python pipeline.py                                   # train on AI4I, evaluate on held-out 20% split -> outputs/
    python pipeline.py --train data/train.csv --test data/test.csv
    python pipeline.py --skab                            # SKAB valve1/valve2/other with IsolationForest -> outputs/skab_*/
    python pipeline.py --predict data/unseen.csv         # score a new file -> outputs/custom/
                                                         # (AI4I-style sensors: saved model; SKAB-style: IsolationForest)
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
CUSTOM_DIR = OUT_DIR / "custom"          # --predict results (dashboard: "Other")
CONFIG_PATH = ROOT / "feature_config.json"
MODEL_PATH = MODEL_DIR / "model.pkl"
DEFAULT_DATA = ROOT / "datasets" / "ai4i2020.csv"
SEED = 42

RAW_SENSORS = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
ABSENT_SENSOR_FILL = "native"   # how to treat a sensor missing entirely at inference: "mean" or "native"
DASHBOARD_EXPLAIN_FLOOR = 0.2  # explain rows down to this probability so the dashboard threshold slider has evidence
DROP_FEATURES: list[str] = ["air_temp", "process_temp", "temp_rpm_ratio"]  # absolute temps are shift-sensitive; deltas carry the physics

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
    # Fallback for sensors only: a column whose name contains an alias (e.g. "spindle_rpm", "torque_nm")
    for canon in RAW_SENSORS:
        if canon in rename.values():
            continue
        for norm, orig in lookup.items():
            if orig not in rename and any(alias in norm for alias in ALIASES[canon]):
                rename[orig] = canon
                break
    df = df.rename(columns=rename)
    df.attrs["id_source"] = next((orig for orig, canon in rename.items() if canon == "id"), None)
    return df


def sensor_coverage(df: pd.DataFrame, log: list) -> list[str]:
    """Warn when the input shares few sensors with the model — predictions would mostly reflect imputation."""
    found = [c for c in RAW_SENSORS if c in df]
    missing = [c for c in RAW_SENSORS if c not in df]
    if len(missing) >= 2:
        msg = (f"WARNING: only {len(found)}/{len(RAW_SENSORS)} model sensors found ({', '.join(found) or 'none'}); "
               f"missing {', '.join(missing)}. Predictions will be unreliable — retrain on this data instead.")
        log.append(msg)
        print(msg)
    return found


def write_run_info(mode: str, path, df: pd.DataFrame, has_labels: bool, sensors: list[str], warnings: list[str],
                   out_dir: Path = OUT_DIR, profile: str = "ai4i", model: str | None = None):
    """Describe what an output folder contains (read by the dashboard)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"mode": mode, "profile": profile, "model": model, "input": Path(path).name, "rows": int(len(df)),
            "id_column": df.attrs.get("id_source"), "has_labels": bool(has_labels), "sensors_found": sensors,
            "warnings": warnings}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))


# ----------------------------------------------------------- feature building

def build_features(df: pd.DataFrame, config: dict | None = None, log: list | None = None) -> pd.DataFrame:
    """Raw sensors + Type + physics-derived features.

    Scattered NaNs are left in place: the model is trained on NaN-augmented data and routes
    them natively. With a config (inference), a sensor that is entirely absent is handled per
    config["absent_sensor_fill"] ("mean" = training mean, "native" = leave NaN), and the output
    is aligned to the training feature order. Extra columns are ignored.
    """
    log = log if log is not None else []
    X = pd.DataFrame(index=df.index)
    for col in RAW_SENSORS:
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    X["type"] = df["type"].map(TYPE_MAP) if "type" in df else np.nan

    if config is not None:
        fill = config.get("absent_sensor_fill", "mean")
        for col in RAW_SENSORS + ["type"]:
            n_missing = int(X[col].isna().sum())
            if n_missing == len(X):
                if fill == "mean":
                    X[col] = config["train_means"][col]
                    log.append(f"sensor '{col}' absent from input -> imputed with training mean {config['train_means'][col]:.3f}")
                else:
                    log.append(f"sensor '{col}' absent from input -> handled as missing by the model")
            elif n_missing:
                log.append(f"sensor '{col}': {n_missing} missing values -> handled natively by the model")

    # Layer 2 — physics features
    X["temp_delta"] = X["process_temp"] - X["air_temp"]                 # heat dissipation
    X["power"] = X["torque"] * X["rpm"] * 2 * np.pi / 60                # mechanical power [W]
    X["temp_rpm_ratio"] = X["process_temp"] / (X["rpm"] + 1)            # thermal stress
    X["wear_rate"] = X["tool_wear"] / (X["rpm"] + 1)                    # tool stress rate
    # Layer 4 — interactions
    X["strain"] = X["torque"] * X["tool_wear"]                          # overstrain indicator
    X["temp_delta_rpm"] = X["temp_delta"] * X["rpm"] / 1000               # HDF interaction (shift-invariant)
    X = X.drop(columns=[c for c in DROP_FEATURES if c in X])

    if config is not None:
        X = X[config["features"]]
    return X


def augment_missing(df: pd.DataFrame, y, fractions=(0.10, 0.20)) -> tuple[pd.DataFrame, pd.Series]:
    """Append copies of the training rows with random sensor dropouts so the model learns to handle NaNs."""
    rng = np.random.default_rng(SEED)
    parts, labels = [df], [y]
    for frac in fractions:
        d = df.copy()
        for c in RAW_SENSORS:
            if c in d:
                d.loc[rng.random(len(d)) < frac, c] = np.nan
        parts.append(d)
        labels.append(y)
    return pd.concat(parts, ignore_index=True), pd.concat(labels, ignore_index=True)


def feature_stats(X: pd.DataFrame) -> tuple[dict, dict]:
    return X.mean().to_dict(), X.std().replace(0, 1).to_dict()


def normal_ranges(X: pd.DataFrame, y) -> dict:
    """5th–95th percentile of each feature on healthy (label 0) training rows."""
    healthy = X[np.asarray(y) == 0]
    return {c: [float(healthy[c].quantile(0.05)), float(healthy[c].quantile(0.95))] for c in X.columns}


# ------------------------------------------------------------ explainability

FEATURE_CATEGORY = {
    **{c: "raw sensor" for c in RAW_SENSORS + ["type"]},
    **{c: "physics-derived" for c in ["temp_delta", "power", "temp_rpm_ratio", "wear_rate", "strain", "temp_delta_rpm"]},
}

# Which AI4I failure mode a feature points to (heuristic, for operator context). rpm and torque feed
# several modes, so they are left out and the mode with the largest summed SHAP push wins.
FAULT_HINT = {
    "temp_delta": "Heat Dissipation Failure (HDF)", "temp_rpm_ratio": "Heat Dissipation Failure (HDF)",
    "temp_delta_rpm": "Heat Dissipation Failure (HDF)",
    "air_temp": "Heat Dissipation Failure (HDF)", "process_temp": "Heat Dissipation Failure (HDF)",
    "power": "Power Failure (PWF)",
    "strain": "Overstrain Failure (OSF)",
    "tool_wear": "Tool Wear Failure (TWF)", "wear_rate": "Tool Wear Failure (TWF)",
}

UNITS = {"air_temp": " K", "process_temp": " K", "rpm": " rpm", "torque": " Nm", "tool_wear": " min",
         "temp_delta": " K", "power": " W", "strain": " Nm·min"}


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP contributions towards the fault class, shape (n_rows, n_features)."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):          # older shap: one array per class
        sv = sv[1]
    if sv.ndim == 3:                  # newer shap on sklearn: (n, f, classes)
        sv = sv[:, :, 1]
    return sv


def fmt_value(feature: str, value: float) -> str:
    num = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"
    return f"{num}{UNITS.get(feature, '')}"


def explain_alerts(X: pd.DataFrame, sv: np.ndarray, ranges: dict, top_k: int = 2) -> list[dict]:
    """For each row: the top features pushing towards 'fault', with their value and healthy range."""
    out = []
    for i in range(len(X)):
        order = np.argsort(-sv[i])[:top_k]
        drivers = []
        for j in order:
            f = X.columns[j]
            if sv[i, j] <= 0:
                continue
            v, (lo, hi) = X.iat[i, j], ranges[f]
            direction = "above" if v > hi else "below" if v < lo else "within"
            drivers.append({"feature": f, "value": float(v), "shap": float(sv[i, j]),
                            "text": f"{f} = {fmt_value(f, v)} ({direction} normal {fmt_value(f, lo)}–{fmt_value(f, hi)})"})
        push: dict[str, float] = {}
        for j, f in enumerate(X.columns):
            if f in FAULT_HINT and sv[i, j] > 0:
                push[FAULT_HINT[f]] = push.get(FAULT_HINT[f], 0.0) + sv[i, j]
        hint = max(push, key=push.get) if push else "Unclassified"
        out.append({"drivers": drivers, "fault_type": hint,
                    "summary": "; ".join(d["text"] for d in drivers)})
    return out


def save_global_explanations(model, X: pd.DataFrame, n_sample: int = 500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    Xs = X.sample(min(n_sample, len(X)), random_state=SEED)
    sv = shap_values(model, Xs)
    imp = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["category"] = imp["feature"].map(FEATURE_CATEGORY).fillna("other")
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(OUT_DIR / "shap_importance.csv", index=False)

    shap.summary_plot(sv, Xs, show=False, max_display=12)
    plt.title("What drives fault predictions (SHAP)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------- models

def perturb(df: pd.DataFrame, nan: float = 0.0, noise: float = 0.0, temp_shift: float = 0.0,
            seed: int = SEED) -> pd.DataFrame:
    """Random NaNs (fraction per sensor), Gaussian noise (σ in sensor-std units) and/or a temperature offset [K]."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    sensors = [c for c in RAW_SENSORS if c in d]
    for c in sensors:
        if noise:
            d[c] = d[c] + rng.normal(0, noise * d[c].std(), len(d))
        if nan:
            d.loc[rng.random(len(d)) < nan, c] = np.nan
    for c in ("air_temp", "process_temp"):
        if temp_shift and c in d:
            d[c] = d[c] + temp_shift
    return d


# Validation scenarios used for model selection — rewards models that stay accurate off the clean path
SELECTION_SCENARIOS = {"clean": {}, "missing10": {"nan": 0.10}, "noise0.1": {"noise": 0.10}, "temp+2K": {"temp_shift": 2.0}}


def train_candidates(X_tr, y_tr, df_val, y_val) -> dict:
    X_val = build_features(df_val)
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
        scenario_f1 = {}
        for sname, kw in SELECTION_SCENARIOS.items():
            p = model.predict_proba(build_features(perturb(df_val, **kw, seed=7)))[:, 1]
            scenario_f1[sname] = f1_score(y_val, (p >= thr).astype(int))
        results[name] = {"model": model, "threshold": thr, "val_f1": f1,
                         "val_aucpr": average_precision_score(y_val, proba),
                         "scenario_f1": scenario_f1,
                         "selection_score": float(np.mean(list(scenario_f1.values())))}
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


def predict_frame(df: pd.DataFrame, model, config: dict, log: list, explain_floor: float | None = None):
    """Score rows. Returns (predictions, features, alerts, detail).

    SHAP explanations are computed for predicted faults and, if explain_floor is given, for every row
    with fault_probability >= explain_floor (used by the dashboard, whose alert threshold is adjustable).
    """
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
    out["evidence_summary"] = ""

    # SHAP evidence for every predicted fault (+ lower-probability rows for the dashboard)
    floor = config["threshold"] if explain_floor is None else min(explain_floor, config["threshold"])
    detail = pd.DataFrame({"likely_fault_type": "", "driver_1": "", "driver_2": ""}, index=out.index)
    alerts = pd.DataFrame()
    exp_idx = np.flatnonzero(proba >= floor)
    expl_by_row = {}
    if len(exp_idx):
        Xe = X.iloc[exp_idx]
        for i, e in zip(exp_idx, explain_alerts(Xe, shap_values(model, Xe), config["normal_ranges"])):
            expl_by_row[i] = e
            d = [x["text"] for x in e["drivers"]] + ["", ""]
            detail.iloc[i] = [e["fault_type"], d[0], d[1]]
    idx = np.flatnonzero(pred)
    if len(idx):
        expl = [expl_by_row[i] for i in idx]
        out.loc[out.index[idx], "evidence_summary"] = [e["summary"] for e in expl]
        rows = []
        for i, e in zip(idx, expl):
            d = e["drivers"] + [{}] * 2
            rows.append({
                "timestamp": out.at[out.index[i], "timestamp"],
                "alert_severity": out.at[out.index[i], "alert_severity"],
                "confidence": round(float(proba[i]), 4),
                "likely_fault_type": e["fault_type"],
                "primary_driver": d[0].get("feature", ""),
                "primary_value": d[0].get("value"),
                "primary_shap": d[0].get("shap"),
                "secondary_driver": d[1].get("feature", ""),
                "secondary_value": d[1].get("value"),
                "secondary_shap": d[1].get("shap"),
                "evidence": e["summary"],
            })
        sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        alerts = (pd.DataFrame(rows)
                  .sort_values(["alert_severity", "confidence"], key=lambda s: s.map(sev_rank) if s.name == "alert_severity" else -s)
                  .reset_index(drop=True))
        alerts.insert(0, "alert_id", np.arange(1, len(alerts) + 1))
    return out, X, alerts, detail


def write_outputs(df, preds, X, alerts, detail, label=None, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    alerts.to_csv(out_dir / "fault_alerts_evidence.csv", index=False)
    # Richer table for the dashboard: sensors + derived features + predictions (+ ground truth)
    raw = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
                        for c in RAW_SENSORS if c not in X}, index=df.index)
    scored = pd.concat([preds.reset_index(drop=True), raw.reset_index(drop=True), X.reset_index(drop=True),
                        detail.reset_index(drop=True)], axis=1)
    if label is not None:
        scored["actual_label"] = np.asarray(label)
    scored.to_csv(out_dir / "scored_data.csv", index=False)


# ----------------------------------------------------------------- robustness

def run_robustness(model, config: dict, df_te: pd.DataFrame, y_te) -> tuple[list[str], list[dict]]:
    """Re-score the test split under perturbed inputs and compare against the clean baseline."""
    y = np.asarray(y_te).astype(int)
    sensors = [c for c in RAW_SENSORS if c in df_te]

    def score(df):
        preds = predict_frame(df, model, config, [])[0]
        p = preds["predicted_label"].values
        return f1_score(y, p), p.mean()

    base_f1, base_rate = score(df_te)

    def scaled(**factors):
        d = df_te.copy()
        for c, k in factors.items():
            d[c] = d[c] * k
        return d

    scenarios = [
        ("Missing values 10% (random NaN per sensor)", perturb(df_te, nan=0.10), 0.10),
        ("Missing values 20%", perturb(df_te, nan=0.20), None),
        ("Gaussian noise σ=0.1·std", perturb(df_te, noise=0.10), 0.10),
        ("Gaussian noise σ=0.25·std", perturb(df_te, noise=0.25), None),
        ("Condition shift: +2 K ambient & process temp", perturb(df_te, temp_shift=2.0), 0.10),
        ("Condition shift: temperatures ×1.1 (plan spec)", scaled(air_temp=1.1, process_temp=1.1), None),
        ("Condition shift: rpm +5%", scaled(rpm=1.05), None),
    ] + [(f"Sensor dropout: '{c}' removed", df_te.drop(columns=c), None) for c in sensors]

    results = [{"scenario": "Clean baseline", "f1": base_f1, "delta_f1": 0.0, "fault_rate": base_rate, "status": "-"}]
    for name, d, tol in scenarios:
        f1, rate = score(d)
        status = "INFO" if tol is None else ("PASS" if base_f1 - f1 <= tol else "FAIL")
        results.append({"scenario": name, "f1": f1, "delta_f1": f1 - base_f1, "fault_rate": rate, "status": status})

    lines = [
        "SENSAFLOW — ROBUSTNESS REPORT", "=" * 34,
        f"Evaluated on the held-out test split (n={len(y)}), model={config['model']}, threshold={config['threshold']:.3f}.",
        "Model trained on NaN-augmented data (10%/20% sensor dropout copies) and routes missing values natively.",
        f"Entirely absent sensors: '{config.get('absent_sensor_fill', 'mean')}' fill (see feature_config.json).",
        "PASS criteria: ΔF1 ≥ -0.10 for missing-10%, noise σ=0.1·std and +2 K shift. Others are informational stress tests.",
        "",
        f"{'Scenario':<50} {'F1':>6} {'ΔF1':>7} {'Fault rate':>11}  Status",
        "-" * 84,
    ]
    for r in results:
        lines.append(f"{r['scenario']:<50} {r['f1']:>6.3f} {r['delta_f1']:>+7.3f} {r['fault_rate']:>10.2%}  {r['status']}")
    drop = [r for r in results if r["scenario"].startswith("Sensor dropout")]
    worst = min(drop, key=lambda r: r["f1"])
    lines += ["", "Notes:",
              f"  - Most critical sensor (largest F1 loss when removed): {worst['scenario'].split(chr(39))[1]} (F1 {worst['f1']:.3f}).",
              "  - The pipeline never crashes on missing/renamed/extra columns; it logs and degrades gracefully.",
              "  - AI4I failure labels follow sharp physical thresholds (e.g. HDF: temp delta < 8.6 K and rpm < 1380),",
              "    so noise near a boundary legitimately flips the true state — some sensitivity is inherent.",
              "  - Multiplicative ×1.1 on Kelvin temperatures is a ~30 K shift — far outside physical operating range;",
              "    the +2 K scenario is the realistic 'warmer conditions' test."]
    return lines, results


# ------------------------------------------------------------------------ SKAB
# SKAB (Skoltech Anomaly Benchmark): water-circulation testbed, 1 Hz, 8 sensors, ';'-separated, real datetimes,
# an `anomaly` label per row. Different sensor domain from AI4I, so it uses its own unsupervised method:
# each experiment file is its own stream; an IsolationForest is fitted on that stream's first 400 rows (normal
# start-up, the standard SKAB protocol) and scores the rest. Labels are used only for evaluation.

SKAB_DIR = ROOT / "datasets" / "SKAB"
SKAB_GROUPS = ["valve1", "valve2", "other"]
SKAB_SENSORS = {   # canonical key -> SKAB column
    "acc1": "Accelerometer1RMS", "acc2": "Accelerometer2RMS", "current": "Current", "pressure": "Pressure",
    "temp": "Temperature", "thermo": "Thermocouple", "voltage": "Voltage", "flow": "Volume Flow RateRMS",
}
SKAB_UNITS = {"acc1": " g", "acc2": " g", "current": " A", "pressure": " bar", "temp": " °C", "thermo": " °C",
              "voltage": " V", "flow": " L/min"}
SKAB_GROUP_OF = {"acc1": "VIB", "acc2": "VIB", "current": "ELEC", "voltage": "ELEC",
                 "pressure": "FLOW", "flow": "FLOW", "temp": "TEMP", "thermo": "TEMP"}
SKAB_MODE_NAME = {"VIB": "Pump vibration anomaly (VIB)", "ELEC": "Motor electrical anomaly (ELEC)",
                  "FLOW": "Valve / flow anomaly (FLOW)", "TEMP": "Thermal anomaly (TEMP)"}
# Chosen on valve1 only (window 0/10/30 s x percentile 95/99); valve2 and other are held out.
SKAB_CONFIG = {"baseline_rows": 400, "window": 10, "threshold_percentile": 95, "n_estimators": 200}


def skab_columns(columns) -> dict:
    """Map a file's columns to SKAB canonical keys (case/units-insensitive)."""
    wanted = {normalise(v): k for k, v in SKAB_SENSORS.items()}
    return {c: wanted[normalise(c)] for c in columns if normalise(c) in wanted}


def is_skab_format(path) -> bool:
    head = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", nrows=5)
    return len(skab_columns(head.columns)) >= 4


def load_skab_file(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df = df.rename(columns=skab_columns(df.columns))
    time_col = next((c for c in df.columns if normalise(c) in ("datetime", "timestamp", "time")), None)
    if time_col:
        df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values("datetime", kind="stable")
    label_col = next((c for c in df.columns if normalise(c) in ("anomaly", "label", "fault")), None)
    df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int) if label_col else np.nan
    return df.reset_index(drop=True)


def skab_features(df: pd.DataFrame, sensors: list[str], n_base: int, window: int) -> pd.DataFrame:
    """Sensor levels standardised against the stream's own baseline + rolling mean/std (trend, instability)."""
    X = df[sensors].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    base = X.iloc[:n_base]
    Z = (X - base.mean()) / base.std().replace(0, 1).fillna(1)
    return pd.concat([Z,
                      Z.rolling(window, min_periods=1).mean().add_suffix("__mean"),
                      Z.rolling(window, min_periods=2).std().fillna(0).add_suffix("__std")], axis=1)


def score_skab_stream(df: pd.DataFrame, cfg: dict = SKAB_CONFIG) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Fit on the stream's baseline, score every row, explain rows near/above threshold. Returns (scored, shap, F)."""
    sensors = [k for k in SKAB_SENSORS if k in df]
    n_base = min(cfg["baseline_rows"], max(len(df) // 3, 30))
    F = skab_features(df, sensors, n_base, cfg["window"])
    model = IsolationForest(n_estimators=cfg["n_estimators"], random_state=SEED, n_jobs=-1).fit(F.iloc[:n_base])
    score = -model.score_samples(F)                         # higher = more anomalous
    base_score = score[:n_base]
    thr = float(np.percentile(base_score, cfg["threshold_percentile"]))
    scale = float(base_score.std()) or 1e-6
    prob = 1 / (1 + np.exp(-(score - thr) / scale))          # 0.5 at the decision threshold
    pred = (score > thr).astype(int)

    out = pd.DataFrame({
        "timestamp": df["datetime"].astype(str) if "datetime" in df else np.arange(len(df)),
        "predicted_label": pred, "fault_probability": prob.round(4),
        "alert_severity": severity(prob, 0.5), "evidence_summary": "",
        "anomaly_score": score.round(4),
    })
    for k in sensors:
        out[k] = pd.to_numeric(df[k], errors="coerce").values
    out["likely_fault_type"], out["driver_1"], out["driver_2"] = "", "", ""

    sv_all = np.zeros(F.shape)
    idx = np.flatnonzero(prob >= DASHBOARD_EXPLAIN_FLOOR)
    if len(idx):
        import shap
        sv = shap.TreeExplainer(model).shap_values(F.iloc[idx])
        sv_all[idx] = sv
        push = pd.DataFrame(-sv, columns=F.columns)          # positive = pushes towards anomaly
        per_sensor = pd.DataFrame({k: push[[k, f"{k}__mean", f"{k}__std"]].sum(axis=1) for k in sensors})
        for row, i in enumerate(idx):
            ranked = per_sensor.iloc[row].sort_values(ascending=False)
            drivers = []
            for k in ranked.index[:2]:
                if ranked[k] <= 0:
                    continue
                parts = push.iloc[row][[k, f"{k}__mean", f"{k}__std"]]
                z = F.iat[i, F.columns.get_loc(k)]
                tag = " · unstable" if parts.idxmax().endswith("__std") else ""
                drivers.append(f"{SKAB_SENSORS[k]} = {out.at[i, k]:.4g}{SKAB_UNITS[k]} ({z:+.1f}σ vs baseline{tag})")
            groups = per_sensor.iloc[row].clip(lower=0).groupby(SKAB_GROUP_OF).sum()
            out.at[i, "likely_fault_type"] = SKAB_MODE_NAME[groups.idxmax()] if groups.max() > 0 else "Unclassified"
            out.at[i, "driver_1"], out.at[i, "driver_2"] = (drivers + ["", ""])[:2]
        flagged = pred[idx] == 1
        out.loc[idx[flagged], "evidence_summary"] = [
            "; ".join(x for x in (out.at[i, "driver_1"], out.at[i, "driver_2"]) if x) for i in idx[flagged]]
    if df["label"].notna().all():
        out["actual_label"] = df["label"].astype(int).values
    return out, sv_all, F


def run_skab_files(files: list[Path], out_dir: Path, name: str, note: str = ""):
    """Score one or more SKAB-format streams and write the standard output set to out_dir."""
    frames, shap_rows, shap_cols = [], [], None
    per_file = []
    for f in files:
        df = load_skab_file(f)
        scored, sv, F = score_skab_stream(df)
        scored.insert(1, "experiment", f.stem)
        frames.append(scored)
        explained = np.flatnonzero(np.abs(sv).sum(axis=1) > 0)
        if len(explained):
            rng = np.random.default_rng(SEED)
            pick = rng.choice(explained, size=min(150, len(explained)), replace=False)
            shap_rows.append(pd.DataFrame(sv[pick], columns=F.columns))
        if "actual_label" in scored:
            y, p = scored["actual_label"], scored["predicted_label"]
            per_file.append((f.stem, len(scored), f1_score(y, p, zero_division=0), y.mean()))
    allx = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("robustness.csv", "robustness_report.txt", "metrics.json"):
        (out_dir / stale).unlink(missing_ok=True)

    pred_cols = ["timestamp", "predicted_label", "fault_probability", "alert_severity", "evidence_summary"]
    allx[pred_cols].to_csv(out_dir / "predictions.csv", index=False)
    allx.to_csv(out_dir / "scored_data.csv", index=False)
    flagged = allx[allx["predicted_label"] == 1]
    alerts = flagged[["timestamp", "experiment", "alert_severity", "fault_probability", "likely_fault_type",
                      "driver_1", "driver_2", "evidence_summary"]].rename(
        columns={"fault_probability": "confidence", "evidence_summary": "evidence"})
    alerts = alerts.sort_values("confidence", ascending=False).reset_index(drop=True)
    alerts.insert(0, "alert_id", np.arange(1, len(alerts) + 1))
    alerts.to_csv(out_dir / "fault_alerts_evidence.csv", index=False)

    if shap_rows:   # global importance, aggregated per sensor
        sv = pd.concat(shap_rows, ignore_index=True).abs().mean()
        imp = pd.DataFrame([{"feature": SKAB_SENSORS[k],
                             "mean_abs_shap": float(sum(sv.get(c, 0) for c in (k, f"{k}__mean", f"{k}__std"))),
                             "category": SKAB_GROUP_OF[k]} for k in SKAB_SENSORS if k in sv.index])
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(out_dir / "shap_importance.csv", index=False)

    cfg = SKAB_CONFIG
    lines = ["SENSAFLOW — SKAB EVALUATION REPORT", "=" * 34, f"Data: {name} ({len(files)} file(s), {len(allx)} rows at 1 Hz)",
             f"Method: IsolationForest ({cfg['n_estimators']} trees) fitted per stream on its first "
             f"{cfg['baseline_rows']} rows (normal start-up);", f"        features = baseline-standardised sensors + "
             f"{cfg['window']} s rolling mean/std; threshold = {cfg['threshold_percentile']}th percentile of baseline scores.",
             "Labels are never used for fitting — only for the evaluation below."]
    if note:
        lines.append(note)
    has_labels = "actual_label" in allx and allx["actual_label"].notna().all()
    if has_labels:
        text, m = metrics_block(allx["actual_label"].astype(int), allx["predicted_label"], allx["fault_probability"])
        (out_dir / "metrics.json").write_text(json.dumps({"model": "IsolationForest", **m}, indent=2))
        lines += ["", "POINT-WISE RESULTS (all rows)", "-" * 34, text, "PER EXPERIMENT", "-" * 34]
        lines += [f"{n:<12} rows={r:<5} anomaly rate={a:5.1%}  F1={f:.3f}" for n, r, f, a in per_file]
        print(f"{name}: F1={m['f1_fault']:.3f}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"({len(files)} file{'s' if len(files) != 1 else ''}) -> {out_dir}")
    else:
        print(f"{name}: {len(allx)} rows scored, {int(allx['predicted_label'].sum())} anomalous -> {out_dir}")
    (out_dir / "evaluation_metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    allx.attrs["id_source"] = "datetime"
    write_run_info("skab", name, allx, has_labels, [SKAB_SENSORS[k] for k in SKAB_SENSORS if k in allx],
                   [], out_dir=out_dir, profile="skab", model="IsolationForest")


def run_skab(groups=SKAB_GROUPS):
    def key(p):
        return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)
    for g in groups:
        files = sorted((SKAB_DIR / g).glob("*.csv"), key=key)
        if not files:
            print(f"SKAB group '{g}' not found under {SKAB_DIR}")
            continue
        note = "Tuning: window/percentile chosen on valve1 — treat valve1 as in-sample." if g == "valve1" else \
               "Held-out group: no settings were tuned on it."
        run_skab_files(files, OUT_DIR / f"skab_{g}", f"SKAB {g}", note)


# ---------------------------------------------------------------------- modes

def run_train(train_path, test_path=None):
    df = load_csv(train_path)
    if "label" not in df:
        raise SystemExit(f"No label column found in {train_path}")
    warnings: list[str] = []
    sensors = sensor_coverage(df, warnings)
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
    raw_means = {c: float(pd.to_numeric(df_tr[c], errors="coerce").mean()) for c in RAW_SENSORS}
    raw_means["type"] = float(df_tr["type"].map(TYPE_MAP).mean()) if "type" in df_tr else 0.0
    feat_means, feat_stds = feature_stats(X_tr)

    df_aug, y_aug = augment_missing(df_tr, y_tr.reset_index(drop=True))
    results = train_candidates(build_features(df_aug), y_aug, df_val, y_val)
    best_name = max(results, key=lambda k: results[k]["selection_score"])
    best = results[best_name]

    config = {
        "model": best_name,
        "features": list(X_tr.columns),
        "raw_sensors": RAW_SENSORS,
        "type_map": TYPE_MAP,
        "absent_sensor_fill": ABSENT_SENSOR_FILL,
        "threshold": best["threshold"],
        "train_means": raw_means,
        "feature_means": feat_means,
        "feature_stds": feat_stds,
        "normal_ranges": normal_ranges(X_tr, y_tr),
        "n_train": len(X_tr), "n_val": len(X_val),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    log = []
    preds, X_te, alerts, detail = predict_frame(df_te, best["model"], config, log, explain_floor=DASHBOARD_EXPLAIN_FLOOR)
    write_outputs(df_te, preds, X_te, alerts, detail, y_te)
    df_te.attrs["id_source"] = df_te.attrs.get("id_source") or df.attrs.get("id_source")
    write_run_info("train" if not test_path else "train+test",
                   test_path or f"{Path(train_path).name} (held-out 20% test split)", df_te, y_te is not None, sensors,
                   warnings, model=best_name)
    save_global_explanations(best["model"], X_te)

    lines = [
        "SENSAFLOW — EVALUATION REPORT", "=" * 34,
        f"Data: {split_desc}",
        f"Samples: train={len(X_tr)} (+{len(df_aug) - len(X_tr)} NaN-augmented copies)  val={len(X_val)}  test={len(df_te)}",
        "", "MODEL SELECTION (validation split only)", "-" * 34,
        "Threshold = F1-optimal cut-off on clean validation data.",
        f"Selection score = mean validation F1 over scenarios: {', '.join(SELECTION_SCENARIOS)}.",
        "",
    ]
    for name, r in results.items():
        per = "  ".join(f"{k}={v:.3f}" for k, v in r["scenario_f1"].items())
        lines.append(f"{name:<13} score={r['selection_score']:.3f}  AUC-PR={r['val_aucpr']:.3f}  thr={r['threshold']:.3f}  | {per}")
    lines += ["", f"Selected: {best_name}  (decision threshold {best['threshold']:.3f})"]
    if best_name == "XGBoost":
        lines.append(f"XGBoost best iteration: {best['model'].best_iteration}")

    if y_te is not None:
        text, m = metrics_block(y_te, preds["predicted_label"], preds["fault_probability"])
        lines += ["", "RESULTS ON TEST SET", "-" * 34, text]
        (OUT_DIR / "metrics.json").write_text(json.dumps({"model": best_name, **m}, indent=2))

        rob_lines, rob = run_robustness(best["model"], config, df_te, y_te)
        (OUT_DIR / "robustness_report.txt").write_text("\n".join(rob_lines), encoding="utf-8")
        pd.DataFrame(rob).to_csv(OUT_DIR / "robustness.csv", index=False)
        lines += ["ROBUSTNESS SUMMARY (see robustness_report.txt)", "-" * 34]
        lines += [f"{r['scenario']:<50} F1 {r['f1']:.3f} (Δ {r['delta_f1']:+.3f}) {r['status']}" for r in rob[1:8]]
    if log:
        lines += ["", "INPUT ALIGNMENT LOG"] + [f"  - {l}" for l in log]

    report = "\n".join(lines)
    (OUT_DIR / "evaluation_metrics.txt").write_text(report, encoding="utf-8")
    print(report)


def run_predict(path):
    """Score a new file into outputs/custom/. SKAB-style sensor files use the IsolationForest method."""
    out_dir = CUSTOM_DIR
    for stale in out_dir.glob("*") if out_dir.exists() else []:
        stale.unlink()
    if is_skab_format(path):
        print("Detected SKAB-style sensors -> unsupervised IsolationForest (per-file baseline).")
        return run_skab_files([Path(path)], out_dir, Path(path).name)
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found — run `python pipeline.py` first.")
    model, config = joblib.load(MODEL_PATH), json.loads(CONFIG_PATH.read_text())
    df = load_csv(path)
    log = []
    warnings: list[str] = []
    sensors = sensor_coverage(df, warnings)
    preds, X, alerts, detail = predict_frame(df, model, config, log, explain_floor=DASHBOARD_EXPLAIN_FLOOR)
    has_labels = "label" in df
    write_outputs(df, preds, X, alerts, detail, df["label"] if has_labels else None, out_dir=out_dir)
    write_run_info("predict", path, df, has_labels, sensors, warnings, out_dir=out_dir, model=config["model"])
    for l in log:
        print("[align]", l)
    print(f"{len(preds)} rows scored, {int(preds['predicted_label'].sum())} faults -> {out_dir / 'predictions.csv'}")
    metrics_path = out_dir / "metrics.json"
    if has_labels:     # metrics now describe this file, not the training test split
        text, m = metrics_block(df["label"].astype(int), preds["predicted_label"], preds["fault_probability"])
        metrics_path.write_text(json.dumps({"model": config["model"], **m}, indent=2))
        print(text)
    elif metrics_path.exists():
        metrics_path.unlink()


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="SensaFlow fault detection pipeline")
    ap.add_argument("--train", default=str(DEFAULT_DATA), help="labelled training CSV")
    ap.add_argument("--test", help="optional separate test CSV (labels optional)")
    ap.add_argument("--predict", help="score a new CSV (no training) -> outputs/custom/")
    ap.add_argument("--skab", action="store_true", help="run the SKAB benchmark (valve1, valve2, other) -> outputs/skab_*/")
    args = ap.parse_args()
    if args.predict:
        run_predict(args.predict)
    elif args.skab:
        run_skab()
    else:
        run_train(args.train, args.test)


if __name__ == "__main__":
    main()
