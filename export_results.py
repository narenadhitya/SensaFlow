"""Collect every evaluation result into one JSON file.

    python export_results.py                   # -> outputs/evaluation_results.json

Reads only files the pipeline already wrote (run `python pipeline.py` / `--skab` / `--predict` first).
"""
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SKAB_GROUPS = {"valve1": "in-sample (window/percentile tuned here)", "valve2": "held out", "other": "held out"}


def _clean(x):
    """Round floats and turn NaN into null, recursively."""
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 4)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _metrics(path: Path) -> dict | None:
    m = _read_json(path)
    if not m:
        return None
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return {"f1_fault": m["f1_fault"], "f1_macro": m["f1_macro"], "precision": m["precision"], "recall": m["recall"],
            "auc_pr": m["aucpr"],
            "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp}}


def _model_selection(report: str) -> dict:
    out = {"criterion": "mean validation F1 over clean / 10% missing / noise 0.1·std / +2 K shift; "
                        "threshold = F1-optimal on clean validation", "candidates": {}}
    for line in report.splitlines():
        m = re.match(r"(\w+)\s+score=([\d.]+)\s+AUC-PR=([\d.]+)\s+thr=([\d.]+)\s+\|\s+(.*)", line)
        if m:
            scen = dict(kv.split("=") for kv in m.group(5).split())
            out["candidates"][m.group(1)] = {
                "selection_score": float(m.group(2)), "val_auc_pr": float(m.group(3)), "threshold": float(m.group(4)),
                "val_f1_by_scenario": {k: float(v) for k, v in scen.items()}}
        sel = re.match(r"Selected: (\w+)\s+\(decision threshold ([\d.]+)\)", line)
        if sel:
            out["selected"], out["decision_threshold"] = sel.group(1), float(sel.group(2))
    return out


def _per_experiment(report: str) -> list[dict]:
    rows = []
    for line in report.split("PER EXPERIMENT")[-1].splitlines():
        m = re.match(r"(\S+)\s+rows=(\d+)\s+anomaly rate=([\d.]+)%\s+F1=([\d.]+)", line.strip())
        if m:
            rows.append({"experiment": m.group(1), "rows": int(m.group(2)),
                         "anomaly_rate": float(m.group(3)) / 100, "f1": float(m.group(4))})
    return rows


def _shap(path: Path, top: int = 10) -> list[dict] | None:
    if not path.exists():
        return None
    return pd.read_csv(path).head(top).to_dict(orient="records")


def collect() -> dict:
    info = _read_json(OUT / "run_info.json") or {}
    config = _read_json(ROOT / "feature_config.json") or {}
    report = (OUT / "evaluation_metrics.txt").read_text(encoding="utf-8")
    rob = pd.read_csv(OUT / "robustness.csv").rename(columns={"oracle_delta_f1": "rules_delta_f1"})

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ai4i": {
            "dataset": "AI4I 2020 Predictive Maintenance (10,000 rows, 3.39% failures)",
            "split": "stratified 64% train / 16% validation / 20% test (seed 42)",
            "test_rows": info.get("rows"),
            "model": config.get("model"),
            "decision_threshold": config.get("threshold"),
            "features": config.get("features"),
            "missing_values": config.get("missing_values"),
            "test_metrics": _metrics(OUT / "metrics.json"),
            "model_selection": _model_selection(report),
            "robustness": {
                "note": "random scenarios averaged over 5 seeds; rules_delta_f1 = loss of the published AI4I failure "
                        "rules on the same perturbed readings (physical floor); PASS* = within that floor",
                "pass_criterion": "delta_f1 >= -0.10 for missing 10%, noise 0.1·std and +2 K shift",
                "scenarios": rob.to_dict(orient="records"),
            },
            "shap_top_features": _shap(OUT / "shap_importance.csv"),
        },
        "skab": {
            "dataset": "SKAB - Skoltech Anomaly Benchmark (water-circulation testbed, 1 Hz, 8 sensors)",
            "method": "IsolationForest per experiment, fitted on its first 400 s; baseline-standardised sensors + "
                      "10 s rolling mean/std; threshold = 95th percentile of baseline scores; labels used only for "
                      "evaluation",
            "groups": {},
        },
    }
    for group, note in SKAB_GROUPS.items():
        d = OUT / f"skab_{group}"
        if not (d / "metrics.json").exists():
            continue
        rep = (d / "evaluation_metrics.txt").read_text(encoding="utf-8")
        result["skab"]["groups"][group] = {
            "evaluation": note,
            "rows": (_read_json(d / "run_info.json") or {}).get("rows"),
            "metrics": _metrics(d / "metrics.json"),
            "per_experiment": _per_experiment(rep),
            "shap_top_sensors": _shap(d / "shap_importance.csv", top=8),
        }

    custom = OUT / "custom"
    cinfo = _read_json(custom / "run_info.json")
    if cinfo:
        result["custom_file"] = {"input": cinfo.get("input"), "model": cinfo.get("model"), "rows": cinfo.get("rows"),
                                 "metrics": _metrics(custom / "metrics.json"),
                                 "note": "last file scored with `pipeline.py --predict`"}
    return _clean(result)


if __name__ == "__main__":
    path = OUT / "evaluation_results.json"
    path.write_text(json.dumps(collect(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")
