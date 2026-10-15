"""SensaFlow AI analyst — explains the selected run in plain language.

Answers are grounded in the pipeline outputs of one dataset folder: a compact, pre-computed briefing (metrics,
alert episodes, failure modes, drivers, misses) plus exact readings for any UDI / timestamp the user mentions.
The language model is served by OpenRouter (free models); without a key or network, a built-in summary built
from the same briefing answers instead.

Key: OPENROUTER_API_KEY in the environment or in a git-ignored `.env` at the project root.
Model: OPENROUTER_MODEL (optional) overrides the default chain below.
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Tried in order; later entries are used when an earlier one is rate-limited or failing.
DEFAULT_MODELS = ["nvidia/nemotron-3-super-120b-a12b:free", "openrouter/free"]
MAX_HISTORY_TURNS = 8
TIMEOUT_S = 90

QUICK_QUESTIONS = {
    "Summary": "Give me a short summary of this run: how the model performs, what kinds of faults it "
                          "found, the most important alerts, and anything an operator should watch.",
    "Top alert": "Explain the highest-confidence alert in plain language: when it happened, what the "
                           "sensors showed, why the model flagged it, and whether it was a real fault.",
    "Drivers": "Which sensors and features drive the model's fault predictions most, and what does that "
                           "mean physically for this equipment?",
    "Reliability": "How reliable is this model on this data? Cover precision/recall in plain terms, false "
                           "alarms vs missed faults, and any robustness or weak spots.",
}

GLOSSARY = """\
- F1: balance of precision and recall for the fault class (1.0 = perfect). Precision: share of alerts that were real
  faults. Recall: share of real faults that were caught. AUC-PR: ranking quality across all thresholds.
- Severity: HIGH = fault probability >= 0.8; MEDIUM = at/above the model's decision threshold.
- SHAP: how much each feature pushed a prediction towards "fault". Drivers are the top SHAP features of an alert.
- AI4I failure modes: HDF heat dissipation (small process-air temperature difference at low speed), PWF power
  (mechanical power outside its band), OSF overstrain (torque x tool wear too high), TWF tool wear, RNF random.
- SKAB groups (pump/valve rig): VIB pump vibration (accelerometers), ELEC motor electrical (current/voltage),
  FLOW valve/flow (pressure/flow rate), TEMP thermal (motor/fluid temperature). "σ vs baseline" = deviation from the
  first 400 s of that run, which is assumed normal.
- AI4I rows are independent simulated snapshots replayed in row-id (UDI) order; SKAB rows are a real 1 Hz series."""

SYSTEM_PROMPT = """You are the SensaFlow analyst: you explain an industrial fault-detection dashboard to operators and
judges in plain, confident language.

Rules:
- Use only the DATA below and the conversation. Quote concrete numbers, timestamps/UDIs and sensor values from it.
- If the data doesn't contain the answer, say so briefly and suggest what to look at (e.g. a timestamp to ask about).
- You cannot see the live replay cursor. If asked "what is happening now", explain the run overall and ask for the
  timestamp or UDI shown on the dashboard clock.
- Explain technical terms the first time you use them. Be concise: short paragraphs or bullets, normally under 180
  words unless the user asks for detail. Use Markdown. Don't invent sensors, faults or causes beyond the data.

Glossary:
{glossary}

DATA ({dataset}):
{briefing}"""


# ------------------------------------------------------------------------------------------- credentials

def api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def models() -> list[str]:
    override = os.environ.get("OPENROUTER_MODEL")
    return [override] + [m for m in DEFAULT_MODELS if m != override] if override else DEFAULT_MODELS


# ------------------------------------------------------------------------------------------- briefing

SENSORS = {
    "ai4i": [("air_temp", "air temp", "K"), ("process_temp", "process temp", "K"), ("temp_delta", "temp delta", "K"),
             ("rpm", "speed", "rpm"), ("torque", "torque", "Nm"), ("power", "power", "W"),
             ("tool_wear", "tool wear", "min"), ("strain", "torque×wear", "Nm·min")],
    "skab": [("acc1", "Accelerometer1RMS", "g"), ("acc2", "Accelerometer2RMS", "g"), ("current", "Current", "A"),
             ("voltage", "Voltage", "V"), ("pressure", "Pressure", "bar"), ("flow", "Volume Flow RateRMS", "L/min"),
             ("temp", "Temperature", "°C"), ("thermo", "Thermocouple", "°C")],
}


def _read(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return path.read_text(encoding="utf-8")


def _order(ids: pd.Series) -> np.ndarray:
    num = pd.to_numeric(ids, errors="coerce")
    if num.notna().all():
        return np.argsort(num.values, kind="stable")
    t = pd.to_datetime(ids, errors="coerce")
    return np.argsort(t.values, kind="stable") if t.notna().all() else np.arange(len(ids))


def load_run(out_dir: Path) -> dict:
    """Everything the analyst needs from one output folder."""
    info = _read(out_dir / "run_info.json") or {}
    df = _read(out_dir / "scored_data.csv")
    df = df.iloc[_order(df["timestamp"])].reset_index(drop=True)
    shap = _read(out_dir / "shap_importance.csv")
    if shap is None and info.get("profile", "ai4i") == "ai4i":
        shap = _read(ROOT / "outputs" / "shap_importance.csv")
    rob = _read(out_dir / "robustness_report.txt")
    if rob is None and info.get("profile", "ai4i") == "ai4i" and info.get("mode") != "predict":
        rob = _read(ROOT / "outputs" / "robustness_report.txt")
    return {"info": info, "df": df, "metrics": _read(out_dir / "metrics.json"), "shap": shap,
            "report": _read(out_dir / "evaluation_metrics.txt"), "robustness": rob}


def episodes(df: pd.DataFrame, max_gap: int = 0) -> list[dict]:
    """Group alert rows into episodes: consecutive rows of the same experiment, allowing gaps of up to `max_gap`
    non-alert rows (SKAB alerts flicker second to second; a 10-row gap still reads as one event)."""
    flagged = np.flatnonzero(df["predicted_label"].values == 1)
    exp = df["experiment"].astype(str).values if "experiment" in df else np.zeros(len(df), dtype=str)
    groups, cur = [], []
    for i in flagged:
        if cur and (i - cur[-1] > max_gap + 1 or exp[i] != exp[cur[-1]]):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        seg = df.iloc[g[0]:g[-1] + 1]
        alerts = df.iloc[g]
        peak = alerts["fault_probability"].idxmax()
        types = (alerts["likely_fault_type"].fillna("").replace("", np.nan).dropna()
                 if "likely_fault_type" in alerts else pd.Series(dtype=str))
        drivers = "; ".join(x for x in (df.at[peak, "driver_1"], df.at[peak, "driver_2"])
                            if isinstance(x, str) and x) if "driver_1" in df else ""
        out.append({
            "start": str(df.at[g[0], "timestamp"]), "end": str(df.at[g[-1], "timestamp"]), "rows": len(seg),
            "peak": float(alerts["fault_probability"].max()),
            "severity": "HIGH" if alerts["fault_probability"].max() >= 0.8 else "MEDIUM",
            "type": types.mode().iat[0] if len(types) else "Unclassified",
            "drivers": drivers,
            "actual": seg["actual_label"].mean() if "actual_label" in seg else np.nan,
            "experiment": exp[g[0]] if "experiment" in df else None,
        })
    return out


def _rank(e: dict, profile: str) -> tuple:
    """SKAB is a time series, so the longest episodes are the main events; AI4I rows are snapshots: rank by confidence."""
    return (-e["rows"], -e["peak"]) if profile == "skab" else (-round(e["peak"], 2), -e["rows"])


def run_episodes(run: dict) -> list[dict]:
    return episodes(run["df"], max_gap=10 if run["info"].get("profile") == "skab" else 0)


def _fmt_episode(e: dict) -> str:
    when = e["start"] if e["rows"] == 1 else f"{e['start']} → {e['end']} ({e['rows']} rows)"
    exp = f" [run {e['experiment']}]" if e.get("experiment") not in (None, "") else ""
    truth = ""
    if not np.isnan(e["actual"]):
        truth = " · confirmed fault" if e["actual"] >= 0.5 else " · FALSE ALARM (no fault in labels)"
    return f"- {when}{exp}: {e['severity']} p={e['peak']:.2f}, {e['type']}{truth}. Drivers: {e['drivers'] or 'n/a'}"


def briefing(run: dict) -> str:
    """Compact text summary of the run (the analyst's grounding)."""
    info, df, m = run["info"], run["df"], run["metrics"]
    profile = info.get("profile", "ai4i")
    eps = run_episodes(run)
    lines = [
        f"Dataset: {info.get('input', '?')} · equipment: {'CNC milling spindle (AI4I 2020)' if profile == 'ai4i' else 'water-circulation pump/valve rig (SKAB)'}",
        f"Model: {info.get('model', '?')} ({'supervised, trained on AI4I' if profile == 'ai4i' else 'unsupervised IsolationForest, fitted per run on its first 400 s'})",
        f"Rows: {len(df)} · id/time from {df['timestamp'].iat[0]} to {df['timestamp'].iat[-1]} · labels available: {info.get('has_labels')}",
        f"Alert rows: {int(df['predicted_label'].sum())} ({df['predicted_label'].mean():.1%}) grouped into {len(eps)} alert episodes "
        f"({sum(e['severity'] == 'HIGH' for e in eps)} HIGH, {sum(e['severity'] == 'MEDIUM' for e in eps)} MEDIUM).",
    ]
    if m:
        cm = m["confusion_matrix"]
        lines += [f"Accuracy on these labels: F1 {m['f1_fault']:.3f}, precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
                  f"AUC-PR {m['aucpr']:.3f}. Confusion matrix: {cm[1][1]} faults caught, {cm[1][0]} missed, "
                  f"{cm[0][1]} false alarms, {cm[0][0]} correct normals."]
    for w in info.get("warnings", []):
        lines.append(f"Input warning: {w}")

    if eps:
        by_type = pd.Series([e["type"] for e in eps]).value_counts()
        lines.append("Alert episodes by likely failure mode: " + ", ".join(f"{k}: {v}" for k, v in by_type.items()))
        if not np.isnan(eps[0]["actual"]):
            fa = [e for e in eps if e["actual"] < 0.5]
            lines.append(f"False-alarm episodes (mostly normal rows per labels): {len(fa)} of {len(eps)}.")
        top = sorted(eps, key=lambda e: _rank(e, profile))[:10]
        title = ("Largest alert episodes (main events, longest first):" if profile == "skab"
                 else "Highest-confidence alert episodes:")
        lines += ["", title] + [_fmt_episode(e) for e in top]
        lines += ["", "First alert episodes in time order:"] + [_fmt_episode(e) for e in eps[:4]]
        lines += ["", "Last alert episodes in time order:"] + [_fmt_episode(e) for e in eps[-4:]]

    if "actual_label" in df:
        missed = df[(df["actual_label"] == 1) & (df["predicted_label"] == 0)]
        if len(missed):
            ex = missed.sort_values("fault_probability", ascending=False).head(5)
            lines += ["", f"Missed faults (labelled fault, no alert): {len(missed)} rows. Examples (closest calls):"]
            lines += [f"- {r.timestamp}: fault probability {r.fault_probability:.2f}" for r in ex.itertuples()]

    if run["shap"] is not None:
        s = run["shap"].head(6)
        lines += ["", "Global drivers (mean |SHAP|, higher = more influence): " +
                  ", ".join(f"{r.feature} {r.mean_abs_shap:.3f} ({r.category})" for r in s.itertuples())]
    if run["report"] and "PER EXPERIMENT" in run["report"]:
        per = run["report"].split("PER EXPERIMENT")[-1].strip("-\n ").splitlines()
        lines += ["", "Per-run F1 (SKAB experiments):"] + [f"- run {p.strip()}" for p in per if p.strip()]
    if run["report"] and "Tuning:" in run["report"] or (run["report"] and "Held-out group" in run["report"]):
        note = next((l for l in run["report"].splitlines() if l.startswith(("Tuning:", "Held-out group"))), "")
        lines.append(note)
    if run["robustness"]:
        body = run["robustness"].split("-" * 95)[-1].split("Notes:")[0].strip()
        lines += ["", "Robustness on the AI4I test split (ΔF1 vs clean; 'Rules ΔF1' = loss of the true failure "
                  "rules on the same perturbed data; PASS* = within that physical floor):", body]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- lookups

def lookup(run: dict, question: str, limit: int = 4) -> str:
    """Exact readings for any UDI / timestamp mentioned in the question."""
    df, profile = run["df"], run["info"].get("profile", "ai4i")
    ts = df["timestamp"].astype(str)
    hits = []
    for t in re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?", question):
        hits += list(df.index[ts.str.startswith(t.replace("T", " "))][:2])
    if not hits:
        for t in re.findall(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", question):
            hits += list(df.index[ts.str.contains(f" {t.zfill(5) if len(t) <= 5 else t}", regex=False)][:2])
    if not hits:
        # "UDI 4271" / "#4271", or a bare 3-6 digit id (so "top 3" isn't read as row 3)
        for n in re.findall(r"(?:udi\s*#?\s*|#)(\d{1,6})\b|\b(\d{3,6})\b", question.lower()):
            n = n[0] or n[1]
            hits += list(df.index[ts == n][:1])
    if not hits:
        return ""
    lines = []
    is_numeric_id = pd.to_numeric(ts, errors="coerce").notna().all()        # e.g. AI4I UDIs, not timestamps
    times = pd.Series(pd.NaT, index=ts.index) if is_numeric_id else pd.to_datetime(ts, errors="coerce", format="ISO8601")
    if times.notna().all():   # time series: also report alert episodes within ±60 s of the mentioned times
        near = []
        for e in run_episodes(run):
            s, t_end = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
            if any(s - pd.Timedelta(seconds=60) <= times[i] <= t_end + pd.Timedelta(seconds=60) for i in hits):
                near.append(e)
        if near:
            lines += ["", "Alert episodes within ±60 s of the mentioned time:"] + [_fmt_episode(e) for e in near[:4]]
    lines += ["", "Exact readings for the rows the user mentioned:"]
    for i in list(dict.fromkeys(hits))[:limit]:
        r = df.loc[i]
        vals = ", ".join(f"{label} {r[col]:.4g} {unit}" for col, label, unit in SENSORS[profile]
                         if col in df and pd.notna(r[col]))
        truth = "" if "actual_label" not in df else (" · labelled FAULT" if r["actual_label"] == 1 else " · labelled normal")
        drivers = "; ".join(x for x in (r.get("driver_1"), r.get("driver_2")) if isinstance(x, str) and x)
        lines.append(f"- {r['timestamp']}: fault probability {r['fault_probability']:.3f}, severity {r['alert_severity']}, "
                     f"alert={'yes' if r['predicted_label'] == 1 else 'no'}{truth}. "
                     f"{('Likely mode: ' + r['likely_fault_type'] + '. ') if isinstance(r.get('likely_fault_type'), str) and r['likely_fault_type'] else ''}"
                     f"{('Drivers: ' + drivers + '. ') if drivers else ''}Readings: {vals}.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- answering

class AnalystError(RuntimeError):
    pass


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _call(model: str, messages: list[dict], key: str) -> tuple[str, str]:
    body = {"model": model, "messages": messages, "max_tokens": 900, "temperature": 0.2}
    req = urllib.request.Request(
        OPENROUTER_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/narenadhitya/SensaFlow", "X-Title": "SensaFlow"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.load(resp)
    if "error" in data:
        raise AnalystError(str(data["error"].get("message", data["error"])))
    text = _clean(data["choices"][0]["message"].get("content"))
    if not text:
        raise AnalystError("empty response")
    return text, data.get("model", model)


def ask(run: dict, history: list[dict], question: str, dataset: str) -> tuple[str, str]:
    """Answer `question` about the run. Returns (markdown answer, model label)."""
    key = api_key()
    grounding = briefing(run) + lookup(run, question)
    if not key:
        return offline_answer(run, question), "built-in summary (no OPENROUTER_API_KEY)"
    system = SYSTEM_PROMPT.format(glossary=GLOSSARY, dataset=dataset, briefing=grounding)
    convo = [{"role": h["role"], "content": h["content"]} for h in history[-2 * MAX_HISTORY_TURNS:]]
    messages = [{"role": "system", "content": system}, *convo, {"role": "user", "content": question}]
    errors = []
    for model in models():
        try:
            text, served = _call(model, messages, key)
            return text, served
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:200]
            errors.append(f"{model}: HTTP {e.code} {detail}")
            if e.code in (401, 402, 403):        # bad key / no credit: other models won't help
                break
        except (urllib.error.URLError, TimeoutError, AnalystError, KeyError, ValueError) as e:
            errors.append(f"{model}: {e}")
    note = errors[-1] if errors else "unknown error"
    return (offline_answer(run, question) +
            f"\n\n_AI model unavailable ({note[:160]}); showing the built-in summary._"), "built-in summary (fallback)"


def offline_answer(run: dict, question: str) -> str:
    """Deterministic summary from the same data — used without a key or when the API fails."""
    info, df, m = run["info"], run["df"], run["metrics"]
    eps = run_episodes(run)
    parts = [f"**{info.get('input', 'This run')}** · model **{info.get('model', '?')}** · {len(df)} rows, "
             f"{len(eps)} alert episodes ({sum(e['severity'] == 'HIGH' for e in eps)} HIGH)."]
    if m:
        cm = m["confusion_matrix"]
        parts.append(f"- F1 **{m['f1_fault']:.3f}** · precision {m['precision']:.3f} (share of alerts that were real) · "
                     f"recall {m['recall']:.3f} (share of faults caught)\n- {cm[1][1]} faults caught, {cm[1][0]} missed, "
                     f"{cm[0][1]} false alarms")
    if eps:
        by_type = pd.Series([e["type"] for e in eps]).value_counts()
        parts.append("- Failure modes: " + ", ".join(f"{k} ×{v}" for k, v in by_type.items()))
        profile = info.get("profile", "ai4i")
        top = min(eps, key=lambda e: _rank(e, profile))
        parts.append(("**Main event**\n" if profile == "skab" else "**Top alert**\n") + _fmt_episode(top))
    if run["shap"] is not None:
        parts.append("- Main drivers: " + ", ".join(run["shap"]["feature"].head(4)))
    detail = lookup(run, question)
    if detail:
        parts.append(detail.strip())
    return "\n\n".join(parts)
