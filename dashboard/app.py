"""SensaFlow dashboard — 3D equipment twin (CNC spindle or SKAB pump rig) replaying the pipeline outputs.

    streamlit run dashboard/app.py                                  # interactive dashboard
    python dashboard/app.py --export twin.html [--dataset skab_valve1]   # standalone HTML (offline)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CONFIG_PATH = ROOT / "feature_config.json"

# Sidebar label -> output folder written by pipeline.py
DATASETS = {
    "AI4I": OUT,
    "SKAB Valve1": OUT / "skab_valve1",
    "SKAB Valve2": OUT / "skab_valve2",
    "SKAB Other": OUT / "skab_other",
    "Other (custom file)": OUT / "custom",
}
HOW_TO_GENERATE = {
    "AI4I": "python pipeline.py",
    "SKAB Valve1": "python pipeline.py --skab",
    "SKAB Valve2": "python pipeline.py --skab",
    "SKAB Other": "python pipeline.py --skab",
    "Other (custom file)": "python pipeline.py --predict path/to/file.csv",
}
# Replay columns per equipment profile: twin key -> scored_data.csv column
PROFILE_COLUMNS = {
    "ai4i": {"air": "air_temp", "proc": "process_temp", "delta": "temp_delta", "rpm": "rpm", "torque": "torque",
             "wear": "tool_wear", "power": "power", "strain": "strain"},
    "skab": {k: k for k in ["acc1", "acc2", "current", "voltage", "pressure", "flow", "temp", "thermo"]},
}
TYPE_CODE = re.compile(r"\((HDF|PWF|OSF|TWF|VIB|ELEC|FLOW|TEMP)\)")
DIRECTION = {"above": "▲ above normal", "below": "▼ below normal", "within": "in normal range"}
CATEGORY_COLORS = {"raw sensor": "#3B82F6", "physics-derived": "#F59E0B",
                   "VIB": "#3B82F6", "ELEC": "#A78BFA", "FLOW": "#06B6D4", "TEMP": "#F59E0B"}


def short_driver(text) -> str:
    """'rpm = 1,331 rpm (below normal 1,340 rpm–1,865 rpm)' -> 'rpm = 1,331 rpm ▼ below normal'."""
    if not isinstance(text, str) or not text:
        return ""
    return re.sub(r"\((above|below|within) normal [^)]*\)", lambda m: DIRECTION[m.group(1)], text)


def run_info(out_dir: Path) -> dict:
    path = out_dir / "run_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


def available(out_dir: Path) -> bool:
    return (out_dir / "scored_data.csv").exists()


def replay_order(ids: pd.Series) -> pd.Series:
    """Sort key for the replay: numeric ids, else parseable datetimes, else file order."""
    as_num = pd.to_numeric(ids, errors="coerce")
    if as_num.notna().all():
        return as_num
    as_time = pd.to_datetime(ids, errors="coerce")
    if as_time.notna().all():
        return as_time
    return pd.Series(np.arange(len(ids)), index=ids.index)


def load_replay(out_dir: Path) -> dict:
    df = pd.read_csv(out_dir / "scored_data.csv")
    df = df.iloc[np.argsort(replay_order(df["timestamp"]).values, kind="stable")].reset_index(drop=True)
    info = run_info(out_dir)
    profile = info.get("profile", "ai4i")
    id_col = info.get("id_column")
    id_label = "UDI" if (id_col or "").upper() == "UDI" else ("" if id_col else "ROW")

    def num(col, default=np.nan):
        return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(default, index=df.index)

    sensors = {key: num(col) for key, col in PROFILE_COLUMNS[profile].items()}
    rows = pd.DataFrame({
        "uid": df["timestamp"].astype(str),
        **sensors,
        "prob": num("fault_probability", 0.0),
        "type": df.get("likely_fault_type", pd.Series("", index=df.index)).fillna("").map(
            lambda s: (TYPE_CODE.search(s).group(1) if TYPE_CODE.search(s) else None)),
        "ev1": df.get("driver_1", pd.Series("", index=df.index)).map(short_driver),
        "ev2": df.get("driver_2", pd.Series("", index=df.index)).map(short_driver),
        "actual": num("actual_label"),
    })
    # Sensors may be absent in scored input; keep the twin drawable with column means
    for c in sensors:
        rows[c] = rows[c].fillna(rows[c].mean()).fillna(0.0).round(5)
    rows["prob"] = rows["prob"].round(4)

    f1 = "n/a (no labels)"
    if rows["actual"].notna().all():
        y, p = rows["actual"].astype(int), (num("predicted_label", 0) == 1).astype(int)
        tp, fp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        f1 = f"{2 * tp / max(2 * tp + fp + fn, 1):.3f}"

    model = info.get("model")
    if not model and CONFIG_PATH.exists():
        model = json.loads(CONFIG_PATH.read_text()).get("model")
    records = rows.astype(object).where(rows.notna(), None).values.tolist()
    return {
        "profile": profile,
        "columns": list(rows.columns),
        "rows": records,
        "meta": {
            "model": model or "unknown",
            "f1": f1,
            "idLabel": id_label,
            "source": f"REPLAY · {info.get('input', 'PIPELINE OUTPUT')} · {len(rows)} ROWS"
                      + (f" ORDERED BY {id_col}" if id_col else " IN FILE ORDER"),
        },
    }


def build_html(props: dict, out_dir: Path = OUT) -> str:
    data = load_replay(out_dir)
    data["props"] = props

    def safe(js: str) -> str:
        return js.replace("</script", "<\\/script")

    page = (HERE / "twin.html").read_text(encoding="utf-8")
    page = page.replace("/*__THREE__*/", safe((HERE / "vendor" / "three.min.js").read_text(encoding="utf-8")), 1)
    page = page.replace("/*__PREACT__*/", safe((HERE / "vendor" / "htm-preact-standalone.umd.js").read_text(encoding="utf-8")), 1)
    page = page.replace("/*__DATA__*/null", safe(json.dumps(data, ensure_ascii=False)), 1)
    return page


def default_props(out_dir: Path = OUT) -> dict:
    """Medium threshold = the model's decision threshold (AI4I: F1-tuned; SKAB: calibrated so 0.5 = threshold)."""
    thr = 0.5
    if run_info(out_dir).get("profile", "ai4i") == "ai4i" and CONFIG_PATH.exists():
        thr = json.loads(CONFIG_PATH.read_text()).get("threshold", 0.5)
    return {"autoRotate": True, "showCallouts": True, "wireframe": True,
            "medThreshold": round(min(max(thr, 0.2), 0.79), 2), "highThreshold": 0.8}


# ------------------------------------------------------------------ streamlit

def model_evidence(st, out_dir: Path):
    import plotly.graph_objects as go

    dark = dict(paper_bgcolor="#161B22", plot_bgcolor="#161B22", font=dict(color="#C9D1D9", family="JetBrains Mono, monospace"),
                margin=dict(l=10, r=10, t=40, b=10))
    info = run_info(out_dir)
    profile = info.get("profile", "ai4i")
    # SHAP/robustness: from this folder, else (custom AI4I-model runs) from the AI4I training run
    shap_path = next((p for p in (out_dir / "shap_importance.csv", OUT / "shap_importance.csv")
                      if p.exists() and (p.parent == out_dir or profile == "ai4i")), None)
    rob_path = next((p for p in (out_dir / "robustness.csv", OUT / "robustness.csv")
                     if p.exists() and (p.parent == out_dir or profile == "ai4i")), None)

    st.markdown("#### MODEL EVIDENCE")
    st.caption(f"Model: {info.get('model', 'unknown')} · scores measured on: {info.get('input', 'the last pipeline run')}"
               + (" · SHAP importance and robustness come from the AI4I training run."
                  if shap_path is not None and shap_path.parent != out_dir else ""))
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        c = st.columns(4)
        c[0].metric("F1 SCORE", f"{m['f1_fault']:.3f}", help=f"anomaly/fault class, measured on {info.get('input')}")
        c[1].metric("PRECISION", f"{m['precision']:.3f}")
        c[2].metric("RECALL", f"{m['recall']:.3f}")
        c[3].metric("AUC-PR", f"{m['aucpr']:.3f}")

        left, right = st.columns(2)
        cm = np.array(m["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred normal", "Pred fault"], y=["Actual normal", "Actual fault"],
                                   text=cm, texttemplate="%{text}", colorscale=[[0, "#0D1117"], [1, "#06B6D4"]],
                                   showscale=False))
        fig.update_layout(title="Confusion matrix", height=320, yaxis_autorange="reversed", **dark)
        left.plotly_chart(fig, width="stretch")
    else:
        st.info("The scored file has no failure labels, so accuracy metrics can't be computed. "
                "Predictions, alerts and evidence are still shown above.")
        left, right = st.columns(2)

    if shap_path is not None:
        imp = pd.read_csv(shap_path).head(10).iloc[::-1]
        colors = imp["category"].map(CATEGORY_COLORS).fillna("#06B6D4")
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h", marker_color=colors))
        fig.update_layout(title="What drives the predictions (mean |SHAP|)", height=320, **dark)
        right.plotly_chart(fig, width="stretch")
        st.caption("Blue = raw sensor · Amber = physics-derived feature" if profile == "ai4i" else
                   "Per sensor (level + 10 s trend + 10 s instability) · Blue = vibration · Violet = electrical · "
                   "Cyan = flow/pressure · Amber = thermal")

    report = out_dir / "evaluation_metrics.txt"
    if profile == "skab" and report.exists():
        with st.expander("Evaluation report (per experiment)", expanded=False):
            st.text(report.read_text(encoding="utf-8"))
    if rob_path is not None:
        with st.expander("Robustness report", expanded=False):
            rob = pd.read_csv(rob_path)
            rob = rob.rename(columns={"oracle_delta_f1": "rules_delta_f1"})
            st.dataframe(rob.style.format({"f1": "{:.3f}", "delta_f1": "{:+.3f}", "rules_delta_f1": "{:+.3f}",
                                           "fault_rate": "{:.2%}"}, na_rep=""),
                         width="stretch", hide_index=True)
            st.caption("rules_delta_f1 = how much the true AI4I failure rules lose on the same perturbed readings "
                       "(the physical floor). PASS* = within that floor.")
            st.text((rob_path.parent / "robustness_report.txt").read_text(encoding="utf-8").split("Notes:")[-1])


def run_streamlit():
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="SensaFlow", page_icon="⚙", layout="wide")
    st.markdown("<style>.block-container{padding-top:3.5rem;padding-bottom:1rem;max-width:100%}</style>",
                unsafe_allow_html=True)

    # ?dataset=<slug> selects a dataset, and the URL follows the sidebar so every view is linkable
    slugs = {n: ("ai4i" if p == OUT else p.name) for n, p in DATASETS.items()}
    names = list(DATASETS)
    if "dataset" not in st.session_state:      # first load only: honour the URL; afterwards the sidebar rules
        wanted = st.query_params.get("dataset")
        st.session_state["dataset"] = next((n for n in names if slugs[n] == wanted), names[0])
    with st.sidebar:
        st.markdown("### DATASET")
        # fixed key and no `index`: a changing default would make Streamlit recreate the widget and drop clicks
        name = st.radio("Dataset", names, key="dataset", label_visibility="collapsed",
                        format_func=lambda n: n if available(DATASETS[n]) else f"{n} (not generated)")
    if st.query_params.get("dataset") != slugs[name]:
        st.query_params["dataset"] = slugs[name]
    out_dir = DATASETS[name]
    if not available(out_dir):
        st.warning(f"No outputs for **{name}** yet. Generate them with:")
        st.code(HOW_TO_GENERATE[name], language="bash")
        st.stop()
    for warning in run_info(out_dir).get("warnings", []):
        st.warning(warning)

    d = default_props(out_dir)
    with st.sidebar:
        st.markdown("### SEVERITY")
        med = st.slider("Medium threshold", 0.2, 0.79, d["medThreshold"], 0.01, key=f"med_{name}",
                        help=f"Default = the model's decision threshold ({d['medThreshold']})")
        hi = st.slider("High threshold", 0.5, 0.99, d["highThreshold"], 0.01, key=f"hi_{name}")
        st.markdown("### 3D VIEW")
        auto = st.toggle("Auto-rotate", d["autoRotate"])
        callouts = st.toggle("Show callouts", d["showCallouts"])
        wire = st.toggle("Wireframe edges", d["wireframe"])
        st.markdown("### PANELS")
        evidence = st.toggle("Show model evidence", True)

    props = {"autoRotate": auto, "showCallouts": callouts, "wireframe": wire, "medThreshold": med, "highThreshold": hi}
    # components.html (not st.iframe): verified to start the WebGL view inside Streamlit 1.66
    components.html(build_html(props, out_dir), height=1100, scrolling=True)
    if evidence:
        model_evidence(st, out_dir)


def running_in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if running_in_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    import argparse

    folders = {p.name if p != OUT else "ai4i": p for p in DATASETS.values()}
    ap = argparse.ArgumentParser(description="Export the SensaFlow twin dashboard as standalone HTML")
    ap.add_argument("--export", default=str(OUT / "dashboard.html"))
    ap.add_argument("--dataset", default="ai4i", choices=list(folders), help="which output folder to replay")
    args = ap.parse_args()
    src = folders[args.dataset]
    Path(args.export).write_text(build_html(default_props(src), src), encoding="utf-8")
    print(f"Wrote {args.export}")
