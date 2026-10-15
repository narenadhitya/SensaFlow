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


def build_html(props: dict, out_dir: Path = OUT, embedded: bool = False) -> str:
    """Self-contained twin page. `embedded` drops the twin's own brand header (the Streamlit page has one)."""
    data = load_replay(out_dir)
    data["props"] = props
    data["embedded"] = embedded

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

APP_CSS = """<style>
.block-container{padding:2.75rem 1.5rem 1.5rem;max-width:100%}
header[data-testid="stHeader"]{background:transparent}
section[data-testid='stSidebar']{background:#0E131A;border-right:1px solid rgba(255,255,255,0.05)}
section[data-testid='stSidebar'][data-collapsed='false']{width:250px !important;min-width:250px}
/* right-hand analyst column: stays in view while the page scrolls */
div[data-testid="stHorizontalBlock"]:has(.sf-analyst){align-items:flex-start}
div[data-testid="stColumn"]:has(.sf-analyst){position:sticky;top:3rem}
.sf-brand{font-size:1.05rem;font-weight:700;letter-spacing:.14em;color:#E6EDF3;margin:.25rem 0 1.25rem}
.sf-brand span{color:#22D3EE}
.sf-label{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#7D8896;margin:.2rem 0 .4rem}
.sf-run{font-size:.78rem;color:#7D8896;line-height:1.5;margin-top:.75rem}
.sf-status{font-size:.75rem;color:#7D8896}
.sf-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:middle}
.sf-empty{color:#7D8896;font-size:.85rem;line-height:1.55;padding:.25rem .1rem}
.sf-empty b{color:#C9D1D9;font-weight:600}
div[data-testid="stMetricValue"]{font-size:1.9rem}
div[data-testid="stChatMessage"]{padding:.6rem .4rem;gap:.6rem}
div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li{font-size:.88rem;line-height:1.55}
div[data-testid="stChatMessage"] h1, div[data-testid="stChatMessage"] h2, div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4{font-size:.95rem;margin:.4rem 0 .2rem}
</style>"""

PLOT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(color="#C9D1D9", family="IBM Plex Sans, Segoe UI, sans-serif", size=12),
                   margin=dict(l=8, r=8, t=36, b=8))


def model_evidence(st, out_dir: Path):
    import plotly.graph_objects as go

    info = run_info(out_dir)
    profile = info.get("profile", "ai4i")
    # SHAP/robustness: from this folder, else (custom AI4I-model runs) from the AI4I training run
    shap_path = next((p for p in (out_dir / "shap_importance.csv", OUT / "shap_importance.csv")
                      if p.exists() and (p.parent == out_dir or profile == "ai4i")), None)
    rob_path = next((p for p in (out_dir / "robustness.csv", OUT / "robustness.csv")
                     if p.exists() and (p.parent == out_dir or profile == "ai4i")), None)

    st.markdown("#### Model evidence")
    st.caption(f"{info.get('model', 'unknown')} · scores measured on {info.get('input', 'the last pipeline run')}"
               + (" · SHAP importance and robustness from the AI4I training run"
                  if shap_path is not None and shap_path.parent != out_dir else ""))
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        with st.container(border=True):
            c = st.columns(4)
            c[0].metric("F1 score", f"{m['f1_fault']:.3f}", help=f"fault/anomaly class, measured on {info.get('input')}")
            c[1].metric("Precision", f"{m['precision']:.3f}", help="share of alerts that were real faults")
            c[2].metric("Recall", f"{m['recall']:.3f}", help="share of real faults that were caught")
            c[3].metric("AUC-PR", f"{m['aucpr']:.3f}", help="ranking quality across all thresholds")
    else:
        st.info("The scored file has no failure labels, so accuracy metrics can't be computed. "
                "Predictions, alerts and evidence are still shown above.")

    left, right = st.columns(2, gap="medium")
    if metrics_path.exists():
        cm = np.array(m["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted normal", "Predicted fault"], y=["Actual normal", "Actual fault"],
                                   text=cm, texttemplate="%{text}", textfont=dict(size=15),
                                   colorscale=[[0, "#121821"], [1, "#0E7490"]], showscale=False, xgap=4, ygap=4))
        fig.update_layout(title="Confusion matrix", height=320, yaxis_autorange="reversed", **PLOT_LAYOUT)
        with left.container(border=True):
            st.plotly_chart(fig, width="stretch")
    if shap_path is not None:
        imp = pd.read_csv(shap_path).head(10).iloc[::-1]
        colors = imp["category"].map(CATEGORY_COLORS).fillna("#06B6D4")
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h", marker_color=colors,
                               marker_cornerradius=4))
        fig.update_layout(title="What drives the predictions (mean |SHAP|)", height=320, **PLOT_LAYOUT)
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        with right.container(border=True):
            st.plotly_chart(fig, width="stretch")
            st.caption("Blue = raw sensor · amber = physics-derived feature" if profile == "ai4i" else
                       "Per sensor (level + 10 s trend + 10 s instability) · blue vibration · violet electrical · "
                       "cyan flow/pressure · amber thermal")

    report = out_dir / "evaluation_metrics.txt"
    if profile == "skab" and report.exists():
        with st.expander("Evaluation report (per experiment)"):
            st.text(report.read_text(encoding="utf-8"))
    if rob_path is not None:
        with st.expander("Robustness report"):
            rob = pd.read_csv(rob_path).rename(columns={"oracle_delta_f1": "rules_delta_f1"})
            st.dataframe(rob.style.format({"f1": "{:.3f}", "delta_f1": "{:+.3f}", "rules_delta_f1": "{:+.3f}",
                                           "fault_rate": "{:.2%}"}, na_rep=""),
                         width="stretch", hide_index=True)
            st.caption("rules_delta_f1 = how much the true AI4I failure rules lose on the same perturbed readings "
                       "(the physical floor). PASS* = within that floor.")
            st.text((rob_path.parent / "robustness_report.txt").read_text(encoding="utf-8").split("Notes:")[-1])


def display_settings(st, name: str, d: dict) -> tuple[dict, bool]:
    """Thresholds and 3D options, tucked into a popover above the twin."""
    with st.popover("⚙ Display", width="content"):
        st.markdown('<div class="sf-label">Severity</div>', unsafe_allow_html=True)
        med = st.slider("Medium threshold", 0.2, 0.79, d["medThreshold"], 0.01, key=f"med_{name}",
                        help=f"Default = the model's decision threshold ({d['medThreshold']})")
        hi = st.slider("High threshold", 0.5, 0.99, d["highThreshold"], 0.01, key=f"hi_{name}")
        st.markdown('<div class="sf-label">3D view</div>', unsafe_allow_html=True)
        auto = st.toggle("Auto-rotate", d["autoRotate"], key="auto_rotate")
        callouts = st.toggle("Show callouts", d["showCallouts"], key="callouts")
        wire = st.toggle("Wireframe edges", d["wireframe"], key="wireframe")
        evidence = st.toggle("Show model evidence", True, key="evidence")
    return {"autoRotate": auto, "showCallouts": callouts, "wireframe": wire, "medThreshold": med, "highThreshold": hi}, evidence


def run_streamlit():
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="SensaFlow", page_icon="⚙", layout="wide", initial_sidebar_state="expanded")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # Left sidebar: datasets only. ?dataset=<slug> selects one, and the URL follows the sidebar.
    slugs = {n: ("ai4i" if p == OUT else p.name) for n, p in DATASETS.items()}
    names = list(DATASETS)
    if "dataset" not in st.session_state:      # first load only: honour the URL; afterwards the sidebar rules
        wanted = st.query_params.get("dataset")
        st.session_state["dataset"] = next((n for n in names if slugs[n] == wanted), names[0])
    with st.sidebar:
        st.markdown('<div class="sf-brand">SENSA<span>FLOW</span></div><div class="sf-label">Dataset</div>',
                    unsafe_allow_html=True)
        # fixed key and no `index`: a changing default would make Streamlit recreate the widget and drop clicks
        name = st.radio("Dataset", names, key="dataset", label_visibility="collapsed",
                        format_func=lambda n: n if available(DATASETS[n]) else f"{n} (not generated)")
        info = run_info(DATASETS[name])
        if info:
            st.markdown(f'<div class="sf-run">{info.get("input", "")}<br>{info.get("rows", "?")} rows · '
                        f'{info.get("model", "")}</div>', unsafe_allow_html=True)
    if st.query_params.get("dataset") != slugs[name]:
        st.query_params["dataset"] = slugs[name]
    out_dir = DATASETS[name]

    main, side = st.columns([3, 1.12], gap="medium")
    with main:
        if not available(out_dir):
            st.warning(f"No outputs for **{name}** yet. Generate them with:")
            st.code(HOW_TO_GENERATE[name], language="bash")
            st.stop()
        for warning in info.get("warnings", []):
            st.warning(warning)
        props, evidence = display_settings(st, name, default_props(out_dir))
        # components.html (not st.iframe): verified to start the WebGL view inside Streamlit 1.66
        # measured content height at ~1000-1200 px wide: CNC twin ~1070 px, SKAB rig (8 sensors) ~1210 px
        height = 1230 if info.get("profile") == "skab" else 1090
        components.html(build_html(props, out_dir, embedded=True), height=height, scrolling=True)
        if evidence:
            model_evidence(st, out_dir)
    with side:   # filled last, so answering a question never delays or reloads the twin
        analyst_panel(st, name, slugs[name], out_dir)


def analyst_panel(st, name: str, slug: str, out_dir: Path):
    """Right-hand chat: plain-language answers about the selected run (dashboard/assistant.py)."""
    import assistant

    online = assistant.api_key() is not None
    st.markdown('<div class="sf-analyst"></div>', unsafe_allow_html=True)
    st.markdown("#### AI analyst")
    st.markdown(f'<div class="sf-status"><span class="sf-dot" style="background:{"#10B981" if online else "#F59E0B"}">'
                f'</span>{"Online · free model via OpenRouter" if online else "Offline · built-in summaries (no API key)"}'
                f'</div>', unsafe_allow_html=True)

    stamp = (out_dir / "scored_data.csv").stat().st_mtime
    cache = st.session_state.setdefault("analyst_runs", {})
    if cache.get(slug, (None,))[0] != stamp:
        cache[slug] = (stamp, assistant.load_run(out_dir))
    run = cache[slug][1]
    history = st.session_state.setdefault("analyst_chats", {}).setdefault(slug, [])

    question = None
    cols = st.columns(2, gap="small")
    for i, (label, prompt) in enumerate(assistant.QUICK_QUESTIONS.items()):
        if cols[i % 2].button(label, key=f"q_{slug}_{i}", width="stretch"):
            question = prompt
    log = st.container(height=560, border=True)
    typed = st.chat_input("Ask about this run…", key=f"chat_{slug}")
    question = typed or question

    with log:
        if not (history or question):
            st.markdown(f'<div class="sf-empty">Ask anything about <b>{name}</b>: a summary, why an alert fired, '
                        'which sensors matter, how reliable the model is.<br><br>The analyst can\'t see the live '
                        'replay. To ask about a moment, quote the <b>time or UDI</b> on the dashboard clock, e.g. '
                        '<i>"What happened at 16:44?"</i></div>', unsafe_allow_html=True)
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("model"):
                    st.caption(msg["model"])
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Analysing this run…"):
                    answer, model = assistant.ask(run, history, question, name)
                st.markdown(answer)
                st.caption(model)
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer, "model": model}]
    if history and st.button("Clear chat", key=f"clear_{slug}", type="tertiary"):
        history.clear()
        st.rerun()


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
