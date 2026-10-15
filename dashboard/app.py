"""MachinaSense dashboard — CNC spindle thermal twin driven by the pipeline outputs.

    streamlit run dashboard/app.py                 # interactive dashboard
    python dashboard/app.py --export twin.html     # standalone HTML (open in any browser, works offline)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CONFIG_PATH = ROOT / "feature_config.json"

TYPE_CODE = re.compile(r"\((HDF|PWF|OSF|TWF)\)")
DIRECTION = {"above": "▲ above normal", "below": "▼ below normal", "within": "in normal range"}


def short_driver(text) -> str:
    """'rpm = 1,331 rpm (below normal 1,340 rpm–1,865 rpm)' -> 'rpm = 1,331 rpm ▼ below normal'."""
    if not isinstance(text, str) or not text:
        return ""
    return re.sub(r"\((above|below|within) normal [^)]*\)", lambda m: DIRECTION[m.group(1)], text)


def load_replay() -> dict:
    df = pd.read_csv(OUT / "scored_data.csv")
    df = df.sort_values("timestamp", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

    def num(col, default=np.nan):
        return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(default, index=df.index)

    rows = pd.DataFrame({
        "uid": df["timestamp"].astype(str),
        "air": num("air_temp"), "proc": num("process_temp"), "delta": num("temp_delta"),
        "rpm": num("rpm"), "torque": num("torque"), "wear": num("tool_wear"),
        "power": num("power"), "strain": num("strain"), "prob": num("fault_probability", 0.0),
        "type": df.get("likely_fault_type", pd.Series("", index=df.index)).fillna("").map(
            lambda s: (TYPE_CODE.search(s).group(1) if TYPE_CODE.search(s) else None)),
        "ev1": df.get("driver_1", pd.Series("", index=df.index)).map(short_driver),
        "ev2": df.get("driver_2", pd.Series("", index=df.index)).map(short_driver),
        "actual": num("actual_label"),
    })
    # Sensors may be absent in scored input; keep the twin drawable with column means
    for c in ["air", "proc", "delta", "rpm", "torque", "wear", "power", "strain"]:
        rows[c] = rows[c].fillna(rows[c].mean()).fillna(0.0).round(4)
    rows["prob"] = rows["prob"].round(4)

    f1 = "n/a (no labels)"
    if rows["actual"].notna().all():
        y, p = rows["actual"].astype(int), (num("predicted_label", 0) == 1).astype(int)
        tp, fp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        f1 = f"{2 * tp / max(2 * tp + fp + fn, 1):.3f}"

    records = rows.astype(object).where(rows.notna(), None).values.tolist()
    return {
        "columns": list(rows.columns),
        "rows": records,
        "meta": {
            "model": config.get("model", "unknown"),
            "f1": f1,
            "threshold": config.get("threshold", 0.5),
            "source": f"REPLAY · PIPELINE OUTPUT · {len(rows)} ROWS ORDERED BY UDI",
        },
    }


def build_html(props: dict) -> str:
    data = load_replay()
    data["props"] = props

    def safe(js: str) -> str:
        return js.replace("</script", "<\\/script")

    page = (HERE / "twin.html").read_text(encoding="utf-8")
    page = page.replace("/*__THREE__*/", safe((HERE / "vendor" / "three.min.js").read_text(encoding="utf-8")), 1)
    page = page.replace("/*__PREACT__*/", safe((HERE / "vendor" / "htm-preact-standalone.umd.js").read_text(encoding="utf-8")), 1)
    page = page.replace("/*__DATA__*/null", safe(json.dumps(data, ensure_ascii=False)), 1)
    return page


def default_props() -> dict:
    thr = json.loads(CONFIG_PATH.read_text()).get("threshold", 0.5) if CONFIG_PATH.exists() else 0.5
    return {"autoRotate": True, "showCallouts": True, "wireframe": True,
            "medThreshold": round(min(max(thr, 0.2), 0.79), 2), "highThreshold": 0.8}


# ------------------------------------------------------------------ streamlit

def model_evidence(st):
    import plotly.graph_objects as go

    dark = dict(paper_bgcolor="#161B22", plot_bgcolor="#161B22", font=dict(color="#C9D1D9", family="JetBrains Mono, monospace"),
                margin=dict(l=10, r=10, t=40, b=10))
    st.markdown("#### MODEL EVIDENCE")
    metrics_path = OUT / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        c = st.columns(4)
        c[0].metric("F1 SCORE (fault)", f"{m['f1_fault']:.3f}", help="held-out 20% test split")
        c[1].metric("PRECISION", f"{m['precision']:.3f}")
        c[2].metric("RECALL", f"{m['recall']:.3f}")
        c[3].metric("AUC-PR", f"{m['aucpr']:.3f}")

        left, right = st.columns(2)
        cm = np.array(m["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred normal", "Pred fault"], y=["Actual normal", "Actual fault"],
                                   text=cm, texttemplate="%{text}", colorscale=[[0, "#0D1117"], [1, "#06B6D4"]],
                                   showscale=False))
        fig.update_layout(title="Confusion matrix (test split)", height=320, yaxis_autorange="reversed", **dark)
        left.plotly_chart(fig, width="stretch")
    else:
        left, right = st.columns(2)

    shap_path = OUT / "shap_importance.csv"
    if shap_path.exists():
        imp = pd.read_csv(shap_path).head(10).iloc[::-1]
        colors = imp["category"].map({"raw sensor": "#3B82F6", "physics-derived": "#F59E0B"}).fillna("#06B6D4")
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h", marker_color=colors))
        fig.update_layout(title="What drives fault predictions (mean |SHAP|)", height=320, **dark)
        right.plotly_chart(fig, width="stretch")
        st.caption("Blue = raw sensor · Amber = physics-derived feature")

    rob_path = OUT / "robustness.csv"
    if rob_path.exists():
        with st.expander("Robustness report", expanded=False):
            rob = pd.read_csv(rob_path)
            st.dataframe(rob.style.format({"f1": "{:.3f}", "delta_f1": "{:+.3f}", "fault_rate": "{:.2%}"}),
                         width="stretch", hide_index=True)
            st.text((OUT / "robustness_report.txt").read_text(encoding="utf-8").split("Notes:")[-1])


def run_streamlit():
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="MachinaSense", page_icon="⚙", layout="wide")
    st.markdown("<style>.block-container{padding-top:3.5rem;padding-bottom:1rem;max-width:100%}</style>",
                unsafe_allow_html=True)
    if not (OUT / "scored_data.csv").exists():
        st.error("No pipeline outputs found. Run `python pipeline.py` first.")
        st.stop()

    d = default_props()
    with st.sidebar:
        st.markdown("### SEVERITY")
        med = st.slider("Medium threshold", 0.2, 0.79, d["medThreshold"], 0.01,
                        help=f"Default = model decision threshold ({d['medThreshold']}) tuned on validation F1")
        hi = st.slider("High threshold", 0.5, 0.99, d["highThreshold"], 0.01)
        st.markdown("### 3D VIEW")
        auto = st.toggle("Auto-rotate", d["autoRotate"])
        callouts = st.toggle("Show callouts", d["showCallouts"])
        wire = st.toggle("Wireframe edges", d["wireframe"])
        st.markdown("### PANELS")
        evidence = st.toggle("Show model evidence", True)

    props = {"autoRotate": auto, "showCallouts": callouts, "wireframe": wire, "medThreshold": med, "highThreshold": hi}
    # components.html (not st.iframe): verified to start the WebGL view inside Streamlit 1.66
    components.html(build_html(props), height=1060, scrolling=True)
    if evidence:
        model_evidence(st)


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

    ap = argparse.ArgumentParser(description="Export the MachinaSense twin dashboard as standalone HTML")
    ap.add_argument("--export", default=str(OUT / "dashboard.html"))
    args = ap.parse_args()
    Path(args.export).write_text(build_html(default_props()), encoding="utf-8")
    print(f"Wrote {args.export}")
