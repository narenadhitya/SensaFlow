# MachinaSense — Industrial Fault Detection & Predictive Maintenance

Detects machine failures from industrial sensor data (AI4I 2020: air/process temperature, spindle speed,
torque, tool wear), explains every alert, and replays results on a 3D thermal twin of a CNC spindle.

## Setup (Python 3.10+)

```bash
python -m venv venv
venv\Scripts\activate            # Windows   |   source venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt
```

## Run the pipeline

```bash
python pipeline.py                                        # train + evaluate on AI4I (held-out 20% split), ~10 s
python pipeline.py --train data/train.csv --test data/test.csv   # train on one file, score another
python pipeline.py --predict data/unseen.csv              # score new data with the saved model
```

Input columns are matched by name (`Air temperature [K]`, `air_temp`, `Ambient_Temp`, …); `,` or `;` separators,
extra columns are ignored and missing sensors are handled without crashing.

## View the dashboard

```bash
streamlit run dashboard/app.py          # opens http://localhost:8501
```

No Python handy? Open `outputs/dashboard.html` directly in a browser (standalone, works offline).
Regenerate it after a new pipeline run with `python dashboard/app.py --export outputs/dashboard.html`.

Controls: play/pause and 1×/2×/4× replay, **NEXT FAULT** to jump to the next alert, drag the timeline to scrub,
click an alert to inspect it, drag the 3D model to orbit / scroll to zoom. The sidebar sets severity thresholds
and 3D options.

## Approach

- **Features:** raw sensors + physics-derived features (temperature delta, mechanical power, torque × wear strain,
  temp delta × rpm). Absolute temperatures are dropped so the model is invariant to uniform temperature shifts.
- **Model:** XGBoost vs Random Forest, both class-weighted for the 3.4% fault rate and trained on NaN-augmented
  copies (10%/20% sensor dropout) so missing readings are handled natively. The winner is chosen on mean validation
  F1 across clean / missing / noisy / +2 K scenarios; the decision threshold is tuned for F1 on validation only.
- **Explainability:** SHAP drivers per alert with healthy ranges, plus a likely failure mode (HDF/PWF/OSF/TWF).

## Results (held-out 20% stratified test split, n = 2000)

| F1 (fault) | Precision | Recall | AUC-PR |
|-----------:|----------:|-------:|-------:|
| 0.842      | 0.862     | 0.824  | 0.870  |

Robustness (missing values, noise, condition shifts, per-sensor dropout): `outputs/robustness_report.txt`.

## Output files

| File | Contents |
|---|---|
| `outputs/predictions.csv` | label, fault probability, severity, evidence per row |
| `outputs/fault_alerts_evidence.csv` | per-alert SHAP drivers and likely failure mode |
| `outputs/evaluation_metrics.txt` | model selection, test metrics, confusion matrix |
| `outputs/shap_summary.png` | global feature importance |
| `outputs/robustness_report.txt` | robustness test results |
| `models/model.pkl`, `feature_config.json` | trained model + feature/threshold config for inference |

Data: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0); SKAB (Skoltech Anomaly Benchmark) included as a fallback.
