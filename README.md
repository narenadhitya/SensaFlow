# SensaFlow — Industrial Fault Detection & Predictive Maintenance

Detects machine failures and anomalies from industrial sensor data, explains every alert, and replays the results
on a 3D twin of the equipment. Two sensor domains are supported:

| Dataset | Equipment | Method | Labels used for |
|---|---|---|---|
| **AI4I 2020** | CNC milling spindle (temperatures, speed, torque, tool wear) | Supervised Random Forest / XGBoost | training + evaluation |
| **SKAB** | Water-circulation testbed (vibration, current, voltage, pressure, flow, temperatures) | Unsupervised IsolationForest | evaluation only |

## Setup (Python 3.10+)

```bash
python -m venv venv
venv\Scripts\activate            # Windows   |   source venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt
```

All outputs are already committed, so the dashboard works straight after setup.

## Run the pipeline

```bash
python pipeline.py                                        # AI4I: train + evaluate (held-out 20% split) -> outputs/
python pipeline.py --skab                                 # SKAB valve1 / valve2 / other -> outputs/skab_*/   (~3 min)
python pipeline.py --predict path/to/file.csv             # score a new file -> outputs/custom/
python pipeline.py --train data/train.csv --test data/test.csv   # retrain AI4I-style model on your own data
```

## View the dashboard

```bash
streamlit run dashboard/app.py          # opens http://localhost:8501
```

Pick the data in the sidebar: **AI4I · SKAB Valve1 · SKAB Valve2 · SKAB Other · Other (custom file)**. The twin
switches equipment to match: a CNC spindle for AI4I, a pump/valve rig for SKAB.

Controls: play/pause and replay speed, **NEXT FAULT** to jump to the next alert, drag the timeline to scrub, click an
alert to inspect it, drag the 3D model to orbit / scroll to zoom. The sidebar sets severity thresholds and 3D options.

No Python handy? Open `outputs/dashboard.html` in a browser (standalone, offline, AI4I). Export any dataset with
`python dashboard/app.py --dataset skab_valve2 --export skab.html` (`ai4i`, `skab_valve1`, `skab_valve2`, `skab_other`, `custom`).

## What the replay shows

- **AI4I** has no timestamps: its 10,000 rows are independent simulated snapshots. The dashboard replays the 2,000
  held-out test rows in `UDI` (row id) order as if they were a stream. Sparklines link neighbouring rows, but those
  rows aren't truly consecutive.
- **SKAB** is a real 1 Hz time series. Each group is a set of experiments (one file each), replayed back-to-back in
  time order.
- The 3D heat colouring is an illustrative mapping of sensor readings onto parts, not a physical simulation. Sensor
  values, probabilities, alerts and evidence are the pipeline's real outputs.

## Test with a new dataset

```bash
python pipeline.py --predict path/to/new.csv      # then choose "Other (custom file)" in the dashboard
```

The file type is detected from its columns:

- **AI4I-style sensors** (air/process temperature, speed, torque, tool wear) are scored by the trained model. Column
  names don't need to match exactly (`Ambient_Temp_K`, `Spindle_RPM`, `Torque_Nm` …) and extra columns are ignored.
  If ≥2 of the 5 sensors are missing, the pipeline warns that predictions are unreliable.
- **SKAB-style sensors** (`Accelerometer1RMS`, `Current`, `Pressure`, `Volume Flow RateRMS` …) use the IsolationForest
  method. The file needs a time column and should start with a few minutes of normal operation, which serve as the baseline.
- **Labels are optional.** If present (`Machine failure`, `label`, `target`, `fault`, `anomaly`), F1 / precision /
  recall are computed. Otherwise only predictions and evidence are shown.
- **Try it:** `python pipeline.py --predict datasets/samples/new_machine_demo.csv` (800 held-out AI4I rows with renamed
  columns, minute timestamps and an extra sensor). Any SKAB experiment works as well, e.g. `datasets/SKAB/valve2/1.csv`.
- Equipment with other signals entirely needs its own feature definitions (see the SKAB section of `pipeline.py`).

## Approach

**AI4I (supervised)**
- **Features:** raw sensors + physics-derived features (temperature delta, mechanical power, torque × wear strain,
  temp delta × rpm). Absolute temperatures are dropped so the model is invariant to uniform temperature shifts.
- **Model:** XGBoost vs Random Forest, class-weighted for the 3.4% fault rate and trained on NaN-augmented copies
  (10%/20% sensor dropout) so missing readings are handled natively. The winner is chosen on mean validation F1 across
  clean / missing / noisy / +2 K scenarios; the decision threshold is tuned for F1 on validation only.
- **Explainability:** SHAP drivers per alert with healthy ranges, plus a likely failure mode (HDF/PWF/OSF/TWF).

**SKAB (unsupervised)**
- Each experiment is its own stream. An IsolationForest is fitted on its first 400 s (normal start-up, the standard
  SKAB protocol) using baseline-standardised sensors + 10 s rolling mean/std. The threshold is the 95th percentile of
  baseline scores. Labels are never used for fitting.
- Window and percentile were chosen on **valve1 only**; valve2 and other are held out.
- **Explainability:** SHAP on the IsolationForest. Each alert names its top 2 sensors (σ vs baseline) and the sensor
  group driving it: pump vibration (VIB), motor electrical (ELEC), valve/flow (FLOW) or thermal (TEMP).

## Results

| Dataset | F1 | Precision | Recall | AUC-PR | Note |
|---|---:|---:|---:|---:|---|
| AI4I held-out test split (n = 2,000) | 0.842 | 0.862 | 0.824 | 0.870 | robustness: `outputs/robustness_report.txt` |
| SKAB valve1 (16 runs, 18,162 rows) | 0.647 | 0.606 | 0.695 | 0.633 | settings tuned here (in-sample) |
| SKAB valve2 (4 runs, 4,312 rows) | 0.735 | 0.660 | 0.829 | 0.788 | held out |
| SKAB other (14 runs, 14,985 rows) | 0.554 | 0.544 | 0.564 | 0.505 | held out |

SKAB scores are point-wise over all rows; per-experiment breakdowns are in `outputs/skab_*/evaluation_metrics.txt`.

## Output files

Each dataset folder (`outputs/`, `outputs/skab_valve1/`, …, `outputs/custom/`) contains:

| File | Contents |
|---|---|
| `predictions.csv` | label, fault/anomaly probability, severity, evidence per row |
| `fault_alerts_evidence.csv` | per-alert SHAP drivers and likely failure mode |
| `evaluation_metrics.txt`, `metrics.json` | metrics and confusion matrix (when labels exist) |
| `scored_data.csv`, `run_info.json` | full scored replay + run description (read by the dashboard) |
| `shap_importance.csv` | global feature importance |

AI4I also writes `shap_summary.png`, `robustness_report.txt`, and `models/model.pkl` + `feature_config.json` (the trained
model and its feature/threshold config).

Data: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0); SKAB — Skoltech Anomaly Benchmark (github.com/waico/SKAB).
