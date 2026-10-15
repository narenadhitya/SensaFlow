# SensaFlow — Complete Implementation Plan
## PS05: Intelligent Industrial Fault Detection & Predictive Maintenance
### Hackathon Duration: 2.5 Hours | Solo Builder | Objective F1 Scoring

---

## 0. Strategic North Star

> Build ONE thing well: a pipeline that takes industrial sensor CSV → outputs predictions.csv with evidence.
> The UI exists to prove your pipeline is real. The pipeline exists to score high on hidden test data.
> Every minute spent on decoration is a minute stolen from F1.

### Scoring Weight Map (Your Priorities)

| Criterion | Weight | How You Cover It |
|-----------|--------|-----------------|
| Core performance (F1 on hidden test) | 🔴 HIGH | XGBoost + rolling features + saved model.pkl |
| Correctness & constraints | 🟡 MEDIUM | No hardcoded thresholds, respect data split |
| Robustness | 🟡 MEDIUM | Noise injection test + missing value log |
| Explainability & evidence | 🟡 MEDIUM | SHAP plot + per-alert evidence strings |
| Engineering & reproducibility | 🟢 LOW | requirements.txt + single pipeline.py |
| Usability & demo | 🟢 LOW | Streamlit dashboard (4 panels, 25 min build) |
| Real-world relevance | 🟢 LOW | README intro paragraph |

---

## 1. Dataset Strategy

### Primary Dataset: AI4I 2020 Predictive Maintenance (Kaggle)

**Why this dataset:**
- Directly matches PS05 language — temperature, vibration, RPM, torque, tool wear
- Most commonly used benchmark for this exact problem type in hackathon settings
- Judges building PS05 would have discovered this same dataset
- Clean, labeled, binary + multiclass targets both available
- Small enough to train in under 5 minutes, realistic enough to generalize

**Download source:** `kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020`

**Dataset structure you will see:**

```
Columns:
  UDI                    → row ID (drop)
  Product ID             → product code (drop)
  Type                   → L/M/H product quality variant (encode or drop)
  Air temperature [K]    → feature
  Process temperature [K]→ feature
  Rotational speed [rpm] → feature
  Torque [Nm]            → feature
  Tool wear [min]        → feature
  Machine failure        → PRIMARY LABEL (binary 0/1)
  TWF                    → Tool Wear Failure sub-label
  HDF                    → Heat Dissipation Failure sub-label
  PWF                    → Power Failure sub-label
  OSF                    → Overstrain Failure sub-label
  RNF                    → Random Failure sub-label
```

**Target decision — go binary:**
- Predict `Machine failure` (0 or 1) for the hackathon
- Binary is cleaner, faster to validate, and safer under hidden test scenarios
- Only upgrade to multi-class if binary pipeline finishes with 45+ minutes to spare

**Class imbalance warning:**
- ~3.4% fault rate in AI4I (336 faults out of 10,000 rows)
- Never report accuracy — it will be ~96% predicting all-normal, which is useless
- Always use F1, precision, recall, and confusion matrix

### Fallback Dataset: SKAB (Skoltech Anomaly Benchmark)
If AI4I download fails or takes too long:
- Available at: `github.com/waico/SKAB`
- Multivariate time-series, labeled anomalies, pre-split
- Pivot to unsupervised (IsolationForest) if SKAB has no fault labels

---

## 2. Model Strategy — Three Candidates

### Model A: XGBoost (Primary Choice)

**Why XGBoost wins this hackathon:**
- Consistently best F1 on tabular sensor data in published benchmarks
- `scale_pos_weight` parameter handles the 97/3 class imbalance natively
- Built-in feature importance — covers explainability criterion for free
- Trains in under 60 seconds on AI4I
- Handles missing values internally — no extra imputation needed
- Saved model.pkl is portable and loadable by judges without re-training

**Key parameters to set:**
```
n_estimators: 300
max_depth: 6
learning_rate: 0.1
scale_pos_weight: ~29   ← (negative samples / positive samples = 9664/336)
eval_metric: 'aucpr'    ← better than AUC for imbalanced data
early_stopping_rounds: 20
```

**Expected F1:** 0.82–0.91 on AI4I held-out split (published benchmark range)

---

### Model B: Random Forest (Strong Backup)

**When to use RF over XGBoost:**
- If XGBoost install fails (no internet / environment issues)
- If you need faster experimentation (RF is more forgiving of hyperparameters)
- RF + SHAP is slightly faster to explain than XGBoost + SHAP

**Key parameters:**
```
n_estimators: 200
class_weight: 'balanced'    ← handles imbalance
max_depth: 15
n_jobs: -1                  ← use all CPU cores
random_state: 42
```

**Expected F1:** 0.78–0.86 on AI4I — slightly lower than XGBoost but very reliable

**Decision rule:** Run both, check F1 on your local validation split, submit whichever wins.

---

### Model C: Isolation Forest (Unsupervised Fallback Only)

**When to use IsolationForest:**
- ONLY if the hidden test set has no labels and you must detect anomalies without supervision
- Or if the organizer's hidden data is from a completely different domain (different equipment type)
- As an ensemble component alongside XGBoost — their disagreement zones flag uncertain predictions

**Key parameters:**
```
n_estimators: 200
contamination: 0.034     ← match known fault rate (~3.4% in AI4I)
random_state: 42
```

**The ensemble trick:**
Train both XGBoost and IsolationForest. Where both agree it's a fault → HIGH severity. Where only one flags it → MEDIUM severity. Where both say normal → no alert. This is worth implementing if you finish early — it's a differentiator that directly reads as rigorous to judges.

---

### Model Comparison Summary

| | XGBoost | Random Forest | Isolation Forest |
|---|---------|--------------|-----------------|
| Needs labels | Yes | Yes | No |
| F1 on AI4I | 0.85–0.91 | 0.78–0.86 | N/A (unsupervised) |
| Training time | ~60s | ~45s | ~30s |
| Handles imbalance | Native param | Native param | Via contamination |
| Explainability | SHAP (built-in) | Feature importance | Anomaly score |
| Portability (pkl) | ✅ | ✅ | ✅ |
| **Use when** | **Primary** | **Backup** | **Fallback only** |

---

## 3. Feature Engineering Plan

This is where you create separation from other teams. Raw sensor values are weak predictors — temporal patterns are where fault signatures live.

### Layer 1 — Clean Raw Features (from AI4I)
```
air_temp, process_temp, rpm, torque, tool_wear
→ Normalize to zero mean, unit variance (StandardScaler)
→ Store scaler parameters for use at inference time on hidden test set
```

### Layer 2 — Derived Physics Features
```
temp_delta      = process_temp - air_temp         (heat dissipation indicator)
power           = torque × (rpm × 2π / 60)        (mechanical power in watts)
temp_rpm_ratio  = process_temp / rpm               (thermal stress indicator)
wear_rate       = tool_wear / (rpm + 1)            (tool stress rate)
```
These features encode domain physics — they generalize better across equipment types than raw sensor readings, which directly helps if the hidden test set is from slightly different equipment.

### Layer 3 — Rolling Statistical Features
Window size: 30 samples (or 10 if dataset is small)
```
For each sensor column:
  rolling_mean      → trend detection
  rolling_std       → volatility / instability detection
  rolling_max       → spike capture
  rolling_min       → dip capture
  z_score           → deviation from normal operating range
  rate_of_change    → first difference (sudden spike detection)
  acceleration      → second difference (rate-of-change-of-change)
```

### Layer 4 — Interaction Features (if time allows)
```
torque × tool_wear            (combined mechanical stress)
abs(temp_delta - mean(temp_delta))  (heat dissipation anomaly)
```

### Feature Engineering Priority Order
1. Derived physics features — highest value, domain-specific, generalizable
2. Z-scores — critical for generalization across equipment types
3. Rolling stats — biggest F1 improvement on time-series faults
4. Interactions — nice-to-have, only if time allows

---

## 4. The Feature Mismatch Problem — Full Defense Plan

**The risk:** Hidden test set may have different column names, fewer sensors, or different scaling.

**Scenario A — Different column names** (e.g., `vib_sensor_1` vs `vibration_ch1`)
→ Defense: Map features by position + type, not by name. Store the ordered feature list from training and match positionally if names don't align.

**Scenario B — Missing sensors in test set** (e.g., trained on 5 sensors, test has 3)
→ Defense: Store training column means. At inference time, detect which training features are missing and impute with training means. Never crash — always predict.

**Scenario C — Extra sensors in test set**
→ Defense: Drop extra columns silently. Only use the features the model was trained on.

**Implementation approach:**
- Store `feature_config.json` alongside `model.pkl`
- Contains: ordered feature list, training means, training std values
- At inference: load config, align test data to config, impute missing, predict
- Document this in README as "auto-adapting feature alignment" — it's a robustness feature

---

## 5. Robustness Testing Plan

**Why this is non-negotiable:** It's a MEDIUM weight criterion and almost no team will do it explicitly. Doing it is free points.

### Test 1 — Missing Values (10% random dropout)
Inject 10% random NaN into test features → re-run pipeline → compare fault rate before and after. Log result. Pass criterion = fault rate changes by less than 5%.

### Test 2 — Gaussian Noise
Add Gaussian noise (σ = 0.1) to all sensor readings → re-run → compare. Pass = stable predictions.

### Test 3 — Sensor Dropout (single sensor removed entirely)
Remove the least important sensor (check SHAP for which one) → re-run → compare F1. Pass = F1 degrades gracefully, not catastrophically.

### Test 4 — Operating Condition Shift
Artificially scale all temperatures by 1.1 (10% warmer operating conditions) → re-run → verify model handles it. This directly addresses "changed conditions" from the PS05 brief.

**All results go into `robustness_report.txt`** — this file is your evidence for the robustness criterion.

---

## 6. Explainability & Evidence Plan

**Target:** Every fault alert must trace back to a specific sensor reading and a specific feature value.

### Level 1 — Global Explainability (model level)
SHAP summary plot: which features drive faults across all predictions.
Saved as: `outputs/shap_summary.png`

### Level 2 — Per-Prediction Evidence (alert level)
For every predicted fault, generate one human-readable string:
```
Alert #47 | Severity: HIGH | Confidence: 94.2%
Primary driver: process_temp_rolling_std = 2.87 (normal range: 0.12–0.45)
Secondary driver: torque = 68.3 Nm (normal range: 35–60 Nm)
Fault type: likely Heat Dissipation Failure (HDF pattern match)
```
Saved as: `outputs/fault_alerts_evidence.csv`

### Level 3 — Threshold Documentation
Document what threshold you use for classifying fault vs normal. If using `predict_proba`, document the cutoff (default 0.5, or tuned using Youden's J statistic on validation set). This shows rigor.

---

## 7. Final Submission Package

```
sensaflow/
├── pipeline.py                  ← single runnable script, end-to-end
├── requirements.txt             ← exact package versions
├── feature_config.json          ← feature list + training stats for inference
│
├── models/
│   ├── xgb_model.pkl            ← primary model
│   └── rf_model.pkl             ← backup model (if trained)
│
├── outputs/
│   ├── predictions.csv          ← THE submission (timestamp, label, confidence)
│   ├── fault_alerts_evidence.csv← per-fault evidence strings
│   ├── evaluation_metrics.txt   ← F1, precision, recall, confusion matrix
│   ├── shap_summary.png         ← global feature importance
│   └── robustness_report.txt    ← noise/missing/shift test results
│
├── dashboard/
│   └── app.py                   ← Streamlit UI
│
└── README.md                    ← 15 lines max
```

### predictions.csv format
```
timestamp, predicted_label, fault_probability, alert_severity, evidence_summary
2024-01-01 08:23:11, 1, 0.942, HIGH, "process_temp spike +3.2σ; torque above range"
2024-01-01 08:23:12, 0, 0.041, NONE, ""
```

### evaluation_metrics.txt format
```
SENSAFLOW — EVALUATION REPORT
==================================
Dataset: AI4I 2020 Predictive Maintenance
Train split: 80% (8000 samples)
Test split:  20% (2000 samples)
Strategy:    Stratified split (preserves fault ratio)

Model: XGBoost (n_estimators=300, scale_pos_weight=29)

RESULTS ON HELD-OUT TEST SPLIT
-------------------------------
F1 Score (macro):   0.XXX
F1 Score (fault):   0.XXX
Precision (fault):  0.XXX
Recall (fault):     0.XXX
AUC-PR:             0.XXX

Confusion Matrix:
             Predicted Normal  Predicted Fault
Actual Normal       XXXX              XX
Actual Fault          XX             XXX

ROBUSTNESS SUMMARY (see robustness_report.txt)
F1 under 10% missing values: 0.XXX (Δ = 0.XXX)
F1 under Gaussian noise:     0.XXX (Δ = 0.XXX)
```

---

## 8. UI Design Plan — Streamlit Dashboard

**Time budget: 25 minutes. Not a second more.**
**Design philosophy: SCADA / industrial monitoring system — not a startup landing page**

### Visual Identity
```
Theme:       Dark industrial (#0D1117 base, #161B22 panels)
Accent:      Amber/orange (#F59E0B) for HIGH alerts
Secondary:   Electric teal (#06B6D4) for data lines and normals
Danger:      Red (#EF4444) for fault zones
Success:     Green (#10B981) for normal operation
Typography:  JetBrains Mono for data values (monospace = industrial credibility)
             Inter for labels and text
Borders:     Subtle (#21262D) — no card shadows, no rounded corners
Layout:      2-column grid, dense information layout, left-aligned
```

**Do NOT use:** Warm cream backgrounds, gradient decorations, card shadows, playful animations, centered layouts, rounded pill buttons. These scream student project.

**DO use:** Dense data tables, real-time style charts with grid lines, status indicators (colored dots), monospace numbers, horizontal rule separators, uppercase status labels only for severity levels.

---

### The 4 Panels (Build in this order)

#### Panel 1 — Live Sensor Timeline (top, full width)
An interactive Plotly line chart showing all 5 sensor readings over time.
- X-axis: timestamps
- Y-axis: normalized sensor values (0–1 scale so all sensors visible together)
- Fault zones: vertical red shaded rectangles where `predicted_label == 1`
- Hover tooltip: raw sensor value + z-score + whether this timestep is flagged

**Why this is the hero panel:** Judges immediately understand it without reading anything. Red zones on a time-series is a universal language for "something went wrong here."

#### Panel 2 — Alert Feed (left column, below timeline)
A live-updating table of all detected faults, sorted by severity.
```
Columns: Severity | Timestamp | Confidence | Primary Driver | Evidence
Row example:
🔴 HIGH | 2024-01-15 08:23 | 94.2% | process_temp_std | Temp volatility +3.2σ above normal
🟡 MED  | 2024-01-15 09:11 | 71.8% | torque           | Torque exceeded operating range
```
Color-code rows by severity: red background for HIGH, amber for MEDIUM, no color for LOW.
Include a filter: "Show ALL / HIGH only / MEDIUM+ only"

#### Panel 3 — Model Performance Metrics (right column, below timeline)
Three big number cards:
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   F1 SCORE      │  │   PRECISION     │  │   RECALL        │
│    0.XXX        │  │    0.XXX        │  │    0.XXX        │
│  on test split  │  │  fault class    │  │  fault class    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```
Below: Confusion matrix as a 2×2 heatmap using Plotly (not text).
This is your "claims need proof" moment — metrics displayed directly on screen.

#### Panel 4 — Feature Intelligence (right column, bottom)
A horizontal Plotly bar chart: Top 10 SHAP feature importances.
- Bars colored by feature category (raw sensor = blue, rolling stat = teal, physics-derived = amber)
- Title: "What drives fault predictions"
- X-axis: mean |SHAP value|

This panel alone signals to judges that you understand ML beyond black-box prediction.

---

### Sidebar Controls
```
Sidebar:
  ├── [Severity Filter]    ALL / HIGH / MEDIUM
  ├── [Confidence Threshold] slider 0.3–0.9 (default 0.5)
  └── [Show Robustness Report] toggle → expands robustness_report.txt inline
```

---

### Dashboard Header
```
⚙  SENSAFLOW                          [● LIVE ANALYSIS]
Industrial Fault Detection & Predictive Maintenance
─────────────────────────────────────────────────────────
336 faults detected across 10,000 operational timesteps
Model: XGBoost  |  F1: 0.XXX  |  Dataset: AI4I 2020
```

No hero image. No marketing copy. Just numbers. That's the aesthetic.

---

## 9. Master Timeline

```
BEFORE CLOCK STARTS
├── Download AI4I 2020 from Kaggle              (5 min)
├── pip install xgboost shap streamlit plotly   (5 min)
└── Scaffold folder structure                   (3 min)

HOUR 1 — PIPELINE
  0:00–0:15   EDA on AI4I: understand shape, label distribution, nulls
  0:15–0:30   Feature engineering: physics features + rolling stats + z-scores
  0:30–0:55   Train XGBoost, check F1, tune scale_pos_weight
  0:55–1:05   Train RandomForest, compare F1, keep winner
  1:05–1:15   Save model.pkl + feature_config.json
              Generate predictions.csv on your 20% test split
              ── SUBMIT VERSION 1 (bare predictions, no UI) ──

HOUR 2 — EVIDENCE + UI
  1:15–1:30   SHAP values → shap_summary.png + fault_alerts_evidence.csv
  1:30–1:40   Robustness tests → robustness_report.txt
  1:40–2:00   Streamlit dashboard (4 panels, sidebar)
  2:00–2:10   Connect dashboard to real predictions (not dummy data)
              ── SUBMIT VERSION 2 (full package) ──
  2:10–2:20   README + evaluation_metrics.txt
  2:20–2:30   Final review, clean up outputs, SUBMIT VERSION 3
```

---

## 10. README Template (copy-paste and fill)

```markdown
# SensaFlow — Industrial Fault Detection

## Setup
pip install -r requirements.txt

## Run Pipeline
python pipeline.py --train data/train.csv --test data/test.csv

## Run Dashboard
streamlit run dashboard/app.py

## Approach
Rolling-window feature extraction (window=30) on 5 raw sensor signals.
Physics-derived features (power, heat delta, wear rate) for domain generalization.
XGBoost with scale_pos_weight for class imbalance handling.
Feature-adaptive inference — handles missing sensors via training-mean imputation.

## Results (held-out 20% split, stratified)
F1:         0.XXX
Precision:  0.XXX
Recall:     0.XXX

## Robustness
Model tested under 10% random missing values, Gaussian noise, and sensor dropout.
Results in outputs/robustness_report.txt

## Output Files
- outputs/predictions.csv          → label + confidence per timestep
- outputs/fault_alerts_evidence.csv→ human-readable evidence per fault
- outputs/evaluation_metrics.txt   → full evaluation report
- outputs/shap_summary.png         → SHAP feature importance
- outputs/robustness_report.txt    → robustness test log
```

---

## 11. Risk Register & Mitigations

| Risk | Probability | Mitigation |
|------|------------|------------|
| AI4I not available (slow Kaggle) | Low | Download now before clock starts |
| XGBoost install fails | Low | RandomForest is fully available in sklearn |
| Hidden test set has different features | Medium | feature_config.json + auto-imputation pipeline |
| SHAP takes too long (>300s) | Low | Sample 500 rows for SHAP, not full dataset |
| Streamlit port conflict | Low | `streamlit run app.py --server.port 8502` |
| Overfitting to AI4I | Medium | Use z-scores and relative features, not absolute thresholds |
| Running out of time | Medium | Submit V1 at 1:15 — bare predictions are better than a late polished submission |

---

## 12. The Scoring Advantage Summary

Most teams will do:
- Train a model ✓
- Output predictions ✓
- Skip robustness ✗
- Skip evidence strings ✗
- Skip feature engineering depth ✗
- Submit a basic UI ✗

You will do all 7. The three MEDIUM criteria are essentially free points that most teams leave on the table. Your edge is preparation — you're thinking about all 7 criteria before the clock starts.

**The mantra:** Submit early, improve iteratively, document everything.
```

---
*Plan version: Final | Hackathon: Bugged Out | Problem: PS05 SensaFlow*
*Solo builder strategy — optimized for 2.5hr execution window*
