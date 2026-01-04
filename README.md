# Purchase Prediction + Campaign Audience Targeting (Notebook-only)

This project builds an end-to-end, **notebook-only** pipeline for:

1) **Purchase prediction (classification, KPI-driven)**  
2) **Campaign audience targeting (ranking, Top‑k KPI-driven)**

It is designed to prevent common pitfalls from early experiments (overconfident probabilities, calibration leakage/omission, EV score collapse, insufficient artifact saving) by enforcing:
- **time-based splits**
- **saved-predictions-based calibration**
- **sanity checks that fail fast**
- **systematic artifact export** (metrics/reports/predictions/figures)

---

## Environment

Executed locally in Jupyter on a PC with **30GB RAM + 9GB GPU** (GPU not required).  
All notebooks ran smoothly and quickly; resource usage stayed well below hardware limits.

---

## Data

Source: eCommerce events history (electronics store).  
Event schema:

- `event_time`, `event_type` (`view`, `cart`, `purchase`), `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`

Event totals (after basic QC):
- view: 793,748  
- cart: 54,035  
- purchase: 37,346  
- time range: 2020‑09‑24 → 2021‑02‑28 (UTC)

---

## Notebook pipeline

### 00_data_check_and_parquet.ipynb
- Streams raw CSV → writes optimized parquet
- Creates monthly parquets
- Produces QC summary (counts, time range, basic sanity)

### 01_build_user_dataset.ipynb
- Builds **weekly user snapshots** using a fixed window design:
  - history window: **23 days**
  - label window: **7 days**
- Creates user-level features (counts, ratios, recency, price stats)
- Produces **time-based splits** by label_end:
  - train / valid / test
- Output dataset: **1,183,222 snapshots**, 19 columns

Split stats:
- train base_rate: **0.001442**
- valid base_rate: **0.002066**
- test  base_rate: **0.001976**

### 02_train_purchase_model.ipynb
- Trains and compares models (LogReg, CatBoost, LightGBM)
- Selects best model by **VALID PR-AUC**
- Saves:
  - purchase metrics tables (VALID/TEST)
  - threshold selection (VALID-only)
  - raw predictions (VALID/TEST) for downstream calibration

**Best model (by VALID PR-AUC): `catboost`**

### 03_train_value_model.ipynb
- Trains **conditional value model** on buyers only:
  - target: `log1p(y_revenue)`
- Writes:
  - `rev_hat` to prediction files
  - `ev = p_hat * rev_hat`

### 04_calibration.ipynb
- Reads **saved prediction files only** (prevents silent calibration leakage)
- Fits **Platt vs Isotonic** on VALID, selects by **VALID logloss**
- Applies calibration to TEST with **file-based sanity checks**
- Saves calibrated predictions with both methods:
  - `p_cal_platt`, `p_cal_isotonic`
  - canonical `p_cal` and `p_cal_method`
  - `ev_cal_platt`, `ev_cal_isotonic`, canonical `ev_cal`

### 05_campaign_ranking_reports.ipynb
- Compares ranking strategies:
  - `p_hat`, `p_cal_*`
  - `ev`, `ev_cal_*`
  - tuned score: `score = (p_hat^alpha) * (rev_hat^beta)`
- Tunes `(alpha, beta)` on VALID by **revenue_capture@1%**
- Exports:
  - top‑k reports
  - decile reports
  - revenue capture curves
  - alpha/beta grid + best params

### 06_summary_tables_for_readme.ipynb
- Auto-generates README-ready summaries and Top‑k highlights
- Writes `artifacts/reports/readme_snippets.md`

---

## Results

### 1) Purchase prediction (classification)

Best model by VALID PR-AUC: **catboost**

**VALID**
- base_rate=0.002066, ROC-AUC=0.8780, PR-AUC=0.0468, LogLoss=0.4132, Brier=0.1239  
- F1-max threshold (thr=0.9556): Precision=0.1049, Recall=0.1233, F1=0.1133, Predicted Positive Rate=0.0024  
- Precision≥10% threshold (thr=0.9545): Precision=0.0999, Recall=0.1267, F1=0.1117, PPR=0.0026  

**TEST**
- base_rate=0.001976, ROC-AUC=0.8711, PR-AUC=0.0407, LogLoss=0.4126, Brier=0.1240  
- F1-max threshold (thr=0.9556): Precision=0.0939, Recall=0.1020, F1=0.0978, PPR=0.0021  
- Precision≥10% threshold (thr=0.9545): Precision=0.0898, Recall=0.1060, F1=0.0972, PPR=0.0023  

> Note: raw `p_hat` is heavily overconfident (mean ~0.285 vs base_rate ~0.002), so probability-based decisions require calibration.

---

### 2) Calibration effect (probability quality)

Best method by VALID logloss: **isotonic**  
(Platt was slightly better on TEST logloss, but both massively improved over raw.)

**VALID**
- raw: p_mean=0.285296, logloss=0.413211, brier=0.123886  
- platt: p_mean=0.002088, logloss=0.011697, brier=0.002009  
- isotonic: p_mean=0.002066, logloss=0.011590, brier=0.002003  

**TEST**
- raw: p_mean=0.284399, logloss=0.412595, brier=0.123964  
- platt: p_mean=0.002069, logloss=0.011574, brier=0.001931  
- isotonic: p_mean=0.002043, logloss=0.011697, brier=0.001932  

**Calibration application sanity check (TEST)**
- max_abs_diff(p_cal − p_hat) > 0 ✅
- p_cal_mean close to base_rate (ratio_to_base ≈ 1.03) ✅

---

### 3) Campaign targeting (ranking)

Scores evaluated:
- conversion scores: `p_hat`, `p_cal`, `p_cal_platt`, `p_cal_isotonic`
- revenue scores: `ev`, `ev_cal`, `ev_cal_platt`, `ev_cal_isotonic`
- hybrid: `tuned = (p_hat^alpha) * (rev_hat^beta)`

**Best tuned params (VALID, objective = revenue_capture@1%)**
- alpha=**2.0**, beta=**0.25**
- Interpretation:
  - higher alpha emphasizes purchase probability differences
  - lower beta limits over-reliance on revenue estimates (prevents EV collapse)

#### Recommended scores (TEST)
- **Conversion-first**: `p_cal` (maximizes purchase_rate@1%)
- **Revenue-first**: `ev_cal_platt` (maximizes revenue_capture@1%)
- **Hybrid**: `tuned` (alpha=2.0, beta=0.25)

Top‑k highlights (TEST):

- **p_cal**
  - Top1%: revenue_capture=0.4001, purchase_rate=0.0537  
  - Top10%: revenue_capture=0.7489, purchase_rate=0.0131  

- **ev_cal_platt**
  - Top1%: revenue_capture=0.4512, purchase_rate=0.0450  
  - Top10%: revenue_capture=0.8140, purchase_rate=0.0124  

- **tuned**
  - Top1%: revenue_capture=0.4349, purchase_rate=0.0399  
  - Top10%: revenue_capture=0.7948, purchase_rate=0.0130  

Key takeaway:
- **Raw EV (`ev`) underperformed** (value estimate dominates and breaks conversion).  
- **Calibrated EV (`ev_cal_*`) fixed this** and consistently led revenue capture across top fractions.

---

## Artifacts

Outputs are organized under:

- `data/processed/` — parquet datasets and split files  
- `artifacts/models/` — trained models  
- `artifacts/predictions/` — raw + calibrated predictions  
- `artifacts/metrics/` — classification + calibration metrics  
- `artifacts/reports/` — top‑k, decile, curves, grid search results  
- `artifacts/figures/` — PR curve, calibration curve, revenue capture curve

---

## Notes / limitations

- Evaluation uses **weekly snapshots** with fixed windows (history=23d, label=7d).  
- Results reflect a single multi-month span (2020‑09 to 2021‑02).  
- For stronger confidence, extend the time horizon and run **rolling backtests**:
  - multiple sequential cutoffs
  - monitoring drift in PR-AUC, calibration (logloss/brier), and revenue capture curves
  - validating stability of the chosen score (e.g., `ev_cal_platt`) over time

---

## How to run

1) Run notebooks in order:
   - 00 → 01 → 02 → 03 → 04 → 05 → 06
2) Use artifacts in `artifacts/reports/readme_snippets.md` for quick summaries.
