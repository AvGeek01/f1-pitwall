# F1 Pit Wall

F1 Pit Wall is a data-driven pit-stop strategy project. It combines historical
race data, models of observed team and driver behaviour, and a Monte Carlo race
simulator to compare possible tyre and pit-stop strategies.

> **Scope:** This is a strategy analysis and decision-support project, not a
> claim that the model can identify a guaranteed optimal race strategy.

## Problem statement

Pit-stop decisions trade the time lost in the pit lane against tyre age,
compound performance, track position, and uncertain race events. The project
asks:

1. Given the race state, when have teams historically chosen to pit?
2. How many stops are drivers likely to make at a race?
3. Under explicit simulation assumptions, which candidate strategy has the
   lowest expected race time?

The first two questions learn from observed decisions. Those decisions are
noisy labels: a team's actual choice is not necessarily the best possible
choice. The simulator therefore evaluates candidate strategies separately
instead of treating observed behaviour as optimal.

## Data sources

Historical data is retrieved from the **OpenF1 REST API**
([documentation](https://openf1.org/docs), base URL
`https://api.openf1.org/v1`). This project uses race-session data for the
2023–2025 seasons and does not use FastF1.

The fetcher caches raw JSON responses under `data/raw/openf1/` and writes
endpoint Parquet files under `data/raw/openf1_parquet/`. The cleaning pipeline
produces:

- `data/processed/laps_clean.parquet` — cleaned driver-lap records.
- `data/processed/pit_loss.parquet` — circuit-level pit-loss estimates.

Raw API data is not edited by the cleaning or modelling steps. The lap cleaner
preserves driver-lap rows and marks unavailable or implausible pace durations
as missing rather than inventing replacement lap times. A 300-second
plausibility threshold is applied to pace-model inputs.

## Method: three layers

### 1. Predict observed team behaviour

The A5 layer uses race-state features to estimate pit timing:

- A regression target estimates laps until the next observed pit stop.
- A classification target estimates whether a pit stop occurs within the next
  three laps.

The targets are derived from observed pit-in laps. They describe historical
behaviour, not an optimal decision. The models are XGBoost estimators wrapped
in preprocessing pipelines; numeric and categorical features follow the
existing A3 feature builder and Streamlit input schema.

### 2. Predict stop count

The A6 layer builds one record per driver per race and classifies observed
pit-stop count as **1 stop**, **2 stops**, or **3+ stops**. It compares a
Random Forest and XGBoost classifier with a circuit-majority baseline.
Historical circuit and global stop-count features use prior seasons only.
Driver-race records with no observed pit stops are excluded from this
three-class target.

### 3. Recommend a strategy via simulation

The simulator enumerates candidate compound sequences and stint lengths, then
uses seeded Monte Carlo simulations to estimate total race time. It combines:

- Fitted tyre-degradation curves from clean, dry, non-pit-in and non-pit-out
  laps.
- Circuit-level pit-loss estimates.
- Compound pace offsets and lap-time noise.
- A stochastic safety-car component and its configured effects.

Candidate strategies are ranked by simulated mean race time. These rankings are
conditional on the simulator's assumptions and should be treated as
recommendations for comparison, not live race predictions.

## Leakage controls and validation

- The primary temporal split trains on **2023–2024** and holds out **2025**.
- The A5 models use race-grouped cross-validation; laps from the same race are
  not randomly split between training and validation folds.
- The A6 models use stratified, race-grouped cross-validation. Historical
  features are recomputed from each fold's training races.
- A6 historical stop-count features for a season use only strictly earlier
  seasons. The 2025 holdout uses 2023–2024 reference data.
- Future pit events are used to construct A5 target labels only; target
  columns are not model input features.
- Degradation fitting and simulator validation use specified historical
  training seasons, keeping the validation season out of degradation fitting.
- Random seeds are fixed at 42 where applicable.

## Results

### A5: observed pit behaviour, 2025 holdout

The following scores are from the latest A5 retraining and validation using
real processed OpenF1 data: 2023–2024 training (51,108 feature rows) and 2025
holdout (26,265 feature rows). The regressor's scored holdout subset contains
rows with a known future pit target; the classifier is scored on its valid
holdout target rows.

| Model / metric | 2025 holdout score |
|---|---:|
| A4 baseline regression MAE (laps) | 9.127 |
| A5 XGBoost regressor MAE (laps) | 6.483 |
| A5 XGBoost classifier PR-AUC | 0.191 |
| A5 XGBoost classifier F1 | 0.280 |
| A5 classifier selected threshold | 0.639 |
| A5 classifier out-of-fold training F1 | 0.293 |

The classifier's threshold was selected on out-of-fold training predictions.
The A5 scores quantify agreement with observed pit behaviour and should not be
interpreted as strategy optimality.

### A6: stop-count classification, saved 2025 holdout report

These are the values in `reports/a6_holdout_metrics.csv`. Macro-F1 confidence
intervals are the saved 95% bootstrap intervals.

| Model | Accuracy | Macro-F1 | Macro-F1 95% CI |
|---|---:|---:|---:|
| Random Forest | 0.757 | 0.755 | [0.700, 0.801] |
| XGBoost | 0.697 | 0.689 | [0.632, 0.743] |
| Circuit-majority baseline | 0.400 | 0.190 | [0.170, 0.209] |

## Limitations

- **Noisy behavioural labels:** teams optimize under private information and
  changing race conditions. Historical pit decisions are not ground truth for
  the best decision.
- **Wet races:** wet laps are excluded from degradation fitting. The current
  strategy simulator does not model the full wet-tyre decision problem, so its
  recommendations are not suitable for interpreting wet-race strategy.
- **Right-censored stints:** races ending before a driver's next stop provide
  no observed next-stop time. These cases are censored or excluded where the
  regression target is unavailable, rather than assigned an arbitrary target.
- **No team-internal information:** the data does not include confidential
  tyre condition, driver feedback, damage, planned strategy, or real-time
  competitor intelligence.
- **Safety cars are unpredictable:** the simulator represents safety cars
  using a configured probability and simplified effects. It cannot forecast
  the timing or consequences of an actual incident.
- **Simplified race dynamics:** traffic, overtaking, pit release interactions,
  exact weather transitions, and many operational constraints are not fully
  modelled. Predicted times and rankings are conditional on model inputs.
- **Historical coverage and quality:** OpenF1 fields may be missing or
  inconsistent by session. Missing pace measurements remain missing and can
  reduce the data available to a particular model.
- **Stop-count classes:** the A6 target groups all races with three or more
  stops into one class and excludes zero-stop driver-race records.

## Build information

This project was built by Rishit & Dennis.
This project is built for informational purposes only.

## Setup

Use Python 3.11. From the repository root, create and activate a virtual
environment, then install the project dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run instructions

### Fetch and build the real-data pipeline

Fetch OpenF1 data for the supported historical seasons:

```powershell
python -m src.data.fetch 2023 2024 2025
```

Run the end-to-end pipeline:

```powershell
python scripts/run_pipeline.py
```

If raw OpenF1 JSON is already cached, rebuild the endpoint Parquet files
without making network requests:

```powershell
python scripts/run_pipeline.py --skip-fetch
```

The default pipeline uses 2023–2024 for training and 2025 as the validation
season. To change the seasons explicitly:

```powershell
python scripts/run_pipeline.py `
  --seasons 2023 2024 2025 `
  --training-seasons 2023 2024 `
  --validation-season 2025
```

### Validate processed data

```powershell
python scripts/check_data.py
```

### Run the Streamlit dashboard

After the processed data and required model artifacts are available:

```powershell
python -m streamlit run src/app/streamlit_app.py
```

### Run tests

```powershell
python -m pytest
```

The pipeline smoke test uses synthetic data for offline testing only. Synthetic
fixtures are not a substitute for the real OpenF1 processed dataset in model
training or strategy analysis.

## Project layout

```text
data/
  raw/                 Cached OpenF1 source data (not modified by cleaning)
  processed/           Clean laps, pit-loss estimates, and test fixtures
models/                Saved model artifacts
reports/               Evaluation tables and figures
scripts/               Data checks and pipeline entry point
src/
  app/                 Streamlit dashboard
  data/                OpenF1 fetching, cleaning, and schema normalization
  features/            Race-state features and targets
  models/              Pit behaviour, stop-count, and degradation models
  sim/                 Monte Carlo strategy simulation and validation
tests/                 Unit, integration, and smoke tests
```
