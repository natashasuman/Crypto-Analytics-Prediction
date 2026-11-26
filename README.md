# Cryptocast: Automated Crypto Forecasting Platform

[![CI](https://github.com/natashasuman/Crypto-Analytics-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/natashasuman/Crypto-Analytics-Prediction/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Architecture: Medallion](https://img.shields.io/badge/architecture-medallion-orange.svg)](#system-architecture)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, fully automated crypto forecasting system that ingests live and historical Binance market data plus GNews headline sentiment, constructs leakage-safe feature pipelines, and outputs **two predictions per asset per bar**:
1. **Direction**: Calibrated probability ($p_{\text{up}} \in [0, 1]$) of a positive return.
2. **Expected Return**: Predicted log return over a configurable horizon (1, 4, 24 bars).

Operating entirely on **free-tier infrastructure**: GitHub Actions cron for ephemeral compute, Parquet + DuckDB with Hugging Face Hub dataset persistence, and static Plotly dashboards served via GitHub Pages.

---

## Key Highlights

- **DAG-Orchestrated Pipelines**: Automated data ingestion, validation, sentiment analysis, feature engineering, and inference with exponential backoff retries, idempotent upserts, and watermark catch-up.
- **Strict Data-Quality Gates**: Schema enforcement, gap detection, OHLC consistency, freshness checks, and anomaly alerts. Downstream tasks halt immediately on severity `error`.
- **Honest & Leakage-Free ML**:
  - Purged walk-forward cross-validation with embargo periods (López de Prado methodology).
  - Strict availability timestamps for news headlines (no lookahead bias).
  - Paired bootstrap confidence intervals against naive baselines (majority class & momentum).
  - Economic validation via fee-adjusted backtesting (10 bps/side buffer).
- **Automated Modeling Lifecycle**:
  - Weekly champion/challenger retraining.
  - Automated feature ablation & pruning.
  - Strict promotion gate requiring statistical outperformance, calibration ($ECE \le 0.05$), and fee-adjusted profitability.
  - Population Stability Index (PSI) drift monitoring triggering automatic retraining.

---

## System Architecture

```
                    ┌──────────────────────── GitHub Actions (cron) ───────────────────────┐
                    │                                                                       │
 Binance Market API │   HOURLY DAG (ingest_and_predict)        WEEKLY DAG (retrain)         │
 GNews API          │   ┌────────────┐                         ┌──────────────────┐         │
                    │   │ ingest_    │                         │ build_training_  │         │
        │           │   │ klines     │                         │ set              │         │
        ▼           │   └─────┬──────┘                         └────────┬─────────┘         │
 ┌────────────┐     │         ▼                                         ▼                   │
 │ Raw (bronze)│◄───┤   ┌────────────┐                         ┌──────────────────┐         │
 └─────┬──────┘     │   │ validate_  │ fail → halt + alert     │ feature_ablation │         │
       ▼            │   │ klines     │                         └────────┬─────────┘         │
 ┌────────────┐     │   └─────┬──────┘                                  ▼                   │
 │ Clean(silver)│◄──┤         ▼                                 ┌──────────────────┐         │
 └─────┬──────┘     │   ┌────────────┐                         │ tune + train     │         │
       ▼            │   │ ingest_news│                         │ (Optuna, XGB/RF) │         │
 ┌────────────┐     │   └─────┬──────┘                         └────────┬─────────┘         │
 │Features(gold)│◄──┤         ▼                                         ▼                   │
 └─────┬──────┘     │   ┌────────────┐                         ┌──────────────────┐         │
       ▼            │   │ score_     │                         │ evaluate vs      │         │
 ┌────────────┐     │   │ sentiment  │                         │ champion         │         │
 │ Signals    │◄────┤   └─────┬──────┘                         └────────┬─────────┘         │
 └─────┬──────┘     │         ▼                                         ▼                   │
       ▼            │   ┌────────────┐                         ┌──────────────────┐         │
 Dashboard (Pages)  │   │build_      │                         │ promote / reject │         │
 Alerts (Issues/    │   │features    │                         │ + report         │         │
 webhook)           │   └─────┬──────┘                         └──────────────────┘         │
                    │         ▼                                                              │
                    │   ┌────────────┐     DAILY DAG (monitor): drift, freshness,           │
                    │   │ predict    │     live accuracy, anomalies → dashboard + alerts    │
                    │   └────────────┘                                                      │
                    └───────────────────────────────────────────────────────────────────────┘
```

### Medallion Storage Model

- **Bronze (Raw)**: Untouched responses from Binance market data and GNews headlines in zstd-compressed Parquet.
- **Silver (Clean)**: Validated, deduplicated OHLCV candles and FinBERT-scored sentiment records.
- **Gold (Features & Signals)**: Leakage-safe feature matrices and calibrated inference outputs.
- **Meta**: Pipeline execution logs (`runs.parquet`), data-quality check results (`dq_results.parquet`), and ingestion timestamps (`watermarks.json`).

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Linting, type checking, and unit tests
│       ├── hourly.yml             # Hourly ingestion and prediction DAG
│       ├── retrain.yml            # Weekly champion/challenger retraining
│       ├── monitor.yml            # Daily drift and DQ monitoring
│       └── pages.yml              # Plotly dashboard publishing
├── config/
│   ├── assets.yaml                # Monitored crypto symbols, intervals, horizons
│   ├── features.yaml              # Feature groups, window sizes, indicators
│   ├── model.yaml                 # XGBoost / RF configs, Optuna budgets
│   ├── promotion.yaml             # Champion/challenger gate criteria & thresholds
│   └── dq.yaml                    # Data quality check severities
├── data/                          # Local storage directory (persisted to HF Hub)
├── scripts/
│   ├── m0_binance_spike.py        # M0 Binance market data endpoint verification
│   ├── m0_hf_spike.py             # M0 Hugging Face Hub persistence verification
│   └── m0_gnews_spike.py          # M0 GNews connectivity verification
├── src/cryptocast/
│   ├── cli.py                     # Typer CLI entrypoint
│   ├── config.py                  # Pydantic settings & YAML configuration loader
│   ├── dashboard/                 # Plotly dashboard generation
│   ├── evaluation/                # Purged walk-forward CV, metrics, and backtest
│   ├── features/                  # Leakage-safe technical and cross-asset features
│   ├── flows/                     # Prefect workflow DAG definitions
│   ├── ingest/                    # Binance & GNews ingestion adapters
│   ├── models/                    # Model training, Optuna tuning, calibration
│   ├── monitoring/                # PSI drift, live accuracy, and alerts
│   ├── quality/                   # Schema validation and integrity checks
│   ├── registry/                  # MLflow tracking and promotion gate
│   ├── sentiment/                 # FinBERT and VADER sentiment scorers
│   └── storage/                   # Storage abstraction (LocalStorage & HFHubStorage)
├── tests/
│   ├── unit/                      # Fast unit tests for storage, config, CLI, features
│   ├── integration/               # Pipeline end-to-end integration tests
│   └── fixtures/                  # Mocked API responses and sample datasets
├── Makefile                       # Developer shortcuts (bootstrap, test, lint)
├── pyproject.toml                 # uv project configuration and dependencies
└── prd.md                         # Product Requirements Document
```

---

## Quickstart & Local Setup

### 1. Prerequisites

- Python `>= 3.11`
- [uv](https://github.com/astral-sh/uv) (recommended package manager) or standard `pip`

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/natashasuman/Crypto-Analytics-Prediction.git
cd Crypto-Analytics-Prediction

# Install core and dev dependencies with uv
uv sync --extra dev
```

### 3. Environment Configuration

Copy the example environment file and configure any optional tokens:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `STORAGE_BACKEND` | Storage backend: `local` or `hf_hub` | `local` |
| `LOCAL_STORAGE_DIR` | Local data folder path | `data` |
| `HF_REPO_ID` | Hugging Face Dataset repository (for cloud persistence) | Optional |
| `HF_TOKEN` | Hugging Face Access Token with write access | Optional |
| `GNEWS_API_KEY` | GNews API token for news ingestion | Optional |
| `ALERT_WEBHOOK_URL` | Discord or Telegram webhook for incident notifications | Optional |

### 4. Run Feasibility Spikes (Milestone M0)

Verify market data connectivity (utilizing Binance's market-data endpoint to prevent US IP blocks):

```bash
uv run python scripts/m0_binance_spike.py
uv run python scripts/m0_hf_spike.py
uv run python scripts/m0_gnews_spike.py
```

### 5. CLI Usage

The system exposes a unified Typer CLI:

```bash
# Display CLI help and available commands
uv run cryptocast --help

# Inspect version and storage configuration
uv run cryptocast version

# View configured assets and horizons
uv run cryptocast info

# Execute pipeline tasks locally
uv run cryptocast ingest --symbol BTCUSDT
uv run cryptocast features --symbol BTCUSDT
uv run cryptocast train --symbol BTCUSDT --model-type xgboost
uv run cryptocast predict
uv run cryptocast monitor
```

### 6. Running Tests & Quality Checks

```bash
# Run unit tests with coverage
uv run pytest tests/unit

# Run linter and type checker
uv run ruff check src tests
uv run mypy src
```

---

## Project Milestones

- [x] **M0: Feasibility Spikes**: Validated public Binance market data endpoint (`data-api.binance.vision`), Hugging Face Hub persistence layer, and GNews API contracts.
- [x] **M1: Foundations**: Repo skeleton, Pydantic YAML configuration loader, Storage abstraction (`LocalStorage` & `HFHubStorage`), Typer CLI, CI workflow, and unit test suite.
- [x] **M2: Data Ingestion & Quality Gates**: 24-month backfill, incremental hourly ingest, watermark tracking, and DQ assertion checks.
- [x] **M3: Sentiment & Feature Engineering**: FinBERT & VADER headline scoring, 50+ technical and market features across 8 groups, dual-head target labels, and automated leakage tests.
- [ ] **M4: Baseline Modeling & Purged CV**: Purged walk-forward cross-validation with embargo, naive benchmarks, XGBoost/Random Forest dual-head models, and fee-adjusted backtests.
- [ ] **M5: Hourly Ingestion & Prediction DAG**: Ephemeral Prefect flow running inside GitHub Actions with watermark catch-up and concurrency controls.
- [ ] **M6: Automated Retraining & Promotion**: Optuna hyperparameter optimization, probability calibration, automated feature ablation, and MLflow champion/challenger gate.
- [ ] **M7: Monitoring, Drift & Alerting**: PSI feature drift tracking, live metric degradation monitoring, automated incident dispatch, and GitHub Pages Plotly dashboard.
- [ ] **M8: Final Polish & Serving**: Production documentation, runbooks, and optional containerized FastAPI serving.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
