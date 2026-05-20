# M5 Inventory Optimizer
### Senior Applied Data Science Project — Demand Forecasting & Supply Chain Optimization

---

## Project Overview

An end-to-end ML pipeline built on the **M5 Forecasting Competition dataset (Walmart)** that combines
hierarchical time-series forecasting with mathematical inventory optimization to produce actionable,
cost-minimizing reorder recommendations.

> **Business Goal:** Replace naive "order what sold last week" purchasing with a statistically-grounded
> system that minimizes total supply chain cost (holding costs + stockout penalties) while maintaining
> a 95% service level.

---

## Project Architecture

```
m5_inventory_optimizer/
├── config/
│   └── settings.py              # All constants, paths, hyperparameters
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # Memory-optimized CSV reader + type downcaster
│   │   ├── preprocessor.py      # Wide→long melt, table merges, cache writer
│   │   └── validator.py         # Schema checks, null audits, leakage guards
│   ├── features/                # (Phase 3) Lag, rolling, calendar features
│   ├── models/                  # (Phase 4) LightGBM training + walk-forward CV
│   └── optimization/            # (Phase 5) Safety stock + cost minimization
├── notebooks/
│   └── 01_data_understanding.py # Standalone EDA script (runnable as notebook)
├── data/
│   ├── raw/                     # ← Place M5 CSVs here
│   ├── processed/               # Parquet outputs from ingestion pipeline
│   └── cache/                   # Intermediate artifacts
└── outputs/
    ├── plots/                   # EDA and evaluation charts
    ├── reports/                 # Financial impact reports
    └── models/                  # Serialized model artifacts
```

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Data understanding & memory-optimized loading |
| 2 | 🔜 Next | Exploratory Data Analysis (EDA) |
| 3 | ⬜ Pending | Feature engineering store |
| 4 | ⬜ Pending | LightGBM + walk-forward validation |
| 5 | ⬜ Pending | Inventory optimization (Safety Stock, ROP) |
| 6 | ⬜ Pending | Business value reporting dashboard |

---

## Data Sources

[M5 Forecasting - Accuracy]

Place these files in `data/raw/`:
```
data/raw/
├── calendar.csv
├── sell_prices.csv
├── sales_train_validation.csv
└── sales_train_evaluation.csv   # optional
```

### File Relationships
```
sales_train_validation.csv  (wide: item × day)
        │
        │  d_1..d_1913 → melt → (item_id, d, sales)
        │
        ├── JOIN calendar.csv       on d → date, events, SNAP flags
        └── JOIN sell_prices.csv    on (store_id, item_id, wm_yr_wk)
```

---

## Setup

```bash
pip install -r requirements.txt

# Run the full ingestion pipeline
python -m src.ingestion.preprocessor

# Run EDA script
python notebooks/01_data_understanding.py
```

---

## Key Engineering Decisions

### Memory Optimization Strategy
Raw M5 data is ~3.5 GB in RAM when naively loaded. This pipeline reduces it to ~400 MB via:
- Explicit dtype downcasting (`float64→float32`, `int64→int16/int8`)
- Column-selective loading (only load columns needed per stage)
- Chunked melting to avoid peak memory spikes
- Parquet caching with Snappy compression for fast re-loads

### Why LightGBM over Deep Learning?
- M5 has high zero-inflation (intermittent demand) — tree models handle sparse features natively
- Gradient boosted trees are interpretable for business stakeholders
- LightGBM with Tweedie loss is state-of-the-art on this exact dataset

### Inventory Math
Safety Stock formula accounts for both demand and lead-time variability:

```
SS = Z × sqrt(L × σ_D² + μ_D² × σ_L²)
```

Total Cost objective minimized:
```
Total Cost = (Holding Cost × Inventory Level) + (Stockout Penalty × Missed Sales)
```
