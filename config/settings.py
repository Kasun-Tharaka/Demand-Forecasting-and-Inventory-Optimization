"""
config/settings.py
──────────────────
Single source of truth for all project constants, file paths,
dtype schemas, and pipeline hyperparameters.

Usage:
    from config.settings import cfg
    df = pd.read_csv(cfg.RAW_SALES)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Root Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CACHE     = PROJECT_ROOT / "data" / "cache"
OUTPUTS_PLOTS  = PROJECT_ROOT / "outputs" / "plots"
OUTPUTS_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_REPORTS= PROJECT_ROOT / "outputs" / "reports"


# ─────────────────────────────────────────────────────────────
# Raw File Paths
# ─────────────────────────────────────────────────────────────
RAW_SALES     = DATA_RAW / "sales_train_validation.csv"
RAW_CALENDAR  = DATA_RAW / "calendar.csv"
RAW_PRICES    = DATA_RAW / "sell_prices.csv"
RAW_SALES_EVAL= DATA_RAW / "sales_train_evaluation.csv"   # optional


# ─────────────────────────────────────────────────────────────
# Processed / Cached Artefacts
# ─────────────────────────────────────────────────────────────
PROCESSED_LONG    = DATA_PROCESSED / "sales_long.parquet"
PROCESSED_MERGED  = DATA_PROCESSED / "sales_merged.parquet"
PROCESSED_CALENDAR= DATA_PROCESSED / "calendar_clean.parquet"
PROCESSED_PRICES  = DATA_PROCESSED / "prices_clean.parquet"


# ─────────────────────────────────────────────────────────────
# M5 Dataset Constants
# ─────────────────────────────────────────────────────────────
# Total training days in validation set
N_TRAIN_DAYS = 1913
# Forecast horizon used in the competition
FORECAST_HORIZON = 28
# Day-column prefix in the wide sales file
DAY_COL_PREFIX = "d_"

# Hierarchical grouping columns
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# M5 states
STATES = ["CA", "TX", "WI"]

# M5 stores
STORES = ["CA_1", "CA_2", "CA_3", "CA_4",
          "TX_1", "TX_2", "TX_3",
          "WI_1", "WI_2", "WI_3"]

# M5 categories
CATEGORIES = ["HOBBIES", "HOUSEHOLD", "FOODS"]


# ─────────────────────────────────────────────────────────────
# Dtype Schemas — the cornerstone of memory optimisation
# ─────────────────────────────────────────────────────────────

# Calendar dtype map (applied immediately on load)
CALENDAR_DTYPES: dict[str, str] = {
    "wm_yr_wk":    "int16",
    "weekday":     "category",
    "wday":        "int8",
    "month":       "int8",
    "year":        "int16",
    "snap_CA":     "int8",
    "snap_TX":     "int8",
    "snap_WI":     "int8",
    "event_name_1":"category",
    "event_type_1":"category",
    "event_name_2":"category",
    "event_type_2":"category",
}

# Sales wide-format meta columns dtype map
SALES_META_DTYPES: dict[str, str] = {
    "id":       "category",
    "item_id":  "category",
    "dept_id":  "category",
    "cat_id":   "category",
    "store_id": "category",
    "state_id": "category",
}

# Sell prices dtype map
PRICE_DTYPES: dict[str, str] = {
    "store_id":  "category",
    "item_id":   "category",
    "wm_yr_wk":  "int16",
    "sell_price": "float32",
}

# Long-format merged dataset dtypes
LONG_DTYPES: dict[str, str] = {
    "id":        "category",
    "item_id":   "category",
    "dept_id":   "category",
    "cat_id":    "category",
    "store_id":  "category",
    "state_id":  "category",
    "d":         "category",
    "sales":     "float32",
    "wm_yr_wk":  "int16",
    "sell_price":"float32",
    "snap_CA":   "int8",
    "snap_TX":   "int8",
    "snap_WI":   "int8",
    "wday":      "int8",
    "month":     "int8",
    "year":      "int16",
}


# ─────────────────────────────────────────────────────────────
# Ingestion Pipeline Settings
# ─────────────────────────────────────────────────────────────
# Number of day-columns to process per chunk during the melt
# Lower this on machines with < 8 GB RAM
MELT_CHUNK_SIZE: int = 300          # process 300 day-columns at a time

# Parquet compression
PARQUET_COMPRESSION: str = "snappy"

# Random seed — used throughout for reproducibility
RANDOM_SEED: int = 42


# ─────────────────────────────────────────────────────────────
# Walk-Forward Cross-Validation Splits (Phase 4)
# ─────────────────────────────────────────────────────────────
CV_TRAIN_END_DAY  = 1800
CV_VALID_START_DAY= 1801
CV_VALID_END_DAY  = 1856
CV_TEST_START_DAY = 1857
CV_TEST_END_DAY   = 1913


# ─────────────────────────────────────────────────────────────
# Inventory Optimisation Parameters (Phase 5)
# ─────────────────────────────────────────────────────────────
TARGET_SERVICE_LEVEL: float = 0.95   # → Z = 1.645
Z_SCORE_95: float = 1.645

# Assumed supplier lead time in days (can be made item-level)
LEAD_TIME_DAYS_MEAN: float = 7.0
LEAD_TIME_DAYS_STD: float  = 1.5

# Cost parameters (unit: USD)
HOLDING_COST_PER_UNIT_DAY: float = 0.05   # cost to hold 1 unit for 1 day
STOCKOUT_PENALTY_PER_UNIT: float = 2.00   # lost-sale cost per unit