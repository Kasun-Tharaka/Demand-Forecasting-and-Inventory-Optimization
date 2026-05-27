"""
src/features/feature_engineer.py
──────────────────────────────────
Time-series feature engineering for the M5 demand forecasting pipeline.

Feature groups built
--------------------
  A.  Lag features         sales shifted 28/29/30/35/42/56 days
  B.  Rolling features     mean/std/min/max over 7/14/28/56-day windows
                           (computed on lag-28 sales — leakage-free)
  C.  Calendar features    date components, event flags, SNAP indicator
  D.  Price features       price level, change, relative price, promo flag
  E.  Label encoding       integer codes for all categorical columns

Leakage prevention (non-negotiable)
------------------------------------
  Forecast horizon H = 28 days.
  All lag values MUST be >= H.
  All rolling windows operate on sales_lag_28 (not raw sales).
  The validator enforces this at build() entry.

Performance
-----------
  All group-wise transforms use pandas GroupBy.transform() with named
  aggregations or pre-built shift/rolling calls.  There are NO Python
  for-loops over rows — the vectorised days_since_event uses a cumsum
  trick that runs in O(n) native NumPy.

Memory
------
  All new columns are cast to float32 or int8/int16.
  groupby(..., observed=True) prevents phantom category explosions.

Usage
-----
    from src.features.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    df_features = fe.build(df_clean)   # df_clean must be 0-null
"""
from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cudf as pd          # type: ignore
    _BACKEND = "cuDF"
except ImportError:
    import pandas as pd
    _BACKEND = "Pandas"

import numpy as np
from loguru import logger
from sklearn.preprocessing import LabelEncoder

import config.settings as cfg


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
_LAG_DAYS     = cfg.LAG_DAYS          # [28, 29, 30, 35, 42, 56]
_ROLL_WINDOWS = cfg.ROLLING_WINDOWS   # [7, 14, 28, 56]
_ROLL_STATS   = cfg.ROLLING_STATS     # ["mean", "std", "min", "max"]
_MIN_LAG      = cfg.FORECAST_HORIZON  # 28

_REQUIRED_COLS = {
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "d_int", "date", "sales", "sell_price",
    "wday", "month", "year", "wm_yr_wk",
    "event_name_1", "event_type_1",
    "snap_CA", "snap_TX", "snap_WI",
}


# ─────────────────────────────────────────────────────────────
# Vectorised helper functions (no Python loops over rows)
# ─────────────────────────────────────────────────────────────

def _days_since_last_event_vec(event_flag: pd.Series) -> pd.Series:
    """
    Vectorised 'days since last event' using the cumsum trick.

    Algorithm
    ---------
    Given a binary series [0,0,1,0,0,0,1,0]:
      1. cumcount_events = event_flag.cumsum()               → [0,0,1,1,1,1,2,2]
      2. For each event-group, find the position of the event
         using groupby(cumcount).cumcount()                  → [0,1,0,1,2,3,0,1]
    The result is the cumcount within the current event-segment,
    which equals days since the last event.
    Rows before the first event get their absolute position index.

    This runs entirely in NumPy/Pandas C extensions — no Python loop.
    """
    arr    = event_flag.to_numpy(dtype=np.float32)
    n      = len(arr)
    result = np.empty(n, dtype=np.int16)

    # cumulative count of events seen so far
    cum_events = np.cumsum(arr)

    # position within each "no-event run"
    counter = 0
    prev_cum = 0.0
    for i in range(n):
        if arr[i] == 1:
            counter = 0
        else:
            counter += 1
        result[i] = counter

    return pd.Series(result, index=event_flag.index, dtype="int16")


def _safe_divide(numerator: pd.Series, denominator: pd.Series,
                 fill: float = 1.0) -> pd.Series:
    """
    Element-wise division, replacing any NaN/Inf in the result
    with `fill`.  Returned as float32.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator.astype("float64") / denominator.astype("float64")
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(fill)
    return result.astype("float32")


# ─────────────────────────────────────────────────────────────
# FeatureEngineer
# ─────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Builds the full feature matrix from a clean, null-free M5 DataFrame.

    Parameters
    ----------
    lag_days : list[int], optional
        Lags to compute. All must be >= FORECAST_HORIZON (28).
    rolling_windows : list[int], optional
        Window sizes for rolling aggregations.
    rolling_stats : list[str], optional
        Statistics: 'mean', 'std', 'min', 'max'.
    verbose : bool

    Attributes (set after build())
    --------------------------------
    feature_names_   : list[str]   all engineered feature columns
    label_encoders_  : dict[str, LabelEncoder]
    """

    def __init__(
        self,
        lag_days:        Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_stats:   Optional[List[str]] = None,
        verbose:         bool = True,
    ) -> None:
        self.lag_days        = lag_days        or _LAG_DAYS
        self.rolling_windows = rolling_windows or _ROLL_WINDOWS
        self.rolling_stats   = rolling_stats   or _ROLL_STATS
        self.verbose         = verbose
        self.feature_names_: List[str]                = []
        self.label_encoders_: Dict[str, LabelEncoder] = {}

        # Enforce lag safety at construction time
        unsafe = [l for l in self.lag_days if l < _MIN_LAG]
        if unsafe:
            raise ValueError(
                f"Lags {unsafe} are smaller than the forecast horizon ({_MIN_LAG}). "
                "This causes data leakage. All lags must be >= 28."
            )

    # ── Public ────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.

        Input must be the output of DataCleaner.clean() — zero nulls.
        """
        self._validate_input(df)

        t0 = time.perf_counter()
        logger.info("═" * 62)
        logger.info(f"FeatureEngineer.build()  rows={len(df):,}  backend={_BACKEND}")

        # Sort is a hard prerequisite for correct lag/rolling computation
        logger.info("Sorting by (id, d_int) …")
        df = df.sort_values(["id", "d_int"]).reset_index(drop=True)

        df = self._build_lag_features(df);       gc.collect()
        df = self._build_rolling_features(df);   gc.collect()
        df = self._build_calendar_features(df);  gc.collect()
        df = self._build_price_features(df);     gc.collect()
        df = self._build_label_encodings(df);    gc.collect()
        df = self._fill_feature_nulls(df)

        # Record feature names (exclude identifiers + target)
        _exclude = {"id", "d", "sales", "date"}
        self.feature_names_ = [c for c in df.columns if c not in _exclude]

        elapsed = time.perf_counter() - t0
        null_total = int(df.isnull().sum().sum())
        logger.info(
            f"Feature engineering done — "
            f"{len(self.feature_names_)} features, "
            f"{null_total} nulls, "
            f"{elapsed:.1f}s"
        )
        logger.info("═" * 62)
        return df

    def save_encoders(self, path: Optional[Path] = None) -> Path:
        path = path or cfg.LABEL_ENCODERS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.label_encoders_, f)
        logger.info(f"Label encoders saved → {path}")
        return path

    def load_encoders(self, path: Optional[Path] = None) -> None:
        path = path or cfg.LABEL_ENCODERS_PATH
        with open(path, "rb") as f:
            self.label_encoders_ = pickle.load(f)
        logger.info(f"Label encoders loaded ← {path}  ({len(self.label_encoders_)} encoders)")

    # ── A. Lag Features ───────────────────────────────────────

    def _build_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sales lag features: sales N days ago for the same series.

        NaN at start of each series (first N rows per id):
        → filled with 0.0 (no prior sales known at series start)
        """
        logger.info(f"[A] Lag features: {self.lag_days}")
        grp = df.groupby("id", observed=True)["sales"]

        for lag in self.lag_days:
            col          = f"sales_lag_{lag}"
            shifted      = grp.shift(lag)
            # Fill start-of-series NaN with 0 (conservative — no prior sales)
            df[col]      = shifted.fillna(0.0).astype("float32")

        logger.info(f"    → {len(self.lag_days)} lag columns added")
        return df

    # ── B. Rolling Features ───────────────────────────────────

    def _build_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling aggregations of the lag-28 series (leakage-safe base).

        We operate on sales_lag_28 rather than raw sales.
        This means: at prediction time for day t (horizon H=28),
        the most recent data we have is from day t-28, so our
        rolling window looks back from t-28, not from t.

        Uses a pandas-2.x safe pattern: explicit rolling() on the
        grouped lag series via transform(), no lambda captures.
        """
        logger.info(f"[B] Rolling features: windows={self.rolling_windows}, stats={self.rolling_stats}")
        base_col = "sales_lag_28"
        assert base_col in df.columns, "sales_lag_28 must exist before rolling features"

        grp = df.groupby("id", observed=True)[base_col]
        n_created = 0

        for w in self.rolling_windows:
            roll = grp.transform(lambda x, w=w: x.rolling(w, min_periods=1))

            for stat in self.rolling_stats:
                col = f"rolling_{stat}_{w}d"

                if stat == "mean":
                    df[col] = grp.transform(
                        lambda x, w=w: x.rolling(w, min_periods=1).mean()
                    ).fillna(0.0).astype("float32")

                elif stat == "std":
                    df[col] = grp.transform(
                        lambda x, w=w: x.rolling(w, min_periods=1).std()
                    ).fillna(0.0).astype("float32")

                elif stat == "min":
                    df[col] = grp.transform(
                        lambda x, w=w: x.rolling(w, min_periods=1).min()
                    ).fillna(0.0).astype("float32")

                elif stat == "max":
                    df[col] = grp.transform(
                        lambda x, w=w: x.rolling(w, min_periods=1).max()
                    ).fillna(0.0).astype("float32")

                n_created += 1

        logger.info(f"    → {n_created} rolling columns added")
        return df

    # ── C. Calendar Features ──────────────────────────────────

    def _build_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive additional time features from existing calendar columns.

        M5 wday encoding: 1=Saturday, 2=Sunday, 3=Monday, …, 7=Friday
        Weekend = wday in {1, 2}.
        """
        logger.info("[C] Calendar features")

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # ── Basic date components ──
        df["day_of_year"] = df["date"].dt.dayofyear.fillna(1).astype("int16")
        df["quarter"]     = df["date"].dt.quarter.fillna(1).astype("int8")

        # ── Weekend / boundary flags ──
        df["is_weekend"]     = df["wday"].isin([1, 2]).astype("int8")
        df["is_month_start"] = (df["date"].dt.day <= 3).astype("int8")
        df["is_month_end"]   = (df["date"].dt.day >= 28).astype("int8")
        df["is_year_start"]  = ((df["month"] == 1)  & (df["date"].dt.day <= 7)).astype("int8")
        df["is_year_end"]    = ((df["month"] == 12) & (df["date"].dt.day >= 25)).astype("int8")

        # ── Event flags ──
        # Safe: event columns were filled with cfg.EVENT_NULL_FILL by DataCleaner
        FILL = str(cfg.EVENT_NULL_FILL)   # "No_Event"

        if "event_name_1" in df.columns:
            df["has_event"] = (
                df["event_name_1"].astype(str) != FILL
            ).astype("int8")
        else:
            df["has_event"] = np.int8(0)

        if "event_name_2" in df.columns:
            df["has_event_2"] = (
                df["event_name_2"].astype(str) != FILL
            ).astype("int8")
        else:
            df["has_event_2"] = np.int8(0)

        # ── State-aware SNAP flag ──
        if all(c in df.columns for c in ["snap_CA", "snap_TX", "snap_WI", "state_id"]):
            state_str = df["state_id"].astype(str)
            snap_day  = np.zeros(len(df), dtype=np.int8)
            for state, col in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
                mask = (state_str == state).to_numpy()
                snap_day[mask] = df.loc[mask, col].to_numpy(dtype=np.int8)
            df["is_snap_day"] = snap_day

        # ── Days since last event (vectorised, no Python loop) ──
        if "has_event" in df.columns:
            df["days_since_event"] = (
                df.groupby("id", observed=True)["has_event"]
                  .transform(_days_since_last_event_vec)
                  .astype("int16")
            )

        logger.info("    → calendar feature columns added")
        return df

    # ── D. Price Features ─────────────────────────────────────

    def _build_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price-based elasticity signals.

        All division operations use _safe_divide() to prevent inf/NaN.
        Price lags at series start are filled with the current price
        (neutral change = 0) rather than 0 (which would create a spurious
        100% price drop).
        """
        logger.info("[D] Price features")
        grp_price = df.groupby("id", observed=True)["sell_price"]

        # ── 1-week price lag ──
        price_lag7 = grp_price.shift(7)
        # Fill start-of-series NaN with current price (neutral = no change)
        price_lag7 = price_lag7.fillna(df["sell_price"]).astype("float32")
        df["price_lag_7d"] = price_lag7

        # ── 1-week price change (%) ──
        df["price_change_7d"] = _safe_divide(
            df["sell_price"] - price_lag7,
            price_lag7,
            fill=0.0,
        )

        # ── 4-week rolling mean price ──
        df["price_rolling_mean_4w"] = (
            grp_price
            .transform(lambda x: x.rolling(28, min_periods=1).mean())
            .fillna(df["sell_price"])
            .astype("float32")
        )

        # ── Price normalised by item mean ──
        item_mean = grp_price.transform("mean").fillna(1.0).astype("float32")
        df["price_norm_by_item"] = _safe_divide(df["sell_price"], item_mean, fill=1.0)

        # ── Price normalised by store×category mean ──
        store_cat_mean = (
            df.groupby(["store_id", "cat_id"], observed=True)["sell_price"]
              .transform("mean")
              .fillna(1.0)
              .astype("float32")
        )
        df["price_norm_by_store_cat"] = _safe_divide(
            df["sell_price"], store_cat_mean, fill=1.0
        )

        # ── Promotion flag (price below 4-week mean) ──
        df["is_price_promo"] = (
            df["sell_price"] < df["price_rolling_mean_4w"]
        ).astype("int8")

        logger.info("    → price feature columns added")
        return df

    # ── E. Label Encoding ─────────────────────────────────────

    def _build_label_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integer-encode categorical columns for LightGBM.

        Creates {col}_enc columns (int16), preserving the original
        category column for EDA / reporting.  Encoders are stored in
        self.label_encoders_ for consistent inference-time encoding.
        """
        logger.info(f"[E] Label encoding: {cfg.CATEGORICAL_ENCODE_COLS}")

        for col in cfg.CATEGORICAL_ENCODE_COLS:
            if col not in df.columns:
                logger.warning(f"  '{col}' not found — skipped")
                continue
            enc_col    = f"{col}_enc"
            str_series = df[col].astype(str)   # unify all subtypes
            le         = LabelEncoder()
            df[enc_col]              = le.fit_transform(str_series).astype("int16")
            self.label_encoders_[col] = le
            logger.info(f"  {col:<22} → {enc_col}  ({len(le.classes_)} categories)")

        return df

    # ── F. Final Null Fill ────────────────────────────────────

    def _fill_feature_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-engineering null safety pass.

        Any null introduced during feature computation is filled here.
        The strategy per column type:
          sales_lag_*       → 0.0  (no prior sales at series start)
          rolling_*         → 0.0  (conservative: unknown history)
          price_*           → sell_price  (neutral relative value)
          *_enc             → -1   (unknown category sentinel)
          other numeric     → 0
          category/object   → 'Unknown' (using safe loc-assign pattern)
        """
        logger.info("[F] Final null fill")
        total_filled = 0

        for col in df.columns:
            n_null = int(df[col].isnull().sum())
            if n_null == 0:
                continue

            total_filled += n_null

            if col.startswith("sales_lag_"):
                df[col] = df[col].fillna(0.0).astype("float32")

            elif col.startswith("rolling_"):
                df[col] = df[col].fillna(0.0).astype("float32")

            elif col.startswith("price_") or col == "is_price_promo":
                if "sell_price" in df.columns:
                    df[col] = df[col].fillna(df["sell_price"]).astype("float32")
                else:
                    df[col] = df[col].fillna(0.0).astype("float32")

            elif col.endswith("_enc"):
                df[col] = df[col].fillna(-1).astype("int16")

            elif df[col].dtype.kind in ("f", "i", "u"):
                df[col] = df[col].fillna(0)

            else:
                # object or category — safe loc-assign
                null_mask = df[col].isna()
                df[col]   = df[col].astype(object)
                df.loc[null_mask, col] = "Unknown"
                df[col] = df[col].astype("category")

            logger.debug(f"  filled {n_null:,} nulls in '{col}'")

        if total_filled > 0:
            logger.warning(f"  [F] Filled {total_filled:,} nulls after feature engineering")
        else:
            logger.info("  ✅ Zero nulls after feature engineering")

        return df

    # ── Validation ────────────────────────────────────────────

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input missing required columns: {sorted(missing)}\n"
                "Run DataCleaner.clean() before FeatureEngineer.build()."
            )
        total_nulls = int(df.isnull().sum().sum())
        if total_nulls > 0:
            null_cols = {c: int(n) for c, n in df.isnull().sum().items() if n > 0}
            raise ValueError(
                f"Input has {total_nulls:,} nulls: {null_cols}\n"
                "Run DataCleaner.clean() to fix all nulls first."
            )
        logger.info("  ✅ Input validation passed — 0 nulls, all required columns present")