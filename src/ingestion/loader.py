"""
src/ingestion/loader.py
───────────────────────
Memory-optimised raw CSV reader for the M5 dataset.

Design principles
-----------------
1.  **Explicit dtype on read** — never let Pandas infer float64 for data
    that fits in float32 or int8. Inference is the #1 RAM killer.
2.  **Selective column loading** — only load the columns required by the
    next pipeline stage (usecols argument).
3.  **RAM audit before / after** — log exactly how much memory was saved
    so the result is reportable in a portfolio README.
4.  **GPU-aware** — if RAPIDS cuDF is installed and a CUDA device is
    available the functions fall back gracefully to cuDF; otherwise plain
    Pandas is used.  Swap the import once at the top and every downstream
    call is accelerated for free.

Usage
-----
    from src.ingestion.loader import M5DataLoader
    loader = M5DataLoader()
    calendar = loader.load_calendar()
    prices   = loader.load_prices()
    sales_meta, day_cols = loader.load_sales_meta_and_days()
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import psutil

try:
    import cudf as pd          # type: ignore  # GPU path
    _BACKEND = "cuDF (GPU)"
except ImportError:
    import pandas as pd        # CPU path
    _BACKEND = "Pandas (CPU)"

from loguru import logger

import config.settings as cfg


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _ram_mb() -> float:
    """Return current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 ** 2


def _df_mb(df: "pd.DataFrame") -> float:
    """Return in-memory size of a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def _log_memory_saved(label: str, before_mb: float, after_mb: float) -> None:
    saved = before_mb - after_mb
    pct   = (saved / before_mb * 100) if before_mb > 0 else 0
    logger.info(
        f"[{label}] Memory: {before_mb:.1f} MB → {after_mb:.1f} MB  "
        f"(saved {saved:.1f} MB / {pct:.0f}%)"
    )


def downcast_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Iterates every column and downcasts numeric types to the smallest
    safe representation.

    - float64 → float32   (sufficient for sales / price data)
    - int64   → smallest signed int that fits (int8 / int16 / int32)
    - object  → category  (when unique ratio < 50 %)

    Returns the modified DataFrame (in-place mutation + return).
    """
    before = _df_mb(df)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "float64":
            df[col] = df[col].astype("float32")

        elif col_type in ("int64", "int32"):
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype("int8")
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype("int16")
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype("int32")

        elif col_type == "object":
            # Convert high-cardinality strings to category
            n_unique = df[col].nunique()
            if n_unique / len(df) < 0.50:
                df[col] = df[col].astype("category")

    after = _df_mb(df)
    _log_memory_saved("downcast", before, after)
    return df


# ─────────────────────────────────────────────────────────────
# Main Loader Class
# ─────────────────────────────────────────────────────────────

class M5DataLoader:
    """
    Encapsulates all raw-CSV reading for the M5 dataset.

    Parameters
    ----------
    data_dir : Path | str, optional
        Override the raw data directory from settings.
    verbose : bool
        Print per-operation memory logs.

    Examples
    --------
    >>> loader = M5DataLoader()
    >>> cal = loader.load_calendar()
    >>> cal.dtypes
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else cfg.DATA_RAW
        self.verbose  = verbose
        logger.info(f"M5DataLoader initialised | backend={_BACKEND} | data_dir={self.data_dir}")
        self._validate_data_dir()

    # ── private ──────────────────────────────────────────────

    def _validate_data_dir(self) -> None:
        """Warn if expected raw files are missing."""
        expected = {
            "calendar.csv":               cfg.RAW_CALENDAR,
            "sell_prices.csv":            cfg.RAW_PRICES,
            "sales_train_validation.csv": cfg.RAW_SALES,
        }
        for name, path in expected.items():
            if not path.exists():
                logger.warning(f"Raw file not found: {path}  — download from Kaggle.")

    def _read_csv(
        self,
        path: Path,
        dtype: Optional[dict] = None,
        usecols: Optional[list] = None,
        **kwargs,
    ) -> "pd.DataFrame":
        """Thin wrapper around pd.read_csv with timing and RAM logging."""
        ram_before = _ram_mb()
        t0 = time.perf_counter()

        df = pd.read_csv(path, dtype=dtype, usecols=usecols, **kwargs)

        elapsed = time.perf_counter() - t0
        ram_after = _ram_mb()

        logger.info(
            f"Loaded {path.name:<45} "
            f"shape={df.shape}  "
            f"size={_df_mb(df):.1f} MB  "
            f"time={elapsed:.2f}s  "
            f"RAM Δ={ram_after - ram_before:+.1f} MB"
        )
        return df

    # ── public loaders ───────────────────────────────────────

    def load_calendar(self) -> "pd.DataFrame":
        """
        Load calendar.csv with pre-specified dtypes.

        Returns
        -------
        pd.DataFrame
            Columns: date, wm_yr_wk, weekday, wday, month, year,
                     d, event_name_1/2, event_type_1/2, snap_CA/TX/WI
        """
        df = self._read_csv(
            cfg.RAW_CALENDAR,
            dtype=cfg.CALENDAR_DTYPES,
            parse_dates=["date"],
        )
        # 'd' column: strip prefix and store as int16 for fast joins
        df["d_int"] = df["d"].str.replace("d_", "", regex=False).astype("int16")
        return df

    def load_prices(self) -> "pd.DataFrame":
        """
        Load sell_prices.csv with pre-specified dtypes.

        Returns
        -------
        pd.DataFrame
            Columns: store_id, item_id, wm_yr_wk, sell_price
        """
        df = self._read_csv(
            cfg.RAW_PRICES,
            dtype=cfg.PRICE_DTYPES,
        )
        return df

    def load_sales_meta(self) -> "pd.DataFrame":
        """
        Load only the identifier columns from sales_train_validation.csv
        (i.e. NOT the d_1..d_1913 day columns).

        Use this when you need to inspect the item hierarchy without
        pulling the full time-series into RAM.

        Returns
        -------
        pd.DataFrame  shape: (30490, 6)
        """
        df = self._read_csv(
            cfg.RAW_SALES,
            usecols=cfg.ID_COLS,
            dtype=cfg.SALES_META_DTYPES,
        )
        return df

    def load_sales_wide(self, day_range: Optional[Tuple[int, int]] = None) -> "pd.DataFrame":
        """
        Load the full wide-format sales table.

        Parameters
        ----------
        day_range : (start_day, end_day), optional
            Load only columns d_{start} to d_{end} (inclusive).
            Useful for testing on a subset without loading all 1913 days.
            E.g. day_range=(1, 300) loads the first 300 days.

        Returns
        -------
        pd.DataFrame  shape: (30490, 6 + n_days)
        """
        # Build the list of day columns to load
        if day_range is not None:
            start, end = day_range
            day_cols = [f"d_{i}" for i in range(start, end + 1)]
        else:
            day_cols = [f"d_{i}" for i in range(1, cfg.N_TRAIN_DAYS + 1)]

        usecols = cfg.ID_COLS + day_cols

        df = self._read_csv(
            cfg.RAW_SALES,
            usecols=usecols,
            dtype={**cfg.SALES_META_DTYPES, **{c: "float32" for c in day_cols}},
        )
        return df

    def get_day_columns(self) -> list[str]:
        """Return the full list of day column names ['d_1', ..., 'd_1913']."""
        return [f"d_{i}" for i in range(1, cfg.N_TRAIN_DAYS + 1)]

    def memory_report(self) -> None:
        """Print a summary of estimated raw-file sizes and current process RAM."""
        logger.info("=" * 60)
        logger.info(f"Backend              : {_BACKEND}")
        logger.info(f"Process RSS (current): {_ram_mb():.1f} MB")
        for label, path in [
            ("calendar.csv",               cfg.RAW_CALENDAR),
            ("sell_prices.csv",            cfg.RAW_PRICES),
            ("sales_train_validation.csv", cfg.RAW_SALES),
        ]:
            if path.exists():
                size_mb = path.stat().st_size / 1024 ** 2
                logger.info(f"{label:<40}: {size_mb:.1f} MB on disk")
            else:
                logger.warning(f"{label:<40}: NOT FOUND")
        logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────
# Quick-run entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = M5DataLoader()
    loader.memory_report()

    cal    = loader.load_calendar()
    prices = loader.load_prices()
    meta   = loader.load_sales_meta()

    print("\n── Calendar sample ──")
    print(cal.head(3).to_string())
    print(f"\nCalendar dtypes:\n{cal.dtypes}")

    print("\n── Prices sample ──")
    print(prices.head(3).to_string())

    print("\n── Sales meta sample ──")
    print(meta.head(3).to_string())

    gc.collect()