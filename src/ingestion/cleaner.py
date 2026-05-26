"""
src/ingestion/cleaner.py
─────────────────────────
Comprehensive, null-guaranteed data cleaning for the M5 merged dataset.

Null root-causes in M5 and their exact fix
-------------------------------------------
  Column             Root cause                     Fix
  ─────────────────  ─────────────────────────────  ──────────────────────────
  sell_price         Items not yet in assortment    groupby ffill+bfill → median
  event_name/type    Most days have no event        fill "No_Event" (robust cat)
  sales              Melt artefact on no-sale days  fill 0.0
  snap_CA/TX/WI      Calendar left-join miss        fill 0
  wday/month/year    Calendar left-join miss        fill mode default
  weekday            Calendar left-join miss        fill "Monday" → category
  wm_yr_wk           Calendar left-join miss        ffill+bfill → int16
  d_int              Missing computation            recompute from d column
  catch-all numeric  Any remaining                  fill 0
  catch-all cat/obj  Any remaining                  fill "Unknown"

Design rules
------------
1.  NEVER use .replace({"nan": ...}) on category columns — it silently fails
    in pandas 2.x when "nan" is not a known category member.
2.  ALWAYS use isna()-mask + loc-assignment. This works regardless of dtype.
3.  After every fix step, cast to the target dtype AFTER filling.
4.  A final assertion at the end proves zero nulls remain — no silent failures.

Usage
-----
    from src.ingestion.cleaner import DataCleaner
    cleaner  = DataCleaner()
    df_clean = cleaner.clean(df_merged)   # raises RuntimeError if nulls remain
    cleaner.print_audit()
"""
from __future__ import annotations

import gc
import time
from typing import Dict, List, Tuple

try:
    import cudf as pd          # type: ignore
except ImportError:
    import pandas as pd

import numpy as np
from loguru import logger

import config.settings as cfg


# ─────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────

def _null_map(df: pd.DataFrame) -> Dict[str, int]:
    counts = df.isnull().sum()
    return {col: int(cnt) for col, cnt in counts.items() if cnt > 0}


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.3f}%" if total > 0 else "0.000%"


# ─────────────────────────────────────────────────────────────
# DataCleaner
# ─────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Cleans the M5 merged long-format DataFrame so it is
    100% null-free before feature engineering.

    Every fix method follows the same safe pattern:
      1. Compute null mask BEFORE any dtype change.
      2. Cast column to object/float to remove dtype restrictions.
      3. Assign fill value via df.loc[mask, col] = value.
      4. Cast column back to the target memory-efficient dtype.
      5. Run a post-fix null count and log it.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose   = verbose
        self._audit: List[Dict] = []

    # ── Public ────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(df)
        t0     = time.perf_counter()

        logger.info("═" * 62)
        logger.info(f"DataCleaner.clean()  rows={n_rows:,}  cols={len(df.columns)}")

        nulls_before  = _null_map(df)
        total_before  = sum(nulls_before.values())
        logger.info(f"Nulls BEFORE: {total_before:,} across {len(nulls_before)} columns")
        for col, cnt in sorted(nulls_before.items(), key=lambda x: -x[1]):
            logger.info(f"   {col:<30} {cnt:>10,}  ({_pct(cnt, n_rows)})")

        logger.info("─" * 62)

        # ── Cleaning steps (order is significant) ─────────────
        df = self._step_sort(df)
        df = self._step_d_int(df)
        df = self._step_sales(df)
        df = self._step_event_columns(df)
        df = self._step_sell_price(df)
        df = self._step_snap(df)
        df = self._step_calendar_ints(df)
        df = self._step_weekday(df)
        df = self._step_wm_yr_wk(df)
        df = self._step_catch_all_numeric(df)
        df = self._step_catch_all_categorical(df)

        # ── Final guarantee ───────────────────────────────────
        nulls_after = _null_map(df)
        total_after = sum(nulls_after.values())
        elapsed     = time.perf_counter() - t0

        logger.info("─" * 62)
        if total_after == 0:
            logger.info(f"✅ DataCleaner done — 0 nulls remain  ({elapsed:.1f}s)")
        else:
            for col, cnt in nulls_after.items():
                logger.error(f"  STILL NULL  {col}: {cnt:,}")
            raise RuntimeError(
                f"DataCleaner failed: {total_after:,} nulls remain in "
                f"{list(nulls_after.keys())}. "
                "Check the logs above for the offending step."
            )
        logger.info("═" * 62)
        return df

    def print_audit(self) -> None:
        logger.info("═" * 62)
        logger.info("DataCleaner Audit")
        logger.info("─" * 62)
        for s in self._audit:
            icon = "🔧" if s["fixed"] > 0 else "✓ "
            logger.info(
                f"  {icon}  {s['step']:<35} "
                f"fixed={s['fixed']:>8,}  "
                f"how={s['how']}"
            )
        logger.info("═" * 62)

    # ── Step implementations ──────────────────────────────────

    def _step_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by (id, d_int) — prerequisite for all group-wise ops."""
        by = ["id", "d_int"] if "d_int" in df.columns else ["id", "d"]
        df = df.sort_values(by).reset_index(drop=True)
        self._record("sort_by_id_d", 0, f"sort({by})")
        return df

    def _step_d_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure d_int exists and is int16 with no nulls."""
        if "d_int" not in df.columns:
            df["d_int"] = (
                df["d"].astype(str)
                       .str.replace("d_", "", regex=False)
                       .astype("int16")
            )
            self._record("d_int_create", len(df), "computed_from_d_col")
            return df

        mask   = df["d_int"].isna()
        n_null = int(mask.sum())
        if n_null > 0:
            # Recompute from the 'd' string column at null positions
            df.loc[mask, "d_int"] = (
                df.loc[mask, "d"].astype(str)
                  .str.replace("d_", "", regex=False)
                  .astype(float)
            )
        df["d_int"] = pd.to_numeric(df["d_int"], errors="coerce").fillna(0).astype("int16")
        self._record("d_int_fix", n_null, "recomputed_from_d" if n_null else "ok")
        return df

    def _step_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Null sales → 0.0  (no sale on that day).
        Negative sales → 0.0  (data-quality guard).
        """
        if "sales" not in df.columns:
            self._record("sales_fix", 0, "col_absent")
            return df

        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        mask_null = df["sales"].isna()
        mask_neg  = df["sales"] < 0
        n_null    = int(mask_null.sum())
        n_neg     = int(mask_neg.sum())

        if n_null > 0:
            df.loc[mask_null, "sales"] = 0.0
        if n_neg > 0:
            df.loc[mask_neg, "sales"] = 0.0
            logger.warning(f"  [sales] clipped {n_neg:,} negative values → 0")

        df["sales"] = df["sales"].astype("float32")
        self._record("sales_fix", n_null + n_neg, "fillna(0)+clip_neg→float32")
        return df

    def _step_event_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robust event-column null fill for pandas 2.x.

        The ONLY safe strategy for category columns with NaN:
          1. isna() mask BEFORE any dtype conversion.
          2. astype(object) to unbox the category.
          3. df.loc[mask, col] = FILL_VALUE  (direct position assignment).
          4. Additional sweep for "nan"/"None" strings left by astype(object).
          5. astype("category") to restore memory efficiency.
          6. Post-fix verification — if nulls remain, use cat.add_categories nuclear option.
        """
        FILL   = cfg.EVENT_NULL_FILL     # "No_Event"
        cols   = [c for c in ["event_name_1", "event_type_1",
                               "event_name_2", "event_type_2"] if c in df.columns]
        total  = 0

        for col in cols:
            # Step 1: null mask (works on any dtype)
            null_mask = df[col].isna()
            n_null    = int(null_mask.sum())

            # Step 2: unbox to plain Python objects
            df[col] = df[col].astype(object)

            # Step 3: fill NaN positions
            if n_null > 0:
                df.loc[null_mask, col] = FILL

            # Step 4: clean residual nan-strings produced by astype(object)
            str_nans = df[col].isin(
                [None, float("nan"), "nan", "NaN", "None", "<NA>", ""]
            )
            n_str = int(str_nans.sum())
            if n_str > 0:
                df.loc[str_nans, col] = FILL

            # Step 5: re-cast to category
            df[col] = df[col].astype("category")

            # Step 6: nuclear fallback — add category then assign
            still_null = int(df[col].isna().sum())
            if still_null > 0:
                if FILL not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([FILL])
                df.loc[df[col].isna(), col] = FILL
                df[col] = df[col].cat.remove_unused_categories()

            total += n_null
            self._record(f"event_{col}", n_null, "isna_loc_assign→category")

        return df

    def _step_sell_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        sell_price NaN: group-level ffill+bfill, then global median fallback.

        Why NaN exists: new items may not have a listed price for early days.
        Strategy: forward-fill within (id), then backward-fill for items that
        only have price at the end. Items with ZERO price rows get the global median.
        """
        if "sell_price" not in df.columns:
            self._record("sell_price_fix", 0, "col_absent")
            return df

        df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce").astype("float32")
        n_before = int(df["sell_price"].isna().sum())

        if n_before == 0:
            self._record("sell_price_fix", 0, "ok")
            return df

        # Group-level fill: ffill then bfill per item series
        df["sell_price"] = (
            df.groupby("id", observed=True)["sell_price"]
              .transform(lambda s: s.ffill().bfill())
        )

        n_after = int(df["sell_price"].isna().sum())
        if n_after > 0:
            # Items that have NO price at any day → global median
            global_median = float(df["sell_price"].median())
            if np.isnan(global_median) or global_median <= 0:
                global_median = 1.0
            still_null = df["sell_price"].isna()
            df.loc[still_null, "sell_price"] = global_median
            logger.warning(
                f"  [sell_price] {int(still_null.sum()):,} rows had no price "
                f"for their item → filled with global median={global_median:.2f}"
            )

        # Clip extreme outlier prices (99.99th percentile)
        p_hi  = float(df["sell_price"].quantile(0.9999))
        n_clip = int((df["sell_price"] > p_hi).sum())
        if n_clip > 0:
            df.loc[df["sell_price"] > p_hi, "sell_price"] = p_hi

        # Ensure no zero prices (avoid div-by-zero downstream)
        df.loc[df["sell_price"] <= 0, "sell_price"] = 0.01
        df["sell_price"] = df["sell_price"].astype("float32")

        self._record("sell_price_fix", n_before,
                     f"groupby_ffill_bfill+median_fallback+clip_p9999({n_clip})")
        return df

    def _step_snap(self, df: pd.DataFrame) -> pd.DataFrame:
        """snap_CA/TX/WI nulls → 0 (no SNAP benefit day)."""
        snap_cols = [c for c in ["snap_CA", "snap_TX", "snap_WI"] if c in df.columns]
        total = 0
        for col in snap_cols:
            mask   = df[col].isna()
            n_null = int(mask.sum())
            if n_null > 0:
                df.loc[mask, col] = 0
            df[col] = df[col].astype("int8")
            total += n_null
        self._record("snap_fix", total, "fillna(0)→int8")
        return df

    def _step_calendar_ints(self, df: pd.DataFrame) -> pd.DataFrame:
        """wday / month / year: fill with safe defaults, cast to int."""
        defs = {"wday": ("int8", 1), "month": ("int8", 1), "year": ("int16", 2011)}
        total = 0
        for col, (dtype, default) in defs.items():
            if col not in df.columns:
                continue
            mask   = df[col].isna()
            n_null = int(mask.sum())
            if n_null > 0:
                df.loc[mask, col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(dtype)
            total += n_null
        self._record("calendar_ints_fix", total, "loc_assign(default)→downcast")
        return df

    def _step_weekday(self, df: pd.DataFrame) -> pd.DataFrame:
        """weekday string column: fill with 'Monday', keep as category."""
        if "weekday" not in df.columns:
            self._record("weekday_fix", 0, "col_absent")
            return df
        null_mask = df["weekday"].isna()
        n_null    = int(null_mask.sum())
        # Unbox to object first (safe for any pandas version)
        df["weekday"] = df["weekday"].astype(object)
        if n_null > 0:
            df.loc[null_mask, "weekday"] = "Monday"
        # Clean residual nan-strings
        str_nans = df["weekday"].isin([None, "nan", "None", "NaN", "<NA>", ""])
        if int(str_nans.sum()) > 0:
            df.loc[str_nans, "weekday"] = "Monday"
        df["weekday"] = df["weekday"].astype("category")
        self._record("weekday_fix", n_null, "loc_assign('Monday')→category")
        return df

    def _step_wm_yr_wk(self, df: pd.DataFrame) -> pd.DataFrame:
        """wm_yr_wk: ffill+bfill, then absolute fallback."""
        if "wm_yr_wk" not in df.columns:
            self._record("wm_yr_wk_fix", 0, "col_absent")
            return df
        n_null = int(df["wm_yr_wk"].isna().sum())
        if n_null > 0:
            df["wm_yr_wk"] = df["wm_yr_wk"].ffill().bfill()
            still = int(df["wm_yr_wk"].isna().sum())
            if still > 0:
                df.loc[df["wm_yr_wk"].isna(), "wm_yr_wk"] = 11101
        df["wm_yr_wk"] = pd.to_numeric(df["wm_yr_wk"], errors="coerce").fillna(11101).astype("int16")
        self._record("wm_yr_wk_fix", n_null, "ffill_bfill+fallback→int16")
        return df

    def _step_catch_all_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safety net: zero-fill any remaining numeric nulls."""
        num_cols    = df.select_dtypes(include="number").columns
        total_fixed = 0
        fixed_cols  = []
        for col in num_cols:
            n = int(df[col].isna().sum())
            if n > 0:
                df[col] = df[col].fillna(0)
                total_fixed += n
                fixed_cols.append(f"{col}({n})")
        if fixed_cols:
            logger.warning(f"  [catch_all_numeric] filled: {fixed_cols}")
        self._record("catch_all_numeric", total_fixed, "fillna(0)")
        return df

    def _step_catch_all_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safety net: fill remaining object/category nulls with 'Unknown'.
        Uses the same isna-mask + loc-assignment pattern for safety.
        """
        FILL     = "Unknown"
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        total    = 0
        fixed    = []

        for col in cat_cols:
            null_mask = df[col].isna()
            n_null    = int(null_mask.sum())
            if n_null == 0:
                continue

            df[col] = df[col].astype(object)
            df.loc[null_mask, col] = FILL
            str_nans = df[col].isin([None, "nan", "None", "NaN", "<NA>", ""])
            if int(str_nans.sum()) > 0:
                df.loc[str_nans, col] = FILL
            df[col] = df[col].astype("category")
            total += n_null
            fixed.append(f"{col}({n_null})")

        if fixed:
            logger.warning(f"  [catch_all_categorical] filled: {fixed}")
        self._record("catch_all_categorical", total, f"loc_assign('{FILL}')→category")
        return df

    # ── Internal helpers ──────────────────────────────────────

    def _record(self, step: str, fixed: int, how: str) -> None:
        self._audit.append({"step": step, "fixed": fixed, "how": how})
        if self.verbose:
            icon = "  🔧" if fixed > 0 else "  ✓ "
            logger.info(f"{icon}  [{step:<30}]  fixed={fixed:>8,}  how={how}")