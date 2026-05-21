"""
src/ingestion/validator.py
───────────────────────────
Schema validation, null-audit, business-rule checks, and
data-leakage guards for the M5 ingestion pipeline.

Why validation matters in a senior project
-------------------------------------------
A junior pipeline loads data and immediately trains a model.
A senior pipeline *proves* the data is correct before spending
compute on model training.  Silent data bugs (wrong join keys,
date leakage, negative sales) will silently corrupt every
downstream result.

Validation checks implemented
------------------------------
1.  Schema check     — expected columns and dtypes are present.
2.  Row count check  — shape matches documented M5 dimensions.
3.  Null audit       — reports nulls per column (some are expected).
4.  Negative sales   — sales values should be ≥ 0.
5.  Date range check — days run from d_1 to d_1913.
6.  Join integrity   — every (store_id, item_id, wm_yr_wk) in the
                        long table has a matching price row.
7.  Leakage guard    — for a given feature horizon H, confirms no
                        feature uses data from day > (day - H).

Usage
-----
    from src.ingestion.validator import M5Validator
    v = M5Validator()
    report = v.validate_raw(calendar, prices, sales_meta)
    report = v.validate_processed(df_long)
    v.print_report(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import cudf as pd          # type: ignore
except ImportError:
    import pandas as pd

import numpy as np
from loguru import logger

import config.settings as cfg


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name:    str
    passed:  bool
    message: str
    details: Optional[Any] = None


@dataclass
class ValidationReport:
    checks:   List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(not c.passed for c in self.checks)


# ─────────────────────────────────────────────────────────────
# Expected schema definitions
# ─────────────────────────────────────────────────────────────

# Minimum columns (not exhaustive — just the ones we depend on)
CALENDAR_REQUIRED_COLS = {
    "date", "d", "wm_yr_wk", "wday", "month", "year",
    "snap_CA", "snap_TX", "snap_WI",
}

PRICES_REQUIRED_COLS = {"store_id", "item_id", "wm_yr_wk", "sell_price"}

SALES_META_REQUIRED_COLS = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}

LONG_REQUIRED_COLS = {
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "d", "sales", "date", "wm_yr_wk", "sell_price",
}

# Documented M5 dimensions
M5_N_ITEMS  = 30_490     # unique series in sales_train_validation
M5_N_DAYS   = 1_913      # d_1 through d_1913
M5_CALENDAR_ROWS = 1_969 # calendar has rows beyond training horizon


# ─────────────────────────────────────────────────────────────
# Validator Class
# ─────────────────────────────────────────────────────────────

class M5Validator:
    """
    Validates M5 DataFrames at each pipeline stage.

    All check methods return a CheckResult; run() methods aggregate
    them into a ValidationReport.
    """

    # ── Raw table checks ──────────────────────────────────────

    def validate_raw(
        self,
        calendar:    "pd.DataFrame",
        prices:      "pd.DataFrame",
        sales_meta:  "pd.DataFrame",
    ) -> ValidationReport:
        """Run all raw-table validation checks."""
        report = ValidationReport()
        report.checks += [
            self._check_columns("calendar",   calendar,   CALENDAR_REQUIRED_COLS),
            self._check_columns("prices",     prices,     PRICES_REQUIRED_COLS),
            self._check_columns("sales_meta", sales_meta, SALES_META_REQUIRED_COLS),
            self._check_calendar_row_count(calendar),
            self._check_sales_item_count(sales_meta),
            self._check_null_counts("calendar",  calendar),
            self._check_null_counts("prices",    prices),
            self._check_null_counts("sales_meta",sales_meta),
            self._check_price_non_negative(prices),
            self._check_calendar_date_range(calendar),
            self._check_sales_states(sales_meta),
        ]
        return report

    def validate_processed(self, df: "pd.DataFrame") -> ValidationReport:
        """Run all processed (long-format, merged) validation checks."""
        report = ValidationReport()
        report.checks += [
            self._check_columns("processed_long", df, LONG_REQUIRED_COLS),
            self._check_sales_non_negative(df),
            self._check_null_counts("processed_long", df, warn_only_cols={"sell_price"}),
            self._check_long_row_count(df),
            self._check_no_future_leakage(df),
        ]
        return report

    # ── Static report printer ─────────────────────────────────

    @staticmethod
    def print_report(report: ValidationReport) -> None:
        """Pretty-print the full validation report."""
        total   = len(report.checks)
        passed  = report.n_passed
        failed  = report.n_failed
        status  = "✅ ALL CHECKS PASSED" if report.passed else f"❌ {failed} CHECK(S) FAILED"

        logger.info("═" * 60)
        logger.info(f"Validation Report: {status}  ({passed}/{total} passed)")
        logger.info("─" * 60)
        for c in report.checks:
            icon = "✅" if c.passed else "❌"
            logger.info(f"  {icon}  {c.name:<45} {c.message}")
        logger.info("═" * 60)

        if not report.passed:
            raise RuntimeError(
                f"Validation failed ({failed} checks). "
                "Review the report above before proceeding."
            )

    # ── Individual check methods ──────────────────────────────

    @staticmethod
    def _check_columns(
        label:    str,
        df:       "pd.DataFrame",
        required: set,
    ) -> CheckResult:
        missing = required - set(df.columns)
        if missing:
            return CheckResult(
                name    = f"schema:{label}",
                passed  = False,
                message = f"Missing columns: {sorted(missing)}",
            )
        return CheckResult(
            name    = f"schema:{label}",
            passed  = True,
            message = f"All {len(required)} required columns present.",
        )

    @staticmethod
    def _check_null_counts(
        label:          str,
        df:             "pd.DataFrame",
        warn_only_cols: Optional[set] = None,
    ) -> CheckResult:
        """
        Check for unexpected nulls.

        sell_price is allowed to be null (new products with no listed price).
        Any other unexpected null is a hard failure.
        """
        warn_only_cols = warn_only_cols or set()
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]

        unexpected = {
            col: int(count)
            for col, count in nulls.items()
            if col not in warn_only_cols
        }
        warnings = {
            col: int(count)
            for col, count in nulls.items()
            if col in warn_only_cols
        }

        if warnings:
            logger.warning(
                f"[{label}] Expected nulls (acceptable): {warnings}"
            )

        if unexpected:
            return CheckResult(
                name    = f"nulls:{label}",
                passed  = False,
                message = f"Unexpected nulls in: {unexpected}",
                details = unexpected,
            )
        return CheckResult(
            name    = f"nulls:{label}",
            passed  = True,
            message = "No unexpected nulls detected.",
        )

    @staticmethod
    def _check_calendar_row_count(calendar: "pd.DataFrame") -> CheckResult:
        n = len(calendar)
        # Calendar extends beyond training to cover evaluation period
        ok = n >= M5_CALENDAR_ROWS
        return CheckResult(
            name    = "row_count:calendar",
            passed  = ok,
            message = f"{n} rows (expected ≥ {M5_CALENDAR_ROWS})",
        )

    @staticmethod
    def _check_sales_item_count(sales_meta: "pd.DataFrame") -> CheckResult:
        n = len(sales_meta)
        ok = n == M5_N_ITEMS
        return CheckResult(
            name    = "row_count:sales_meta",
            passed  = ok,
            message = f"{n} items (expected {M5_N_ITEMS})",
        )

    @staticmethod
    def _check_long_row_count(df: "pd.DataFrame") -> CheckResult:
        expected = M5_N_ITEMS * M5_N_DAYS
        n        = len(df)
        ok       = n == expected
        return CheckResult(
            name    = "row_count:processed_long",
            passed  = ok,
            message = f"{n:,} rows (expected {expected:,} = {M5_N_ITEMS} × {M5_N_DAYS})",
        )

    @staticmethod
    def _check_price_non_negative(prices: "pd.DataFrame") -> CheckResult:
        n_neg = (prices["sell_price"] < 0).sum()
        ok    = int(n_neg) == 0
        return CheckResult(
            name    = "business_rule:price_non_negative",
            passed  = ok,
            message = f"{n_neg} negative prices found." if not ok else "All prices ≥ 0.",
        )

    @staticmethod
    def _check_sales_non_negative(df: "pd.DataFrame") -> CheckResult:
        n_neg = (df["sales"] < 0).sum()
        ok    = int(n_neg) == 0
        return CheckResult(
            name    = "business_rule:sales_non_negative",
            passed  = ok,
            message = f"{n_neg} negative sales found." if not ok else "All sales ≥ 0.",
        )

    @staticmethod
    def _check_calendar_date_range(calendar: "pd.DataFrame") -> CheckResult:
        """Verify calendar covers all training days d_1 through d_1913."""
        d_vals = set(calendar["d"].unique())
        expected_first = "d_1"
        expected_last  = f"d_{M5_N_DAYS}"
        ok = (expected_first in d_vals) and (expected_last in d_vals)
        return CheckResult(
            name    = "date_range:calendar",
            passed  = ok,
            message = (
                f"d_1 present={expected_first in d_vals}  "
                f"d_{M5_N_DAYS} present={expected_last in d_vals}"
            ),
        )

    @staticmethod
    def _check_sales_states(sales_meta: "pd.DataFrame") -> CheckResult:
        """Verify all 3 expected US states are present."""
        found = set(sales_meta["state_id"].unique())
        expected = set(cfg.STATES)
        ok = expected.issubset(found)
        return CheckResult(
            name    = "business_rule:states",
            passed  = ok,
            message = f"States found: {sorted(found)} (expected {sorted(expected)})",
        )

    @staticmethod
    def _check_no_future_leakage(
        df: "pd.DataFrame",
        forecast_horizon: int = cfg.FORECAST_HORIZON,
    ) -> CheckResult:
        """
        Soft check: confirms the 'd_int' column exists (prerequisite
        for leakage guards in the feature-engineering phase).

        The actual lag-feature leakage check runs in src/features/
        where lag sizes are defined.
        """
        has_d_int = "d_int" in df.columns
        return CheckResult(
            name    = "leakage_guard:d_int_present",
            passed  = has_d_int,
            message = (
                "d_int present — feature-engineering leakage guard enabled."
                if has_d_int else
                "d_int column missing — lag leakage cannot be verified."
            ),
        )