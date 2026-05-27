"""
src/features/feature_store.py
──────────────────────────────
Persists, versions, and splits the engineered feature DataFrame.

Why a feature store?
---------------------
Re-running 58M-row feature engineering takes 10–30 min on CPU.
This store writes the result once and lets every downstream
experiment load in seconds via Parquet projection push-down.

Walk-forward split boundaries
------------------------------
  Train      : d_int in [1,    1800]   →  1,800 days × 30,490 items
  Validation : d_int in [1801, 1856]   →     56 days × 30,490 items
  Test       : d_int in [1857, 1913]   →     57 days × 30,490 items

Usage
-----
    from src.features.feature_store import FeatureStore
    store = FeatureStore()
    store.save(df_features, label_encoders=fe.label_encoders_)

    df_train = store.load_split("train")
    df_valid  = store.load_split("valid")
    df_test   = store.load_split("test")
"""
from __future__ import annotations

import datetime
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cudf as pd          # type: ignore
except ImportError:
    import pandas as pd

import numpy as np
from loguru import logger

import config.settings as cfg


class FeatureStore:
    """
    Manages saving and loading of the feature matrix and splits.

    Parameters
    ----------
    store_dir : Path, optional
        Root directory. Defaults to cfg.FEATURES_STORE_DIR.
    """

    SPLIT_COL = "d_int"

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = store_dir or cfg.FEATURES_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._path_full     = self.store_dir / "features_full.parquet"
        self._path_train    = cfg.FEATURES_TRAIN
        self._path_valid    = cfg.FEATURES_VALID
        self._path_test     = cfg.FEATURES_TEST
        self._path_encoders = cfg.LABEL_ENCODERS_PATH
        self._path_manifest = cfg.FEATURE_MANIFEST

    # ── Public API ────────────────────────────────────────────

    def save(
        self,
        df:             pd.DataFrame,
        label_encoders: Optional[Dict] = None,
        force:          bool           = False,
    ) -> None:
        """Persist full features, splits, encoders, and manifest."""
        if self._path_train.exists() and not force:
            logger.info("Feature store exists. Pass force=True to overwrite.")
            return

        logger.info("=" * 62)
        logger.info(f"FeatureStore.save()  rows={len(df):,}  dir={self.store_dir}")
        t0 = time.perf_counter()

        # Null assertion — never write a null-contaminated store
        null_total = int(df.isnull().sum().sum())
        if null_total > 0:
            raise RuntimeError(
                f"FeatureStore.save() refused: {null_total:,} nulls in DataFrame. "
                "Run DataCleaner + FeatureEngineer first."
            )

        # 1. Full feature file
        logger.info(f"  Writing full features ({len(df):,} rows) …")
        df.to_parquet(
            self._path_full,
            index=False,
            compression=cfg.PARQUET_COMPRESSION,
        )

        # 2. Train / valid / test splits
        self._write_splits(df)

        # 3. Label encoders
        if label_encoders:
            with open(self._path_encoders, "wb") as f:
                pickle.dump(label_encoders, f)
            logger.info(f"  Label encoders saved ({len(label_encoders)} encoders)")

        # 4. Manifest
        self._write_manifest(df)

        elapsed = time.perf_counter() - t0
        logger.info(f"FeatureStore.save() done in {elapsed:.1f}s")
        logger.info("=" * 62)

    def load_split(
        self,
        split:   str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load one data split.

        Parameters
        ----------
        split   : 'train' | 'valid' | 'test' | 'full'
        columns : optional projection for faster loads
        """
        paths = {
            "train": self._path_train,
            "valid": self._path_valid,
            "test":  self._path_test,
            "full":  self._path_full,
        }
        if split not in paths:
            raise ValueError(f"split must be one of {list(paths)}, got '{split}'")

        path = paths[split]
        if not path.exists():
            raise FileNotFoundError(
                f"Split '{split}' not found at {path}.\n"
                "Run FeatureStore.save(df_features) first."
            )
        t0 = time.perf_counter()
        df = pd.read_parquet(path, columns=columns)
        mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        logger.info(
            f"Loaded split='{split}'  shape={df.shape}  "
            f"size={mb:.1f} MB  time={time.perf_counter()-t0:.2f}s"
        )
        return df

    def load_encoders(self) -> Dict:
        if not self._path_encoders.exists():
            raise FileNotFoundError(f"No encoders at {self._path_encoders}")
        with open(self._path_encoders, "rb") as f:
            return pickle.load(f)

    def load_manifest(self) -> Dict:
        if not self._path_manifest.exists():
            raise FileNotFoundError(f"No manifest at {self._path_manifest}")
        with open(self._path_manifest) as f:
            return json.load(f)

    def print_manifest(self) -> None:
        m = self.load_manifest()
        logger.info("═" * 62)
        logger.info("Feature Store Manifest")
        logger.info("─" * 62)
        logger.info(f"  Total features  : {m['n_features']}")
        logger.info(f"  Total rows      : {m['n_rows']:,}")
        logger.info(f"  Train rows      : {m['train_rows']:,}")
        logger.info(f"  Valid rows      : {m['valid_rows']:,}")
        logger.info(f"  Test rows       : {m['test_rows']:,}")
        logger.info(f"  Null count      : {m['null_count']}  {'✅' if m['null_count']==0 else '❌'}")
        logger.info(f"  Created at      : {m['created_at']}")
        logger.info("─" * 62)
        for grp, cols in m["feature_groups"].items():
            logger.info(f"  {grp:<30} {len(cols):>3} features")
        logger.info("═" * 62)

    def exists(self) -> bool:
        return all(
            p.exists()
            for p in [self._path_train, self._path_valid, self._path_test]
        )

    # ── Private helpers ───────────────────────────────────────

    def _write_splits(self, df: pd.DataFrame) -> None:
        split_defs = {
            "train": (1,                      cfg.CV_TRAIN_END_DAY),
            "valid": (cfg.CV_VALID_START_DAY, cfg.CV_VALID_END_DAY),
            "test":  (cfg.CV_TEST_START_DAY,  cfg.N_TRAIN_DAYS),
        }
        paths = {
            "train": self._path_train,
            "valid": self._path_valid,
            "test":  self._path_test,
        }
        col = self.SPLIT_COL
        if col not in df.columns:
            raise RuntimeError(f"'{col}' column required for splits but not in DataFrame.")

        for name, (d_start, d_end) in split_defs.items():
            mask     = (df[col] >= d_start) & (df[col] <= d_end)
            split_df = df[mask].reset_index(drop=True)
            split_df.to_parquet(
                paths[name], index=False, compression=cfg.PARQUET_COMPRESSION
            )
            logger.info(
                f"  Split '{name}'  → {len(split_df):>10,} rows  "
                f"(days {d_start}–{d_end})  → {paths[name].name}"
            )

    def _write_manifest(self, df: pd.DataFrame) -> None:
        groups = self._classify_features(df)
        col    = self.SPLIT_COL

        train_rows = int(((df[col] >= 1) & (df[col] <= cfg.CV_TRAIN_END_DAY)).sum())
        valid_rows = int(((df[col] >= cfg.CV_VALID_START_DAY) & (df[col] <= cfg.CV_VALID_END_DAY)).sum())
        test_rows  = int((df[col] >= cfg.CV_TEST_START_DAY).sum())

        manifest = {
            "n_features":      len(df.columns),
            "n_rows":          len(df),
            "train_rows":      train_rows,
            "valid_rows":      valid_rows,
            "test_rows":       test_rows,
            "null_count":      int(df.isnull().sum().sum()),
            "created_at":      datetime.datetime.now().isoformat(),
            "feature_groups":  groups,
            "column_dtypes":   {c: str(t) for c, t in df.dtypes.items()},
            "lag_days":        cfg.LAG_DAYS,
            "rolling_windows": cfg.ROLLING_WINDOWS,
            "rolling_stats":   cfg.ROLLING_STATS,
            "forecast_horizon":cfg.FORECAST_HORIZON,
        }
        with open(self._path_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"  Manifest written → {self._path_manifest.name}")

    @staticmethod
    def _classify_features(df: pd.DataFrame) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {
            "identifiers":       [],
            "target":            [],
            "lag_features":      [],
            "rolling_features":  [],
            "calendar_features": [],
            "price_features":    [],
            "label_encoded":     [],
            "other":             [],
        }
        _ID_COLS    = {"id","item_id","dept_id","cat_id","store_id","state_id","d","d_int"}
        _CAL_COLS   = {
            "wday","month","year","quarter","day_of_year","date","weekday","wm_yr_wk",
            "is_weekend","is_month_start","is_month_end","is_year_start","is_year_end",
            "has_event","has_event_2","is_snap_day","days_since_event",
            "snap_CA","snap_TX","snap_WI",
            "event_name_1","event_type_1","event_name_2","event_type_2",
        }
        _PRICE_COLS = {"sell_price","is_price_promo"}

        for col in df.columns:
            if col in _ID_COLS:
                groups["identifiers"].append(col)
            elif col == "sales":
                groups["target"].append(col)
            elif col.startswith("sales_lag_"):
                groups["lag_features"].append(col)
            elif col.startswith("rolling_"):
                groups["rolling_features"].append(col)
            elif col in _CAL_COLS:
                groups["calendar_features"].append(col)
            elif col in _PRICE_COLS or col.startswith("price_"):
                groups["price_features"].append(col)
            elif col.endswith("_enc"):
                groups["label_encoded"].append(col)
            else:
                groups["other"].append(col)
        return groups