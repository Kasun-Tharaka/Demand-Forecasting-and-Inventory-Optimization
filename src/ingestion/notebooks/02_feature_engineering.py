"""
notebooks/02_feature_engineering.py
─────────────────────────────────────
Phase 2 orchestration script: Cleaning → Feature Engineering → Store.

Pipeline steps
--------------
  1.  Load Phase 1 merged Parquet (or run Phase 1 if not yet done)
  2.  DataCleaner  → 100% null-free DataFrame (hard assert)
  3.  FeatureEngineer → 40+ feature columns, 0 nulls (hard assert)
  4.  FeatureStore → train/valid/test Parquet splits + manifest
  5.  Feature validation (9 automated checks)
  6.  Diagnostic plots (6 charts saved to outputs/plots/)

Run:
    python notebooks/02_feature_engineering.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
# Ensure project root is on sys.path so top-level packages like `config` import correctly.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    ),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from loguru import logger

try:
    import cudf as pd          # type: ignore
except ImportError:
    import pandas as pd

import config.settings as cfg
from src.ingestion.preprocessor  import M5Preprocessor
from src.ingestion.cleaner       import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.features.feature_store  import FeatureStore


# ─────────────────────────────────────────────────────────────
# Plot setup
# ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOT_DIR = cfg.OUTPUTS_PLOTS
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    path = PLOT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path.name}")


# ─────────────────────────────────────────────────────────────
# Step 1: Load
# ─────────────────────────────────────────────────────────────

def load_merged() -> pd.DataFrame:
    prep = M5Preprocessor()
    if not cfg.PROCESSED_MERGED.exists():
        logger.warning("Merged Parquet not found — running Phase 1 pipeline …")
        prep.run()
    logger.info("Loading merged dataset …")
    df = prep.load_processed()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info(f"Loaded: shape={df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# Step 2: Clean
# ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n── STEP 2: Data Cleaning ─────────────────────")
    cleaner  = DataCleaner(verbose=True)
    df_clean = cleaner.clean(df)
    cleaner.print_audit()

    null_total = int(df_clean.isnull().sum().sum())
    if null_total != 0:
        raise RuntimeError(
            f"DataCleaner left {null_total:,} nulls — cannot proceed to feature engineering."
        )
    logger.info(f"✅ Clean: shape={df_clean.shape}, nulls=0")
    return df_clean


# ─────────────────────────────────────────────────────────────
# Step 3: Feature engineering
# ─────────────────────────────────────────────────────────────

def engineer(df_clean: pd.DataFrame) -> tuple:
    logger.info("\n── STEP 3: Feature Engineering ───────────────")
    fe          = FeatureEngineer(verbose=True)
    df_features = fe.build(df_clean)

    null_total = int(df_features.isnull().sum().sum())
    if null_total != 0:
        raise RuntimeError(
            f"FeatureEngineer left {null_total:,} nulls — store aborted."
        )
    logger.info(f"✅ Features: {len(fe.feature_names_)} columns, nulls=0")
    return df_features, fe


# ─────────────────────────────────────────────────────────────
# Step 4: Feature Store
# ─────────────────────────────────────────────────────────────

def store_features(df_features: pd.DataFrame, fe: FeatureEngineer) -> FeatureStore:
    logger.info("\n── STEP 4: Feature Store ─────────────────────")
    store = FeatureStore()
    store.save(df_features, label_encoders=fe.label_encoders_, force=True)
    store.print_manifest()
    return store


# ─────────────────────────────────────────────────────────────
# Step 5: Validation
# ─────────────────────────────────────────────────────────────

def validate(store: FeatureStore) -> None:
    logger.info("\n── STEP 5: Feature Validation ────────────────")
    m       = store.load_manifest()
    passed  = 0
    failed  = 0

    def chk(name: str, ok: bool, msg: str) -> None:
        nonlocal passed, failed
        if ok:
            logger.info(f"  ✅  {name:<45} {msg}")
            passed += 1
        else:
            logger.error(f"  ❌  {name:<45} {msg}")
            failed += 1

    chk("null_count_zero",
        m["null_count"] == 0,
        f"null_count={m['null_count']}")

    chk("train_rows_nonzero", m["train_rows"] > 0,
        f"train_rows={m['train_rows']:,}")
    chk("valid_rows_nonzero", m["valid_rows"] > 0,
        f"valid_rows={m['valid_rows']:,}")
    chk("test_rows_nonzero",  m["test_rows"]  > 0,
        f"test_rows={m['test_rows']:,}")

    lag_feats  = m["feature_groups"]["lag_features"]
    lag_values = [int(f.replace("sales_lag_", "")) for f in lag_feats]
    unsafe     = [l for l in lag_values if l < cfg.FORECAST_HORIZON]
    chk("lag_features_exist", len(lag_feats) > 0,
        f"{len(lag_feats)} lag features")
    chk("all_lags_safe",      len(unsafe) == 0,
        f"all >= {cfg.FORECAST_HORIZON}" if not unsafe else f"UNSAFE: {unsafe}")

    roll_feats = m["feature_groups"]["rolling_features"]
    chk("rolling_features_exist", len(roll_feats) > 0,
        f"{len(roll_feats)} rolling features")

    chk("target_present",
        "sales" in m["feature_groups"].get("target", []),
        "'sales' column present")

    enc_feats = m["feature_groups"]["label_encoded"]
    chk("label_encoded_exist", len(enc_feats) > 0,
        f"{len(enc_feats)} encoded columns")

    chk("feature_count_adequate",
        m["n_features"] >= 30,
        f"n_features={m['n_features']} (>= 30 required)")

    logger.info(f"\n  Result: {passed} passed | {failed} failed")
    if failed:
        raise RuntimeError(f"Feature validation failed ({failed} checks). See above.")
    logger.info("  ✅ All validation checks passed")


# ─────────────────────────────────────────────────────────────
# Step 6: Diagnostic plots
# ─────────────────────────────────────────────────────────────

def plot_feature_coverage(store: FeatureStore) -> None:
    m      = store.load_manifest()
    groups = {k: v for k, v in m["feature_groups"].items()
              if k not in ("identifiers", "target") and len(v) > 0}

    labels = [k.replace("_", "\n") for k in groups]
    counts = [len(v) for v in groups.values()]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, counts,
                  color=sns.color_palette("muted", len(labels)), edgecolor="white")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(val),
                ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Number of Features")
    ax.set_title(
        f"Feature Engineering Coverage — {sum(counts)} Total Features",
        fontsize=13, fontweight="bold"
    )
    savefig("phase2_01_feature_coverage.png")


def plot_lag_distributions(store: FeatureStore) -> None:
    lag_cols = [c for c in store.load_manifest()["column_dtypes"]
                if c.startswith("sales_lag_")]
    if not lag_cols:
        return

    df = store.load_split("train", columns=lag_cols)
    sample = df[df["sales_lag_28"] > 0].sample(
        min(200_000, len(df)), random_state=cfg.RANDOM_SEED
    )

    fig, axes = plt.subplots(1, len(lag_cols), figsize=(4 * len(lag_cols), 4), sharey=True)
    if len(lag_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, lag_cols):
        ax.hist(sample[col].clip(0, 20), bins=30,
                color="#3498db", edgecolor="white", log=True)
        ax.set_title(col.replace("sales_lag_", "Lag "))
        ax.set_xlabel("Units (clipped 0–20)")
    axes[0].set_ylabel("Count (log scale)")
    plt.suptitle("Lag Feature Distributions (non-zero sales)", fontsize=12, fontweight="bold")
    savefig("phase2_02_lag_distributions.png")


def plot_rolling_sample(store: FeatureStore) -> None:
    need = ["id", "d_int", "sales", "sales_lag_28",
            "rolling_mean_7d", "rolling_mean_28d"]
    available = [c for c in need if c in store.load_manifest()["column_dtypes"]]
    if "id" not in available or "d_int" not in available:
        return

    df = store.load_split("train", columns=available)
    best_item = df.groupby("id", observed=True)["sales"].sum().idxmax()
    item_df   = df[df["id"] == best_item].sort_values("d_int").head(400)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(item_df["d_int"], item_df["sales"],
            alpha=0.3, color="#bdc3c7", linewidth=0.8, label="Actual Sales")
    if "rolling_mean_7d" in item_df.columns:
        ax.plot(item_df["d_int"], item_df["rolling_mean_7d"],
                color="#e74c3c", linewidth=1.5, label="Rolling Mean 7d")
    if "rolling_mean_28d" in item_df.columns:
        ax.plot(item_df["d_int"], item_df["rolling_mean_28d"],
                color="#2980b9", linewidth=1.5, label="Rolling Mean 28d")
    if "sales_lag_28" in item_df.columns:
        ax.plot(item_df["d_int"], item_df["sales_lag_28"],
                color="#27ae60", linewidth=1.0, linestyle="--",
                alpha=0.7, label="Lag-28")
    ax.set_xlabel("Day Index (d_int)")
    ax.set_ylabel("Units Sold")
    ax.set_title(f"Rolling Feature Visualisation — Item: {best_item}")
    ax.legend(loc="upper left")
    savefig("phase2_03_rolling_feature_sample.png")


def plot_price_features(store: FeatureStore) -> None:
    manifest = store.load_manifest()
    want = ["sell_price", "price_norm_by_item", "price_change_7d", "is_price_promo"]
    cols = [c for c in want if c in manifest["column_dtypes"]]
    if not cols:
        return

    df     = store.load_split("train", columns=cols)
    sample = df.sample(min(100_000, len(df)), random_state=cfg.RANDOM_SEED)

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    pal = sns.color_palette("Set2", len(cols))

    for ax, col, color in zip(axes, cols, pal):
        data = sample[col].replace([np.inf, -np.inf], np.nan).dropna()
        if col == "is_price_promo":
            ax.bar(["Not Promo", "Promo"],
                   [(data == 0).sum(), (data == 1).sum()],
                   color=[color, "#e74c3c"])
        else:
            clipped = data.clip(data.quantile(0.01), data.quantile(0.99))
            ax.hist(clipped, bins=50, color=color, edgecolor="white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Value")
    plt.suptitle("Price Feature Distributions (training set)",
                 fontsize=12, fontweight="bold")
    savefig("phase2_04_price_features.png")


def plot_split_sizes(store: FeatureStore) -> None:
    m = store.load_manifest()
    splits  = ["Train", "Validation", "Test"]
    rows    = [m["train_rows"], m["valid_rows"], m["test_rows"]]
    days    = [
        cfg.CV_TRAIN_END_DAY,
        cfg.CV_VALID_END_DAY - cfg.CV_VALID_START_DAY + 1,
        cfg.N_TRAIN_DAYS - cfg.CV_TEST_START_DAY + 1,
    ]
    colors  = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, title, unit in [
        (axes[0], [r/1e6 for r in rows], "Row Counts", "Millions"),
        (axes[1], days,                   "Day Coverage", "Days"),
    ]:
        bars = ax.bar(splits, vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.1f}M" if unit == "Millions" else f"{int(val)}d",
                    ha="center", fontweight="bold")
        ax.set_ylabel(unit)
        ax.set_title(f"Walk-Forward Split: {title}")
    plt.suptitle("Train / Validation / Test Split Strategy",
                 fontsize=12, fontweight="bold")
    savefig("phase2_05_split_sizes.png")


def plot_null_audit(store: FeatureStore) -> None:
    """Visual proof of zero nulls in every split — for portfolio."""
    results = {}
    for split in ["train", "valid", "test"]:
        df    = store.load_split(split)
        nulls = int(df.isnull().sum().sum())
        results[split] = nulls
        del df
        gc.collect()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    table_data = [
        [s.upper(), f"{n:,}", "✅ Clean" if n == 0 else "❌ Has Nulls"]
        for s, n in results.items()
    ]
    tbl = ax.table(
        cellText  = table_data,
        colLabels = ["Split", "Null Count", "Status"],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for i, (_, n) in enumerate(results.items(), 1):
        tbl[(i, 2)].set_facecolor("#d5f5e3" if n == 0 else "#fadbd8")
        tbl[(i, 0)].set_facecolor("#eaf2ff")
    plt.title("Null Audit — All Feature Splits",
              fontsize=13, fontweight="bold", pad=20)
    savefig("phase2_06_null_audit.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║  M5 Inventory Optimizer — Phase 2: Feature Engineering   ║")
    logger.info("╚" + "═" * 60 + "╝")
    t_start = time.perf_counter()

    # 1. Load
    df_merged = load_merged()

    # 2. Clean
    df_clean = clean(df_merged)
    del df_merged
    gc.collect()

    # 3. Feature engineering
    df_features, fe = engineer(df_clean)
    del df_clean
    gc.collect()

    # 4. Store
    store = store_features(df_features, fe)
    del df_features
    gc.collect()

    # 5. Validate
    validate(store)

    # 6. Plots
    logger.info("\n── STEP 6: Diagnostic Plots ──────────────────")
    plot_feature_coverage(store)
    plot_lag_distributions(store)
    plot_rolling_sample(store)
    plot_price_features(store)
    plot_split_sizes(store)
    plot_null_audit(store)

    elapsed = time.perf_counter() - t_start
    logger.info("\n" + "═" * 62)
    logger.info(f"Phase 2 complete in {elapsed:.0f}s")
    logger.info(f"Feature store  : {cfg.FEATURES_STORE_DIR}")
    logger.info(f"Plots          : {PLOT_DIR}")
    logger.info("Next → Phase 3: LightGBM + Walk-Forward CV (src/models/)")
    logger.info("═" * 62)


if __name__ == "__main__":
    main()