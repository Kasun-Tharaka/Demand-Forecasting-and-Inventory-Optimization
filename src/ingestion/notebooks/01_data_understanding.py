"""
notebooks/01_data_understanding.py
────────────────────────────────────
Phase 1 EDA — Data Understanding & Distribution Analysis

Run as a script:   python notebooks/01_data_understanding.py
Run as notebook:   jupyter nbconvert --to notebook --execute ...

What this script produces
--------------------------
  outputs/plots/01_sales_distribution.png
  outputs/plots/02_zero_inflation.png
  outputs/plots/03_hierarchy_aggregation.png
  outputs/plots/04_weekly_seasonality.png
  outputs/plots/05_yearly_seasonality.png
  outputs/plots/06_price_elasticity.png
  outputs/plots/07_snap_effect.png
  outputs/plots/08_memory_comparison.png

All plots are saved at 150 DPI — presentation quality without
being excessively large files.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from loguru import logger

try:
    import cudf as pd   # type: ignore
    import cudf.pandas  # type: ignore
    _BACKEND = "cuDF"
except ImportError:
    import pandas as pd
    _BACKEND = "Pandas"

import config.settings as cfg
from src.ingestion.loader import M5DataLoader, downcast_dataframe, _df_mb
from src.ingestion.preprocessor import M5Preprocessor
from src.ingestion.validator import M5Validator


# ─────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOT_DIR = cfg.OUTPUTS_PLOTS
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150


def savefig(name: str) -> None:
    path = PLOT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot → {path}")


# ─────────────────────────────────────────────────────────────
# 1. Memory Comparison: naive vs optimised
# ─────────────────────────────────────────────────────────────

def plot_memory_comparison(loader: M5DataLoader) -> None:
    """
    Demonstrate memory savings from dtype downcasting by loading
    a small subset of the sales data with and without optimisation.
    """
    logger.info("Plot 1/8 — Memory comparison")
    DAY_SUBSET = 200   # use 200 days for demo

    # Naive load (all float64 / int64)
    df_naive = pd.read_csv(
        cfg.RAW_SALES,
        usecols=cfg.ID_COLS + [f"d_{i}" for i in range(1, DAY_SUBSET + 1)],
    )
    naive_mb = _df_mb(df_naive)

    # Optimised load
    df_opt = pd.read_csv(
        cfg.RAW_SALES,
        usecols=cfg.ID_COLS + [f"d_{i}" for i in range(1, DAY_SUBSET + 1)],
        dtype={
            **cfg.SALES_META_DTYPES,
            **{f"d_{i}": "float32" for i in range(1, DAY_SUBSET + 1)},
        },
    )
    opt_mb = _df_mb(df_opt)
    del df_naive, df_opt
    gc.collect()

    # Extrapolate to full 1913 days
    scale = cfg.N_TRAIN_DAYS / DAY_SUBSET
    labels = ["Naive\n(float64 / int64)", "Optimised\n(float32 / int16 / category)"]
    values = [naive_mb * scale / 1024, opt_mb * scale / 1024]  # GB

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f} GB",
            ha="center", va="bottom", fontweight="bold", fontsize=12
        )
    ax.set_ylabel("Estimated RAM Usage (GB)")
    ax.set_title(
        f"Memory Optimisation: {values[0]:.2f} GB → {values[1]:.2f} GB "
        f"({(1 - values[1]/values[0])*100:.0f}% saved)"
    )
    ax.set_ylim(0, values[0] * 1.3)
    savefig("08_memory_comparison.png")


# ─────────────────────────────────────────────────────────────
# 2. Sales Distribution & Zero-Inflation
# ─────────────────────────────────────────────────────────────

def plot_sales_distribution(df: pd.DataFrame) -> None:
    """
    Plot the right-skewed sales distribution.
    This motivates our choice of Tweedie loss in Phase 4.
    """
    logger.info("Plot 2/8 — Sales distribution")
    sales_nonzero = df.loc[df["sales"] > 0, "sales"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Log scale histogram
    axes[0].hist(sales_nonzero, bins=80, color="#3498db", edgecolor="white", log=True)
    axes[0].set_xlabel("Daily Unit Sales")
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("Sales Distribution (non-zero values)")
    axes[0].set_xlim(0, sales_nonzero.quantile(0.99))

    # Box plot by category
    cat_order = df.groupby("cat_id")["sales"].median().sort_values(ascending=False).index
    df_sample = df.sample(min(200_000, len(df)), random_state=cfg.RANDOM_SEED)
    sns.boxplot(
        data     = df_sample,
        x        = "cat_id",
        y        = "sales",
        order    = cat_order,
        palette  = "muted",
        ax       = axes[1],
        showfliers=False,
    )
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Daily Unit Sales")
    axes[1].set_title("Sales by Category (outliers hidden)")

    plt.suptitle("M5 Sales Distribution Analysis", fontsize=14, fontweight="bold")
    savefig("01_sales_distribution.png")


def plot_zero_inflation(df: pd.DataFrame) -> None:
    """
    Visualise the proportion of zero-sales days by store and category.
    High zero rates → Tweedie / Poisson loss, not RMSE.
    """
    logger.info("Plot 3/8 — Zero-inflation analysis")

    zero_rates = (
        df.groupby(["store_id", "cat_id"])["sales"]
        .apply(lambda x: (x == 0).mean())
        .reset_index(name="zero_rate")
    )

    pivot = zero_rates.pivot(index="store_id", columns="cat_id", values="zero_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot      = True,
        fmt        = ".0%",
        cmap       = "YlOrRd",
        ax         = ax,
        linewidths = 0.5,
        cbar_kws   = {"label": "% Days with Zero Sales"},
    )
    ax.set_title(
        "Zero-Inflation Rate by Store × Category\n"
        "(High rates motivate Tweedie loss over standard RMSE)",
        fontsize=12,
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("Store")
    savefig("02_zero_inflation.png")


# ─────────────────────────────────────────────────────────────
# 3. Hierarchical Aggregation
# ─────────────────────────────────────────────────────────────

def plot_hierarchy_aggregation(df: pd.DataFrame) -> None:
    """
    Show total sales at different hierarchy levels:
    item → department → category → store → state → national.
    """
    logger.info("Plot 4/8 — Hierarchical aggregation")

    # National daily total
    national = df.groupby("date")["sales"].sum().reset_index()
    national["date"] = pd.to_datetime(national["date"])

    # By state
    by_state = df.groupby(["date", "state_id"])["sales"].sum().reset_index()
    by_state["date"] = pd.to_datetime(by_state["date"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(national["date"], national["sales"], color="#2c3e50", linewidth=0.8)
    axes[0].set_ylabel("Total Daily Units Sold")
    axes[0].set_title("National Aggregate Sales")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    for state, grp in by_state.groupby("state_id"):
        axes[1].plot(grp["date"], grp["sales"], label=state, linewidth=0.8)
    axes[1].set_ylabel("Daily Units Sold")
    axes[1].set_title("Sales by State")
    axes[1].legend(title="State", loc="upper left")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.suptitle("M5 Hierarchical Sales — National & State Level", fontsize=13, fontweight="bold")
    savefig("03_hierarchy_aggregation.png")


# ─────────────────────────────────────────────────────────────
# 4. Seasonality Analysis
# ─────────────────────────────────────────────────────────────

def plot_seasonality(df: pd.DataFrame) -> None:
    """Weekly and monthly seasonality patterns."""
    logger.info("Plot 5/8 — Seasonality analysis")

    avg_by_wday  = df.groupby("wday")["sales"].mean().reset_index()
    avg_by_month = df.groupby("month")["sales"].mean().reset_index()

    wday_labels  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(
        avg_by_wday["wday"] - 1,
        avg_by_wday["sales"],
        color=["#e67e22" if w >= 5 else "#3498db" for w in range(7)],
        edgecolor="white",
    )
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(wday_labels)
    axes[0].set_ylabel("Mean Daily Sales")
    axes[0].set_title("Weekly Seasonality\n(orange = weekend)")

    axes[1].bar(
        avg_by_month["month"] - 1,
        avg_by_month["sales"],
        color="#9b59b6",
        edgecolor="white",
    )
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels(month_labels, rotation=45)
    axes[1].set_ylabel("Mean Daily Sales")
    axes[1].set_title("Monthly / Yearly Seasonality")

    plt.suptitle("M5 Seasonality Patterns", fontsize=13, fontweight="bold")
    savefig("04_seasonality.png")


# ─────────────────────────────────────────────────────────────
# 5. Price Elasticity
# ─────────────────────────────────────────────────────────────

def plot_price_elasticity(df: pd.DataFrame) -> None:
    """
    Scatter of log(price) vs log(sales) per category.
    A negative slope indicates price elasticity.
    """
    logger.info("Plot 6/8 — Price elasticity")

    sub = df.dropna(subset=["sell_price"]).copy()
    sub = sub[sub["sales"] > 0]
    sub["log_price"] = np.log1p(sub["sell_price"].astype(float))
    sub["log_sales"] = np.log1p(sub["sales"].astype(float))

    sample = sub.sample(min(100_000, len(sub)), random_state=cfg.RANDOM_SEED)

    g = sns.FacetGrid(
        sample,
        col        = "cat_id",
        height     = 4,
        aspect     = 1.2,
        col_wrap   = 3,
        sharex     = False,
        sharey     = False,
    )
    g.map_dataframe(
        sns.regplot,
        x          = "log_price",
        y          = "log_sales",
        scatter_kws= {"alpha": 0.05, "s": 8, "color": "#3498db"},
        line_kws   = {"color": "#e74c3c", "linewidth": 2},
    )
    g.set_axis_labels("log(1 + Price)", "log(1 + Sales)")
    g.set_titles("{col_name}")
    g.figure.suptitle(
        "Price Elasticity by Category\n(negative slope = demand falls as price rises)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    savefig("05_price_elasticity.png")


# ─────────────────────────────────────────────────────────────
# 6. SNAP Benefit Day Effect
# ─────────────────────────────────────────────────────────────

def plot_snap_effect(df: pd.DataFrame) -> None:
    """
    Compare mean sales on SNAP days vs non-SNAP days per state.
    SNAP days create demand spikes that a naive model will miss.
    """
    logger.info("Plot 7/8 — SNAP effect")

    records = []
    for state in cfg.STATES:
        snap_col = f"snap_{state}"
        if snap_col not in df.columns:
            continue
        grp = df[df["state_id"] == state].copy()
        mean_snap    = grp.loc[grp[snap_col] == 1, "sales"].mean()
        mean_nonsnap = grp.loc[grp[snap_col] == 0, "sales"].mean()
        records.append({
            "state": state,
            "SNAP day":     mean_snap,
            "Non-SNAP day": mean_nonsnap,
        })

    snap_df = pd.DataFrame(records).melt(id_vars="state", var_name="Day Type", value_name="Mean Sales")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=snap_df, x="state", y="Mean Sales", hue="Day Type", palette=["#27ae60", "#95a5a6"], ax=ax)
    ax.set_title("SNAP Benefit Day Effect on Mean Daily Sales\n(Green = SNAP day — expect demand spike)")
    ax.set_xlabel("State")
    ax.set_ylabel("Mean Unit Sales per Item-Day")
    savefig("06_snap_effect.png")


# ─────────────────────────────────────────────────────────────
# 7. Summary Statistics Table
# ─────────────────────────────────────────────────────────────

def print_summary_statistics(df: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Print key statistics to the logger for the README."""
    logger.info("═" * 60)
    logger.info("M5 Dataset Summary Statistics")
    logger.info("─" * 60)
    logger.info(f"  Total item-day observations : {len(df):>15,}")
    logger.info(f"  Unique items (series)       : {df['id'].nunique():>15,}")
    logger.info(f"  Unique stores               : {df['store_id'].nunique():>15,}")
    logger.info(f"  Unique states               : {df['state_id'].nunique():>15,}")
    logger.info(f"  Calendar rows               : {len(cal):>15,}")
    logger.info(f"  Price observations          : {len(prices):>15,}")
    logger.info(f"  Date range                  : {df['date'].min()} → {df['date'].max()}")
    logger.info(f"  Zero sales days (%)         : {(df['sales']==0).mean()*100:>14.1f}%")
    logger.info(f"  Mean daily sales (all items): {df['sales'].mean():>15.3f}")
    logger.info(f"  Median sell price           : ${prices['sell_price'].median():>13.2f}")
    logger.info(f"  Max sell price              : ${prices['sell_price'].max():>13.2f}")
    logger.info("═" * 60)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Phase 1 EDA starting …")

    loader = M5DataLoader()
    loader.memory_report()

    # ── Load reference tables ─────────────────────────────────
    cal    = loader.load_calendar()
    prices = loader.load_prices()
    meta   = loader.load_sales_meta()

    # ── Validate raw tables ───────────────────────────────────
    validator = M5Validator()
    raw_report = validator.validate_raw(cal, prices, meta)
    M5Validator.print_report(raw_report)

    # ── Memory comparison plot (uses raw files, no processed needed) ──
    if cfg.RAW_SALES.exists():
        plot_memory_comparison(loader)

    # ── Load or build processed long-format data ──────────────
    prep = M5Preprocessor()
    if not cfg.PROCESSED_MERGED.exists():
        logger.warning(
            "Processed Parquet not found. Running ingestion pipeline … "
            "(This will take several minutes on first run)"
        )
        prep.run()

    # Load a representative sample for EDA plots
    # (load ALL rows for final analysis; sample here for speed in dev)
    logger.info("Loading processed dataset …")
    df = prep.load_processed(
        columns=[
            "id", "item_id", "dept_id", "cat_id",
            "store_id", "state_id",
            "d", "d_int", "date", "sales", "sell_price",
            "wday", "month", "year", "wm_yr_wk",
            "snap_CA", "snap_TX", "snap_WI",
        ]
    )
    df["date"] = pd.to_datetime(df["date"])

    # ── Validate processed data ───────────────────────────────
    processed_report = validator.validate_processed(df)
    M5Validator.print_report(processed_report)

    # ── Summary stats ─────────────────────────────────────────
    print_summary_statistics(df, cal, prices)

    # ── EDA plots ─────────────────────────────────────────────
    plot_sales_distribution(df)
    plot_zero_inflation(df)
    plot_hierarchy_aggregation(df)
    plot_seasonality(df)
    plot_price_elasticity(df)
    plot_snap_effect(df)

    logger.info(f"\n✅ Phase 1 complete. All plots saved to: {PLOT_DIR}")
    logger.info("Next step → Phase 2: Feature Engineering (src/features/)")


if __name__ == "__main__":
    main()