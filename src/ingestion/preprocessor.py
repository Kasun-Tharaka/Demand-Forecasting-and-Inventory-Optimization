"""
src/ingestion/preprocessor.py
──────────────────────────────
Transforms the three raw M5 CSVs into a single, analysis-ready
long-format Parquet file.

Pipeline steps
--------------
1.  Load calendar and prices (small tables — load fully).
2.  Load sales wide table in DAY-COLUMN CHUNKS to avoid peak-RAM spike.
3.  Melt each chunk from wide to long format.
4.  Join calendar on 'd'; join prices on (store_id, item_id, wm_yr_wk).
5.  Downcast all columns to smallest safe dtypes.
6.  Append each processed chunk to a Parquet dataset on disk.

The chunked approach means peak RAM usage is bounded by
(n_items × chunk_size) rather than (n_items × 1913).

Usage
-----
    # From project root:
    python -m src.ingestion.preprocessor

    # Or programmatically:
    from src.ingestion.preprocessor import M5Preprocessor
    prep = M5Preprocessor()
    prep.run()
    df   = prep.load_processed()   # load the finished Parquet
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import cudf as pd          # type: ignore
    _BACKEND = "cuDF"
except ImportError:
    import pandas as pd
    _BACKEND = "Pandas"

from loguru import logger

import config.settings as cfg
from src.ingestion.loader import M5DataLoader, downcast_dataframe, _df_mb, _ram_mb


# ─────────────────────────────────────────────────────────────
# Calendar helper
# ─────────────────────────────────────────────────────────────

def _prepare_calendar(cal: "pd.DataFrame") -> "pd.DataFrame":
    """
    Select and rename calendar columns for merging.
    Returns a lean DataFrame indexed on 'd' (the string column).
    """
    keep = [
        "d", "d_int", "date", "wm_yr_wk", "weekday", "wday",
        "month", "year",
        "event_name_1", "event_type_1",
        "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    # Only keep columns that actually exist (future-proofing)
    keep = [c for c in keep if c in cal.columns]
    return cal[keep].copy()


# ─────────────────────────────────────────────────────────────
# Main Preprocessor Class
# ─────────────────────────────────────────────────────────────

class M5Preprocessor:
    """
    Orchestrates the full Phase-1 ingestion pipeline.

    Parameters
    ----------
    data_dir : Path, optional
        Raw data directory (defaults to config).
    output_path : Path, optional
        Where to write the processed Parquet file.
    chunk_size : int
        Number of day-columns to process per iteration.
        Reduce to 100-150 on machines with < 8 GB RAM.
    force_reprocess : bool
        If False and the output Parquet already exists, skip processing.
    """

    def __init__(
        self,
        data_dir:         Optional[Path] = None,
        output_path:      Optional[Path] = None,
        chunk_size:       int            = cfg.MELT_CHUNK_SIZE,
        force_reprocess:  bool           = False,
    ) -> None:
        self.loader        = M5DataLoader(data_dir)
        self.output_path   = output_path or cfg.PROCESSED_MERGED
        self.chunk_size    = chunk_size
        self.force_reprocess = force_reprocess

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── public API ───────────────────────────────────────────

    def run(self) -> Path:
        """
        Execute the full pipeline and return the path of the output
        Parquet file.

        Returns
        -------
        Path  path to sales_merged.parquet
        """
        if self.output_path.exists() and not self.force_reprocess:
            logger.info(f"Processed file already exists: {self.output_path} — skipping.")
            logger.info("Pass force_reprocess=True to rebuild from scratch.")
            return self.output_path

        logger.info("═" * 60)
        logger.info("M5 Ingestion Pipeline — START")
        logger.info(f"  Backend     : {_BACKEND}")
        logger.info(f"  Chunk size  : {self.chunk_size} day-columns")
        logger.info(f"  Output      : {self.output_path}")
        logger.info(f"  RAM at start: {_ram_mb():.0f} MB")
        logger.info("═" * 60)

        t_total = time.perf_counter()

        # ── Step 1: Load small reference tables ──────────────
        calendar = _prepare_calendar(self.loader.load_calendar())
        prices   = self.loader.load_prices()

        # Build a lookup dict: d_int → wm_yr_wk for fast vectorised joins
        d_to_wm = dict(zip(calendar["d_int"], calendar["wm_yr_wk"]))

        # ── Step 2: Load sales meta (IDs only) ───────────────
        sales_meta = self.loader.load_sales_meta()

        # ── Step 3: Get all day columns and split into chunks ─
        all_day_cols = self.loader.get_day_columns()
        n_chunks = (len(all_day_cols) + self.chunk_size - 1) // self.chunk_size

        logger.info(
            f"Processing {len(all_day_cols)} day-columns "
            f"in {n_chunks} chunks of {self.chunk_size}"
        )

        # ── Step 4: Chunked melt + merge ──────────────────────
        # We write each processed chunk to Parquet and concatenate at the
        # end — this keeps peak RAM ~constant regardless of dataset size.
        tmp_dir = cfg.DATA_CACHE / "_merge_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths: list[Path] = []

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end   = min(chunk_start + self.chunk_size, len(all_day_cols))
            day_cols    = all_day_cols[chunk_start:chunk_end]

            logger.info(
                f"Chunk {chunk_idx + 1}/{n_chunks}  "
                f"days d_{chunk_start + 1}..d_{chunk_end}  "
                f"RAM={_ram_mb():.0f} MB"
            )

            chunk_path = self._process_chunk(
                chunk_idx  = chunk_idx,
                day_cols   = day_cols,
                sales_meta = sales_meta,
                calendar   = calendar,
                prices     = prices,
                d_to_wm    = d_to_wm,
                tmp_dir    = tmp_dir,
            )
            chunk_paths.append(chunk_path)

            gc.collect()

        # ── Step 5: Combine all chunk Parquet files ───────────
        logger.info("Combining chunks into final Parquet dataset …")
        self._combine_chunks(chunk_paths)

        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

        elapsed = time.perf_counter() - t_total
        logger.info(f"Pipeline finished in {elapsed:.1f}s  |  output: {self.output_path}")
        return self.output_path

    def load_processed(self, columns: Optional[list] = None) -> "pd.DataFrame":
        """
        Load the processed Parquet file back into memory.

        Parameters
        ----------
        columns : list of str, optional
            Projection pushdown — only load these columns (much faster).
        """
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Processed file not found: {self.output_path}\n"
                "Run M5Preprocessor().run() first."
            )
        t0 = time.perf_counter()
        df = pd.read_parquet(self.output_path, columns=columns)
        logger.info(
            f"Loaded processed Parquet  shape={df.shape}  "
            f"size={_df_mb(df):.1f} MB  time={time.perf_counter()-t0:.2f}s"
        )
        return df

    # ── private helpers ──────────────────────────────────────

    def _process_chunk(
        self,
        chunk_idx:  int,
        day_cols:   list[str],
        sales_meta: "pd.DataFrame",
        calendar:   "pd.DataFrame",
        prices:     "pd.DataFrame",
        d_to_wm:    dict,
        tmp_dir:    Path,
    ) -> Path:
        """
        Process a single chunk of day-columns through the full
        melt → calendar-join → price-join → downcast pipeline.

        Returns the path of the written chunk Parquet file.
        """
        # 1. Load only this chunk's day columns + identifiers
        chunk_sales = pd.read_csv(
            cfg.RAW_SALES,
            usecols=cfg.ID_COLS + day_cols,
            dtype={
                **cfg.SALES_META_DTYPES,
                **{c: "float32" for c in day_cols},
            },
        )

        # 2. Melt wide → long
        long = chunk_sales.melt(
            id_vars    = cfg.ID_COLS,
            value_vars = day_cols,
            var_name   = "d",
            value_name = "sales",
        )
        del chunk_sales
        gc.collect()

        # 3. Extract integer day index (faster joins than string)
        long["d_int"] = (
            long["d"].str.replace("d_", "", regex=False).astype("int16")
        )

        # 4. Join calendar (left join on 'd')
        long = long.merge(
            calendar,
            on  = ["d", "d_int"],
            how = "left",
        )

        # 5. Map wm_yr_wk from d_int (already available post-calendar join)
        # Join sell_prices on (store_id, item_id, wm_yr_wk)
        long = long.merge(
            prices,
            on  = ["store_id", "item_id", "wm_yr_wk"],
            how = "left",
        )

        # 6. Downcast everything
        long = downcast_dataframe(long)

        # 7. Write chunk to disk
        chunk_path = tmp_dir / f"chunk_{chunk_idx:04d}.parquet"
        long.to_parquet(
            chunk_path,
            compression = cfg.PARQUET_COMPRESSION,
            index       = False,
        )
        logger.debug(f"  → Wrote {chunk_path.name}  {_df_mb(long):.1f} MB in memory")

        del long
        gc.collect()
        return chunk_path

    def _combine_chunks(self, chunk_paths: list[Path]) -> None:
        """
        Use PyArrow to concatenate all chunk Parquet files into one
        without loading everything into Pandas RAM simultaneously.
        """
        tables = []
        for p in chunk_paths:
            tables.append(pq.read_table(str(p)))

        combined = pa.concat_tables(tables)
        pq.write_table(
            combined,
            str(self.output_path),
            compression     = cfg.PARQUET_COMPRESSION,
            use_dictionary  = True,   # good for category columns
        )
        final_mb = self.output_path.stat().st_size / 1024 ** 2
        logger.info(f"Final Parquet size on disk: {final_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M5 Ingestion Pipeline")
    parser.add_argument(
        "--chunk-size", type=int, default=cfg.MELT_CHUNK_SIZE,
        help="Day-columns per chunk (lower = less RAM, slower)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force reprocess even if output already exists"
    )
    args = parser.parse_args()

    prep = M5Preprocessor(
        chunk_size      = args.chunk_size,
        force_reprocess = args.force,
    )
    out_path = prep.run()
    print(f"\n✅ Pipeline complete. Processed data at: {out_path}")