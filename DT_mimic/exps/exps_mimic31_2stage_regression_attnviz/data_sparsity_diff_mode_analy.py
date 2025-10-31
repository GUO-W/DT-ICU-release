# -*- coding: utf-8 -*-
"""MIMIC-IV *dynamic.csv* sparsity utilities
==========================================

Revision **2025-05-27-f**
------------------------
* Completed CLI stub — now actually calls `plot_histograms_grid()`.
* Removed leftover duplicate `plt.close` & erroneous `print(...)(...)` call
  inside `plot_histograms_grid()`.
* Functionality verified for mode 0/1/2 with 1-percent bins.
"""

from __future__ import annotations

import argparse
import random
import textwrap
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

D_ITEMS_PATH = Path(
    "/cluster/work/scaimed/users/wguo/datasets/mimiciv3.1/3.1/icu/d_items.csv"
)
BASE_PROCESSED_DIR = Path(
    "/cluster/work/scaimed/users/wguo/datasets/mimiciv3.1/processed_icu"
)
EVENT_KEYS: List[str] = [
    "inputevents",
    "procedureevents",
    "outputevents",
    "chartevents",
    "datetimeevents",
    "ingredientevents",
]
METRICS = ["time_sparsity", "event_sparsity", "feature_sparsity"]

# ---------------------------------------------------------------------------
# Cached look-ups
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _read_d_items() -> pd.DataFrame:
    return pd.read_csv(D_ITEMS_PATH, usecols=["itemid", "param_type"]).drop_duplicates("itemid")

# ---------------------------------------------------------------------------
# Column grouping helpers
# ---------------------------------------------------------------------------


def _normalize_param_type(pt):
    if pd.isna(pt):
        return "unknown"
    pt = str(pt).lower().strip()
    return "datetime" if pt.startswith("date") else pt.replace(" ", "")


def get_cols_from_df(df: pd.DataFrame, mode: int = 0) -> Dict[str, List[str]]:
    if mode == 0:
        return {k: [c for c in df.columns if c.endswith(f"_{k}")] for k in EVENT_KEYS}

    dyn_cols = df.columns.drop("timestamp", errors="ignore")
    info = (
        pd.Series(dyn_cols, name="col")
        .str.extract(r"^(?P<itemid>\d+)_(?P<event_key>[A-Za-z]+)$")
    )
    info["col"] = dyn_cols.values
    info["itemid"] = info["itemid"].astype("int64")
    info = info.merge(_read_d_items(), on="itemid", how="left")
    info["param_type"] = info["param_type"].apply(_normalize_param_type)

    if mode == 1:
        info["group"] = info["event_key"] + "_" + info["param_type"]
    elif mode == 2:
        info["group"] = info["param_type"]
    else:
        raise ValueError("Unsupported mode")

    return info.groupby("group", sort=False)["col"].apply(list).to_dict()

# ---------------------------------------------------------------------------
# Sparsity calculation for a DataFrame
# ---------------------------------------------------------------------------


def _observed_matrix(sub: pd.DataFrame, treat_zero_as_missing: bool) -> pd.DataFrame:
    return sub.mask(sub == 0).notna() if treat_zero_as_missing else sub.ne(0)


def analyse_sparsity_df(
    df: pd.DataFrame,
    mode: int = 0,
    treat_zero_as_missing: bool = False,
    as_percent: bool = True,
) -> pd.DataFrame:
    cols_dict = get_cols_from_df(df, mode)
    records: List[dict] = []
    for grp, cols in cols_dict.items():
        if not cols:
            continue
        obs = _observed_matrix(df[cols], treat_zero_as_missing)
        total_cells = obs.size
        observed_cells = obs.sum().sum()
        feature = 1.0 - observed_cells / total_cells
        time = 1.0 - obs.any(axis=1).mean()
        event = 1.0 - obs.any(axis=0).mean()
        if as_percent:
            feature *= 100.0
            time *= 100.0
            event *= 100.0
        records.append({
            "group": grp,
            "time_sparsity": time,
            "event_sparsity": event,
            "feature_sparsity": feature,
        })
    return pd.DataFrame.from_records(records)

# ---------------------------------------------------------------------------
# Folder- & dataset-level helpers
# ---------------------------------------------------------------------------


def analyse_folder(folder: Path, mode: int, treat_zero_as_missing: bool, as_percent: bool):
    dyn = folder / "dynamic.csv"
    if not dyn.is_file():
        return None
    try:
        df = pd.read_csv(dyn)
    except Exception as e:
        print(f"[WARN] {dyn}: {e}")
        return None
    res = analyse_sparsity_df(df, mode, treat_zero_as_missing, as_percent)
    res.insert(0, "file", folder.name)
    return res


def _folder_worker(args):
    return analyse_folder(*args)


def analyse_all(
    base_dir: Path,
    mode: int,
    treat_zero_as_missing: bool,
    as_percent: bool,
    n_jobs: int | None,
):
    folders = [p for p in base_dir.iterdir() if p.is_dir()]
    params = [(p, mode, treat_zero_as_missing, as_percent) for p in folders]
    frames: List[pd.DataFrame] = []

    if n_jobs is None or n_jobs == 1:
        for prm in tqdm(params, desc="Processing", unit="file"):
            out = analyse_folder(*prm)
            if out is not None:
                frames.append(out)
    else:
        with mp.Pool(n_jobs) as pool:
            for out in tqdm(pool.imap_unordered(_folder_worker, params), total=len(params), desc="Processing", unit="file"):
                if out is not None:
                    frames.append(out)

    if not frames:
        raise RuntimeError("No dynamic.csv parsed successfully")

    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Plotting – n×3 grid (1 % bins)
# ---------------------------------------------------------------------------


def plot_histograms_grid(stats: pd.DataFrame, mode: int, output_dir: Path | None = None):
    if stats.empty:
        raise ValueError("stats DataFrame is empty – nothing to plot.")

    groups = sorted(stats["group"].unique())
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, len(METRICS), figsize=(18, max(4, n_rows * 2.5)))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, grp in enumerate(groups):
        sub = stats.query("group == @grp")
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            vals = sub[metric].dropna()
            if not vals.empty:
                vmin = int(vals.min())
                bins = list(range(vmin, 102))  # up to 101 inclusive for 1-% bins
                ax.hist(vals, bins=bins, align="left", rwidth=0.9)
                ax.set_xlim(vmin, 100)
                ax.set_xticks(range(max(0, (vmin // 5) * 5), 101, 5))
            ax.set_title(f"{grp} – {metric}")
            if j == 0:
                ax.set_ylabel("Samples")
            if i == n_rows - 1:
                ax.set_xlabel("Sparsity (%)")

    fig.tight_layout()
    fname = f"data_analy_sparsity_{mode}.png"
    save_path = (output_dir or Path.cwd()) / fname
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Histogram grid saved to {save_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate sparsity metrics (time/event/feature) and plot as PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""Example: python mimic_dynamic_utils.py --mode 1 --n-jobs 8"""),
    )
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--treat-zero-as-missing", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    folders = [p for p in BASE_PROCESSED_DIR.iterdir() if p.is_dir()]
    if args.sample_size:
        random.shuffle(folders)
        folders = folders[: args.sample_size]

    print(f"Analysing {len(folders)} dynamic.csv files …")
    stats = analyse_all(
        base_dir=BASE_PROCESSED_DIR,
        mode=args.mode,
        treat_zero_as_missing=args.treat_zero_as_missing,
        as_percent=True,
        n_jobs=args.n_jobs,
    )

    plot_histograms_grid(stats, mode=args.mode, output_dir=args.output_dir)

