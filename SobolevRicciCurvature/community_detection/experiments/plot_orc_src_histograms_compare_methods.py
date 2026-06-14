#!/usr/bin/env python3
"""
Compare ORC, SRC(MST), and SRC(SPT) histograms for selected eta values.

Layout:
  row 1: SRC(SPT)  light green
  row 2: SRC(MST)  orange
  row 3: ORC       pink

Default x-axis behavior:
  For each eta column, the x-axis range and histogram bins are shared across
  SRC(SPT), SRC(MST), and ORC. Different eta columns may have different x-axis
  ranges.

Example:
  python experiments/orc_src_cycle_gap/plot_orc_src_histograms_compare_methods.py \
    --mst-edge-csv outputs/cycle_controlled_grid/orc_src_gap_mst/orc_src_edge_curvatures.csv \
    --spt-edge-csv outputs/cycle_controlled_grid/orc_src_gap_spt/orc_src_edge_curvatures.csv \
    --etas 0 0.1 0.2 1.0 \
    --edge-set tree \
    --output-dir outputs/cycle_controlled_grid/orc_src_gap_compare_hist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "SRC(SPT)": "#8fd18f",  # light green
    "SRC(MST)": "#f4a261",  # orange
    "ORC": "#ff66cc",       # pink
}

ROW_ORDER = ["SRC(SPT)", "SRC(MST)", "ORC"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ORC/SRC(MST)/SRC(SPT) histograms for selected eta values."
    )
    parser.add_argument(
        "--mst-edge-csv",
        required=True,
        help="Path to orc_src_edge_curvatures.csv produced with --src-method src-mst.",
    )
    parser.add_argument(
        "--spt-edge-csv",
        required=True,
        help="Path to orc_src_edge_curvatures.csv produced with --src-method src-spt.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--etas",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 1.0],
        help="Target eta values to visualize.",
    )
    parser.add_argument(
        "--eta-column",
        choices=["eta_actual", "eta_requested"],
        default="eta_actual",
    )
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--edge-set",
        default=None,
        help="Optional filter, e.g. tree / all / extra. Applied if edge_set exists.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use density=True instead of raw counts.",
    )
    parser.add_argument(
        "--xscale",
        choices=["column", "global", "free"],
        default="column",
        help=(
            "column: same x-axis within each eta column. "
            "global: same x-axis for all panels. "
            "free: each panel has its own x-axis."
        ),
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Manually set x-axis limits for all panels, e.g. --xlim -10 1.",
    )
    parser.add_argument(
        "--percentile-xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Set x-axis limits using percentiles. "
            "With --xscale column, this is computed per eta column."
        ),
    )
    parser.add_argument(
        "--show-suptitle",
        action="store_true",
        help="Add a figure-level title. Default: no suptitle.",
    )
    parser.add_argument(
        "--no-column-labels",
        action="store_true",
        help="Do not add bottom labels like (a) eta=0.1.",
    )
    return parser.parse_args()


def load_edges(path: str | Path, edge_set: str | None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    required = {"orc", "src"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if edge_set is not None and "edge_set" in df.columns:
        df = df[df["edge_set"] == edge_set].copy()

    return df


def nearest_eta(df: pd.DataFrame, target_eta: float, eta_column: str) -> float:
    if eta_column not in df.columns:
        raise ValueError(f"CSV does not contain eta column '{eta_column}'.")

    available = np.sort(df[eta_column].dropna().astype(float).unique())
    if len(available) == 0:
        raise ValueError(f"No eta values found in column '{eta_column}'.")

    return float(available[np.argmin(np.abs(available - target_eta))])


def selected_values_for_eta(
    df_mst: pd.DataFrame,
    df_spt: pd.DataFrame,
    eta: float,
    eta_column: str,
) -> dict:
    eta_mst = nearest_eta(df_mst, eta, eta_column)
    eta_spt = nearest_eta(df_spt, eta, eta_column)

    mst_sub = df_mst[np.isclose(df_mst[eta_column].astype(float), eta_mst)].copy()
    spt_sub = df_spt[np.isclose(df_spt[eta_column].astype(float), eta_spt)].copy()

    return {
        "eta_requested": float(eta),
        "eta_mst_used": eta_mst,
        "eta_spt_used": eta_spt,
        "SRC(SPT)": spt_sub["src"].to_numpy(dtype=float),
        "SRC(MST)": mst_sub["src"].to_numpy(dtype=float),
        "ORC": mst_sub["orc"].to_numpy(dtype=float),
    }


def finite_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def concat_finite(arrays) -> np.ndarray:
    clean = [finite_values(a) for a in arrays]
    clean = [a for a in clean if len(a) > 0]
    if not clean:
        return np.asarray([], dtype=float)
    return np.concatenate(clean)


def padded_xlim(values: np.ndarray, percentile_xlim=None):
    values = finite_values(values)
    if len(values) == 0:
        return (-1.0, 1.0)

    if percentile_xlim is not None:
        xmin, xmax = np.percentile(values, percentile_xlim)
    else:
        xmin, xmax = np.min(values), np.max(values)

    xmin = float(xmin)
    xmax = float(xmax)

    if np.isclose(xmin, xmax):
        eps = 1e-6 if np.isclose(xmin, 0.0) else abs(xmin) * 1e-6
        xmin -= eps
        xmax += eps
    else:
        pad = 0.03 * (xmax - xmin)
        xmin -= pad
        xmax += pad

    return xmin, xmax


def compute_column_xlims(data_by_eta, args):
    if args.xlim is not None:
        xlim = (float(args.xlim[0]), float(args.xlim[1]))
        return [xlim for _ in data_by_eta]

    if args.xscale == "global":
        all_values = concat_finite(
            [item[method] for item in data_by_eta for method in ROW_ORDER]
        )
        xlim = padded_xlim(all_values, args.percentile_xlim)
        return [xlim for _ in data_by_eta]

    if args.xscale == "column":
        xlims = []
        for item in data_by_eta:
            values = concat_finite([item[method] for method in ROW_ORDER])
            xlims.append(padded_xlim(values, args.percentile_xlim))
        return xlims

    return None  # free scale


def bins_from_xlim(xlim, bins: int):
    return np.linspace(xlim[0], xlim[1], bins + 1)


def subplot_letter(j: int) -> str:
    return chr(ord("a") + j)


def plot_split(data_by_eta, output_path: Path, args):
    ncols = len(data_by_eta)
    fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 8.6), squeeze=False)

    column_xlims = compute_column_xlims(data_by_eta, args)

    for j, item in enumerate(data_by_eta):
        eta_label = item["eta_requested"]

        if column_xlims is not None:
            column_xlim = column_xlims[j]
            column_bins = bins_from_xlim(column_xlim, args.bins)
        else:
            column_xlim = None
            column_bins = None

        for i, method in enumerate(ROW_ORDER):
            ax = axes[i, j]
            values = item[method]

            if args.xscale == "free" and args.xlim is None:
                xlim = padded_xlim(values, args.percentile_xlim)
                bin_edges = bins_from_xlim(xlim, args.bins)
            else:
                xlim = column_xlim
                bin_edges = column_bins

            ax.hist(
                values,
                bins=bin_edges,
                density=args.normalize,
                color=METHOD_COLORS[method],
                alpha=0.9,
                edgecolor="none",
            )
            ax.set_xlim(*xlim)
            ax.set_title(f"{method}, $\eta={eta_label:g}$", fontsize=20, pad=3)

            if j == 0:
                ax.set_ylabel("Density" if args.normalize else "Count", fontsize=12)
            else:
                ax.set_ylabel("")

            if i == len(ROW_ORDER) - 1:
                ax.set_xlabel("Curvature", fontsize=10)
            else:
                ax.set_xlabel("")

            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=10)

        if not args.no_column_labels:
            axes[-1, j].text(
                0.5,
                -0.40,
                f"({subplot_letter(j)}) $\\eta={eta_label:g}$",
                transform=axes[-1, j].transAxes,
                ha="center",
                va="top",
                fontsize=13,
            )

    if args.show_suptitle:
        fig.suptitle("ORC / SRC(MST) / SRC(SPT) histogram comparison", fontsize=13)

    fig.tight_layout()
    bottom = 0.13 if not args.no_column_labels else 0.04
    top = 0.94 if args.show_suptitle else 0.98
    fig.subplots_adjust(hspace=0.42, wspace=0.25, bottom=bottom, top=top)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(data_by_eta, output_path: Path):
    rows = []
    for item in data_by_eta:
        row = {
            "eta_requested": item["eta_requested"],
            "eta_mst_used": item["eta_mst_used"],
            "eta_spt_used": item["eta_spt_used"],
        }
        for method in ROW_ORDER:
            vals = pd.Series(item[method])
            key = method.lower().replace("(", "_").replace(")", "").replace(" ", "_")
            row[f"{key}_n"] = int(vals.notna().sum())
            row[f"{key}_mean"] = float(vals.mean())
            row[f"{key}_std"] = float(vals.std())
            row[f"{key}_min"] = float(vals.min())
            row[f"{key}_max"] = float(vals.max())
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    args = parse_args()

    mst_path = Path(args.mst_edge_csv)
    spt_path = Path(args.spt_edge_csv)

    out_dir = Path(args.output_dir) if args.output_dir is not None else mst_path.parent / "compare_hist_mst_spt"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_mst = load_edges(mst_path, args.edge_set)
    df_spt = load_edges(spt_path, args.edge_set)

    data_by_eta = [
        selected_values_for_eta(df_mst, df_spt, eta, args.eta_column)
        for eta in args.etas
    ]

    summary_path = out_dir / "orc_src_hist_compare_mst_spt_selected_eta_summary.csv"
    write_summary(data_by_eta, summary_path)

    norm = "density" if args.normalize else "count"
    xscale = "manual_x" if args.xlim is not None else f"{args.xscale}_x"
    fig_path = out_dir / f"orc_src_histograms_compare_mst_spt_split_{norm}_{xscale}.pdf"

    plot_split(data_by_eta, fig_path, args)

    print("Saved:")
    print(" ", summary_path)
    print(" ", fig_path)


if __name__ == "__main__":
    main()
