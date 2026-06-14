#!/usr/bin/env python3
"""
Plot ORC/SRC gap summaries for SRC(MST) and SRC(SPT).

This version creates:
  1. individual metric figures
  2. one combined 2 x 2 panel figure

Each panel overlays cycle rank beta_1 on a secondary y-axis as a
thin gray dashed line. In this experiment beta_1 is also the number
of non-tree edges discarded when projecting the graph back to a tree.

Example:
  python experiments/orc_src_cycle_gap/plot_orc_src_gap_compare_methods.py \
    --mst-summary outputs/cycle_controlled_grid/orc_src_gap_mst/orc_src_gap_summary.csv \
    --spt-summary outputs/cycle_controlled_grid/orc_src_gap_spt/orc_src_gap_summary.csv \
    --x eta_actual \
    --output-dir outputs/cycle_controlled_grid/orc_src_gap_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_COLORS = {
    "SRC(MST)": "tab:orange",
    "SRC(SPT)": "tab:green",
}

BETA1_COLOR = "0.55"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ORC/SRC gap and beta_1 for SRC(MST) and SRC(SPT)."
    )
    parser.add_argument("--mst-summary", required=True)
    parser.add_argument("--spt-summary", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--x",
        default="eta_actual",
        choices=["beta1", "eta_actual", "eta_requested"],
        help="Column used for the horizontal axis. Default: eta_actual.",
    )
    parser.add_argument("--no-markers", action="store_true")
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--individual-only", action="store_true")
    parser.add_argument(
        "--no-beta1",
        action="store_true",
        help="Do not overlay beta_1 on the secondary y-axis.",
    )
    parser.add_argument(
        "--beta1-label",
        default=r"Cycle rank $\beta_1$ / cut edges",
        help="Label for the secondary y-axis.",
    )
    return parser.parse_args()


def load_summary(path: str | Path, method_label: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "beta1" not in df.columns:
        raise ValueError(f"{path} does not contain required column 'beta1'.")
    df = df.copy()
    df["src_method_label"] = method_label
    df["source_csv"] = str(path)
    return df


def get_x_label(x_col: str) -> str:
    if x_col == "beta1":
        return r"Cycle rank $\beta_1$"
    if x_col == "eta_actual":
        return r"Actual $\eta$"
    if x_col == "eta_requested":
        return r"Requested $\eta$"
    return x_col


def beta1_curve(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' is not found in summary CSV.")
    out = df[[x_col, "beta1"]].dropna().drop_duplicates()
    out = out.groupby(x_col, as_index=False)["beta1"].mean()
    return out.sort_values(x_col)


def add_beta1_axis(
    ax,
    df_ref: pd.DataFrame,
    x_col: str,
    beta1_label: str,
    show_label: bool,
):
    bdf = beta1_curve(df_ref, x_col)
    ax2 = ax.twinx()
    ax2.plot(
        bdf[x_col],
        bdf["beta1"],
        linestyle="--",
        marker="o",
        markersize=5,
        linewidth=1.6,
        color=BETA1_COLOR,
        markerfacecolor=BETA1_COLOR,
        markeredgecolor=BETA1_COLOR,
        alpha=0.65,
        label=r"$\beta_1$",
        zorder=1,
    )
    ax2.set_ylabel(beta1_label if show_label else "", color=BETA1_COLOR)
    ax2.tick_params(axis="y", colors=BETA1_COLOR)
    ax2.grid(False)
    return ax2


def plot_one_axis(
    ax,
    df_mst: pd.DataFrame,
    df_spt: pd.DataFrame,
    x_col: str,
    y_col: str,
    ylabel: str,
    title: str,
    markers: bool = True,
    show_legend: bool = False,
    show_beta1: bool = True,
    beta1_label: str = r"Cycle rank $\beta_1$ / cut edges",
    show_beta1_label: bool = True,
):
    marker = "o" if markers else None

    for df, label in [(df_mst, "SRC(MST)"), (df_spt, "SRC(SPT)")]:
        if y_col not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"Missing column: {y_col}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return None

        if x_col not in df.columns:
            raise ValueError(f"Column '{x_col}' is not found in {label} summary CSV.")

        plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
        ax.plot(
            plot_df[x_col],
            plot_df[y_col],
            marker=marker,
            linewidth=2,
            label=label,
            color=METHOD_COLORS[label],
            zorder=3,
        )

    ax.set_xlabel(get_x_label(x_col))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=20)

    ax2 = None
    if show_beta1:
        ax2 = add_beta1_axis(
            ax=ax,
            df_ref=df_mst,
            x_col=x_col,
            beta1_label=beta1_label,
            show_label=show_beta1_label,
        )

    if show_legend:
        h1, l1 = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=True)
        else:
            ax.legend(h1, l1, frameon=True)

    return ax2


def plot_metric(
    df_mst: pd.DataFrame,
    df_spt: pd.DataFrame,
    x_col: str,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    markers: bool = True,
    show_beta1: bool = True,
    beta1_label: str = r"Cycle rank $\beta_1$ / cut edges",
):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_one_axis(
        ax=ax,
        df_mst=df_mst,
        df_spt=df_spt,
        x_col=x_col,
        y_col=y_col,
        ylabel=ylabel,
        title=title,
        markers=markers,
        show_legend=True,
        show_beta1=show_beta1,
        beta1_label=beta1_label,
        show_beta1_label=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_2x2(
    df_mst: pd.DataFrame,
    df_spt: pd.DataFrame,
    x_col: str,
    metrics,
    output_path: Path,
    markers: bool = True,
    show_beta1: bool = True,
    beta1_label: str = r"Cycle rank $\beta_1$ / cut edges",
):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), squeeze=False)
    flat_axes = axes.ravel()

    for idx, (ax, (col, ylabel, title, _filename)) in enumerate(zip(flat_axes, metrics)):
        plot_one_axis(
            ax=ax,
            df_mst=df_mst,
            df_spt=df_spt,
            x_col=x_col,
            y_col=col,
            ylabel=ylabel,
            title=title,
            markers=markers,
            show_legend=(idx == 3),
            show_beta1=show_beta1,
            beta1_label=beta1_label,
            show_beta1_label=(idx in (1, 3)),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.combined_only and args.individual_only:
        raise ValueError("--combined-only and --individual-only cannot both be set.")

    mst_path = Path(args.mst_summary)
    spt_path = Path(args.spt_summary)

    out_dir = Path(args.output_dir) if args.output_dir is not None else mst_path.parent / "compare_mst_spt"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_mst = load_summary(mst_path, "SRC(MST)")
    df_spt = load_summary(spt_path, "SRC(SPT)")

    combined = pd.concat([df_mst, df_spt], ignore_index=True)
    combined_path = out_dir / "orc_src_gap_summary_compare_mst_spt.csv"
    combined.to_csv(combined_path, index=False)

    metrics_2x2 = [
        ("mean_abs_gap", "Mean |ORC - SRC|", "Mean absolute gap", "mean_abs_gap_compare_mst_spt.pdf"),
        ("rmse_gap", "RMSE(ORC - SRC)", "RMSE gap", "rmse_gap_compare_mst_spt.pdf"),
        ("spearman_orc_src", "Spearman correlation", "Spearman correlation", "spearman_compare_mst_spt.pdf"),
        ("sign_mismatch_rate", "Sign mismatch rate", "Sign mismatch rate", "sign_mismatch_compare_mst_spt.pdf"),
    ]

    extra_individual_metrics = [
        ("median_abs_gap", "Median |ORC - SRC|", "Median absolute gap", "median_abs_gap_compare_mst_spt.pdf"),
        ("pearson_orc_src", "Pearson correlation", "Pearson correlation", "pearson_compare_mst_spt.pdf"),
    ]

    saved = [combined_path]
    show_beta1 = not args.no_beta1

    if not args.individual_only:
        combined_fig_path = out_dir / "orc_src_gap_compare_mst_spt_2x2_with_beta1.pdf"
        plot_combined_2x2(
            df_mst=df_mst,
            df_spt=df_spt,
            x_col=args.x,
            metrics=metrics_2x2,
            output_path=combined_fig_path,
            markers=not args.no_markers,
            show_beta1=show_beta1,
            beta1_label=args.beta1_label,
        )
        saved.append(combined_fig_path)

    if not args.combined_only:
        for col, ylabel, title, filename in metrics_2x2 + extra_individual_metrics:
            if col in df_mst.columns or col in df_spt.columns:
                stem = Path(filename).stem
                suffix = "_with_beta1.pdf" if show_beta1 else ".pdf"
                output_path = out_dir / f"{stem}{suffix}"
                plot_metric(
                    df_mst=df_mst,
                    df_spt=df_spt,
                    x_col=args.x,
                    y_col=col,
                    ylabel=ylabel,
                    title=f"{title} vs {get_x_label(args.x)}",
                    output_path=output_path,
                    markers=not args.no_markers,
                    show_beta1=show_beta1,
                    beta1_label=args.beta1_label,
                )
                saved.append(output_path)

    print("Saved:")
    for path in saved:
        print(" ", path)


if __name__ == "__main__":
    main()
