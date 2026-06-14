import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


OPTIMIZATION_METHODS = ("GD", "MM", "NAG")
GROUPS = ("discrete", "continuous_h400")
GROUP_LABELS = {
    "discrete": "discrete",
    "continuous_h400": "continuous (h=400)",
}
METHOD_COLORS = {
    "GD": "tab:blue",
    "MM": "tab:orange",
    "NAG": "tab:green",
}
GROUP_LINESTYLES = {
    "discrete": "-",
    "continuous_h400": "--",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot KL-divergence trajectories over k for the discrete and continuous "
            "h=400 comparison outputs."
        )
    )
    parser.add_argument(
        "--costs-csv",
        default="outputs/comparison_embedding_stage_gd_mm_nag_h400_only/costs_h400_only.csv",
        help="Long-format costs CSV written by compute_embedding_stage_gd_mm_nag_h400_only.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures. Defaults to the directory containing --costs-csv.",
    )
    parser.add_argument(
        "--figure-name",
        default="kl_divergence_h400_only_by_k.pdf",
        help="Filename of the saved PDF.",
    )
    return parser.parse_args()


def make_stage_plot(axis, costs_df, stage, title):
    if stage is None:
        stage_df = costs_df.copy()
    else:
        stage_df = costs_df[costs_df["stage"] == stage].copy()

    for method in OPTIMIZATION_METHODS:
        for group in GROUPS:
            subset = stage_df[
                (stage_df["method"] == method)
                & (stage_df["group"] == group)
            ].sort_values("iteration")

            if subset.empty:
                continue

            axis.plot(
                subset["iteration"],
                subset["cost"],
                color=METHOD_COLORS[method],
                linestyle=GROUP_LINESTYLES[group],
                linewidth=1.8,
                label=f"{method} / {GROUP_LABELS[group]}",
            )

    axis.set_title(title, fontsize=18)
    axis.set_xlabel("k")
    axis.set_ylabel("KL divergence")
    axis.grid(True, alpha=0.25)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.costs_csv))
    os.makedirs(output_dir, exist_ok=True)

    costs_df = pd.read_csv(args.costs_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), squeeze=False)
    ee_axis = axes[0, 0]
    embedding_axis = axes[0, 1]

    make_stage_plot(
        axis=ee_axis,
        costs_df=costs_df,
        stage="early_exaggeration",
        title="Early exaggeration stage",
    )
    make_stage_plot(
        axis=embedding_axis,
        costs_df=costs_df,
        stage=None,
        title="EE + embedding stages",
    )

    handles, labels = ee_axis.get_legend_handles_labels()
    embedding_axis.legend(
        handles,
        labels,
        loc="center right",
        fontsize=12,
        framealpha=0.9,
    )
    fig.suptitle("KL-divergence trajectory by k", y=0.98, fontsize=20)
    fig.subplots_adjust(top=0.84, wspace=0.22)

    output_path = os.path.join(output_dir, args.figure_name)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(" ", output_path)


if __name__ == "__main__":
    main()