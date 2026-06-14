import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


OPTIMIZATION_METHODS = ("GD", "MM", "NAG")
EE_GROUPS = ("discrete", "continuous_h400")


def parse_args():
    """
    Parse command-line options for plotting.

    - inputs:
        - CLI arguments provided by the user.
    - outputs:
        - argparse.Namespace with resolved plotting and path options.
    - process:
        - Define flags/defaults/help text and parse arguments.
    """
    parser = argparse.ArgumentParser(
        description="Read computed npz/csv outputs and create 2x3 EE/final embedding plots."
    )
    parser.add_argument(
        "--npz",
        default="outputs/comparison_embedding_stage_gd_mm_nag_h400_only/comparison_embedding_stage_gd_mm_nag_h400_only.npz",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional summary CSV. The plot can be created from the npz alone.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures. Defaults to the directory containing the npz.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Plot raw coordinates instead of per-panel display normalization.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
    )
    return parser.parse_args()


def _scalar(npz, key, default=None):
    """
    Safely extract one scalar value from npz.

    - inputs:
        - npz archive, key name, and default fallback.
    - outputs:
        - Python scalar if available, otherwise default.
    - process:
        - Check key existence and non-empty array, then unwrap to scalar.
    """
    if key not in npz:
        return default
    arr = npz[key]
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def normalize_for_display(embedding):
    """
    Normalize coordinates for visual comparison.

    - inputs:
        - Embedding array of shape (N, 2).
    - outputs:
        - Normalized coordinates and x/y axis limits.
    - process:
        - Center points, scale by per-axis span, and add padded limits.
    """
    center = np.mean(embedding, axis=0)
    centered = embedding - center[np.newaxis, :]
    span = np.maximum(centered.max(axis=0) - centered.min(axis=0), 1e-12)
    display = centered / span[np.newaxis, :]
    min_xy = display.min(axis=0)
    max_xy = display.max(axis=0)
    padding = 0.08 * np.maximum(max_xy - min_xy, 1e-12)
    x_limits = (min_xy[0] - padding[0], max_xy[0] + padding[0])
    y_limits = (min_xy[1] - padding[1], max_xy[1] + padding[1])
    return display, x_limits, y_limits


def raw_for_display(embedding):
    """
    Prepare raw coordinates and limits for plotting.

    - inputs:
        - Embedding array of shape (N, 2).
    - outputs:
        - Raw coordinates and x/y axis limits.
    - process:
        - Keep coordinates unchanged and compute padded min/max limits.
    """
    display = embedding
    min_xy = display.min(axis=0)
    max_xy = display.max(axis=0)
    padding = 0.08 * np.maximum(max_xy - min_xy, 1e-12)
    x_limits = (min_xy[0] - padding[0], max_xy[0] + padding[0])
    y_limits = (min_xy[1] - padding[1], max_xy[1] + padding[1])
    return display, x_limits, y_limits


def get_array(npz, group_name, method, stage):
    """
    Retrieve one stage-specific embedding array.

    - inputs:
        - npz archive, group name, method name, and stage name.
    - outputs:
        - ndarray for the requested stage.
    - process:
        - Build key prefix and select EE-state or final-embedding array.
    """
    prefix = f"ee_{group_name}_{method.lower()}"
    if stage == "ee":
        return npz[f"{prefix}_state"]
    if stage == "embedding":
        return npz[f"{prefix}_embedding_final"]
    raise ValueError(f"Unknown stage: {stage}")


def get_ee_iteration(npz, group_name, method):
    """
    Read EE iteration metadata for one setting.

    - inputs:
        - npz archive, group name, and method name.
    - outputs:
        - EE iteration scalar or None.
    - process:
        - Construct the key and read via the safe scalar helper.
    """
    prefix = f"ee_{group_name}_{method.lower()}"
    return _scalar(npz, f"{prefix}_ee_iteration", default=None)


def get_h(npz, group_name, method):
    """
    Read step-size h metadata for one setting.

    - inputs:
        - npz archive, group name, and method name.
    - outputs:
        - Step size h scalar or None.
    - process:
        - Construct the key and read via the safe scalar helper.
    """
    prefix = f"ee_{group_name}_{method.lower()}"
    return _scalar(npz, f"{prefix}_h", default=None)


def plot_grid(npz, stage, output_path, title, normalize=True, point_size=18.0):
    """
    Generate and save a 2x3 comparison scatter grid.

    - inputs:
        - npz archive, stage name, output path, figure title, and plot options.
    - outputs:
        - Saved figure file at output_path.
    - process:
        - Load arrays, map labels to colors, plot each panel, and save the figure.
    """
    labels = npz["labels"].astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    palette = sns.color_palette("deep", n_colors=len(unique_labels))
    color_map = {label: palette[index] for index, label in enumerate(unique_labels)}

    row_titles = {
        "discrete": "EE: discrete",
        "continuous_h400": "EE: continuous",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)

    for row_index, group_name in enumerate(EE_GROUPS):
        for column_index, method in enumerate(OPTIMIZATION_METHODS):
            key = (group_name, method)
            axis = axes[row_index, column_index]

            Y = get_array(npz, group_name, method, stage)
            if normalize:
                display, x_limits, y_limits = normalize_for_display(Y)
            else:
                display, x_limits, y_limits = raw_for_display(Y)

            for label in unique_labels:
                mask = labels == label
                axis.scatter(
                    display[mask, 0],
                    display[mask, 1],
                    s=point_size,
                    alpha=0.75,
                    color=color_map[label],
                    label=label,
                )

            k = get_ee_iteration(npz, group_name, method)
            k_text = f", k={int(k)}" if k is not None else ""

            axis.set_title(f"{row_titles[group_name]} / method={method}{k_text}", fontsize=14)
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
            axis.tick_params(axis="both", which="major", labelsize=7)

            # Use scientific notation such as 1e-3 to avoid long tick labels.
            axis.ticklabel_format(
                axis="both",
                style="sci",
                scilimits=(-2, 2),
                useOffset=False,
            )
            axis.xaxis.get_offset_text().set_size(7)
            axis.yaxis.get_offset_text().set_size(7)

            # Put legend inside the EE discrete / NAG subplot (top-right), lower-right.
            if row_index == 0 and column_index == 2:
                axis.legend(
                    title="label",
                    loc="lower right",
                    fontsize=8,
                    title_fontsize=8,
                    framealpha=0.85,
                    markerscale=1.2,
                )

    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(hspace=0.30, wspace=0.18, top=0.90)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_difference_checks(npz):
    """
    Print norm-based differences between two EE groups.

    - inputs:
        - npz archive containing EE and final embeddings.
    - outputs:
        - Console text with absolute and relative norm differences.
    - process:
        - Compare pairwise arrays per method and report summary statistics.
    """
    print("\\n=== Difference check: discrete vs continuous_h400 ===")
    for method in OPTIMIZATION_METHODS:
        Y_ee_disc = get_array(npz, "discrete", method, "ee")
        Y_ee_cont = get_array(npz, "continuous_h400", method, "ee")
        Y_final_disc = get_array(npz, "discrete", method, "embedding")
        Y_final_cont = get_array(npz, "continuous_h400", method, "embedding")

        ee_diff = np.linalg.norm(Y_ee_disc - Y_ee_cont)
        final_diff = np.linalg.norm(Y_final_disc - Y_final_cont)
        ee_rel = ee_diff / (np.linalg.norm(Y_ee_disc) + 1e-12)
        final_rel = final_diff / (np.linalg.norm(Y_final_disc) + 1e-12)

        print(
            f"{method}: "
            f"EE diff={ee_diff:.6e} (rel={ee_rel:.6e}), "
            f"final diff={final_diff:.6e} (rel={final_rel:.6e})"
        )


def main():
    """
    Run the full plotting workflow end to end.

    - inputs:
        - CLI arguments from the current process.
    - outputs:
        - Two saved figure files and console logs.
    - process:
        - Parse args, load npz, print checks, render grids, and write output paths.
    """
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(output_dir, exist_ok=True)

    npz = np.load(args.npz)
    normalize = not args.no_normalize
    suffix = "normalized" if normalize else "raw"

    ee_path = os.path.join(output_dir, f"comparison_ee_stage_end_gd_mm_nag_h400_only_{suffix}.pdf")
    embedding_path = os.path.join(output_dir, f"comparison_embedding_stage_gd_mm_nag_h400_only_{suffix}.pdf")

    print_difference_checks(npz)

    plot_grid(
        npz=npz,
        stage="ee",
        output_path=ee_path,
        title="EE stage end",
        normalize=normalize,
        point_size=args.point_size,
    )
    plot_grid(
        npz=npz,
        stage="embedding",
        output_path=embedding_path,
        title="Common original t-SNE embedding stage after EE variants",
        normalize=normalize,
        point_size=args.point_size,
    )

    print("\\nSaved:")
    print(" ", ee_path)
    print(" ", embedding_path)


if __name__ == "__main__":
    main()
