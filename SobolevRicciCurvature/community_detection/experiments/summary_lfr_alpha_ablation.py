import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================
# Config
# =========================
IN_CSV  = "./results/lfr_src_alpha.csv"
OUT_PDF = "./results/result_alpha_ablation.pdf"

METHOD_LEFT  = "src-spt"
METHOD_RIGHT = "src-mst"

DISPLAY_NAME = {
    "src-spt": "SRC(SPT)",
    "src-mst": "SRC(MST)",
}

SCORE_COLS = ["best_ari_louvain"]  # e.g., ["best_ari_louvain", "best_ari_length"]
MEAN_YLIM = (-0.02, 1.02)
STD_PAD_RATIO = 0.05


def style_cycle_for_alphas(alphas):
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "x", "+"]
    linestyles = ["-", "--", "-.", ":"]
    alpha_style = {}
    i = 0
    for a in alphas:
        alpha_style[a] = {
            "marker": markers[i % len(markers)],
            "linestyle": linestyles[(i // len(markers)) % len(linestyles)],
        }
        i += 1
    return alpha_style


def plot_two_methods_one_page(pdf: PdfPages, g: pd.DataFrame, score_type: str):
    g2 = g[
        (g["score_type"] == score_type) &
        (g["method"].isin([METHOD_LEFT, METHOD_RIGHT]))
    ].copy()

    if g2.empty:
        print(f"[skip] no data for score_type={score_type} with methods {METHOD_LEFT}/{METHOD_RIGHT}")
        return

    alphas = sorted(g2["alpha"].dropna().unique().tolist())
    if len(alphas) == 0:
        print(f"[skip] no alpha values for score_type={score_type}")
        return

    # =========================
    # Global y-limits (shared across left/right)
    # =========================
    std_max = float(np.nanmax(g2["std"].to_numpy()))
    std_top = std_max * (1.0 + STD_PAD_RATIO) if std_max > 0 else 1.0
    std_ylim = (0.0, std_top)

    mean_ylim = MEAN_YLIM

    alpha_style = style_cycle_for_alphas(alphas)

    # =========================
    # Make 2x2 figure
    # =========================
    fig, axs = plt.subplots(
        2, 2, sharex=True,
        figsize=(11.0, 6.0),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.10, "wspace": 0.10},
        constrained_layout=True
    )
    ax_std_L, ax_std_R = axs[0, 0], axs[0, 1]
    ax_mean_L, ax_mean_R = axs[1, 0], axs[1, 1]

    handles, labels = [], []

    def plot_one(method: str, ax_std, ax_mean, collect_legend: bool):
        sub_m = g2[g2["method"] == method].copy()

        for a in alphas:
            sub = sub_m[sub_m["alpha"] == a].sort_values("mu")
            if sub.empty:
                continue

            x = sub["mu"].to_numpy()
            y_mean = sub["mean"].to_numpy()
            y_std  = sub["std"].to_numpy()

            st = alpha_style[a]
            label = rf"$\alpha={a:g}$"

            ax_std.plot(
                x, y_std,
                label=label,
                linewidth=2.0,
                markersize=5,
                alpha=0.85,
                **st
            )
            h = ax_mean.plot(
                x, y_mean,
                label=label,
                linewidth=2.0,
                markersize=5,
                alpha=0.90,
                **st
            )[0]

            if collect_legend and (label not in labels):
                handles.append(h)
                labels.append(label)

        ax_std.set_title(DISPLAY_NAME.get(method, method))
        ax_std.grid(True)
        ax_mean.grid(True)

    plot_one(METHOD_LEFT,  ax_std_L, ax_mean_L, collect_legend=False)
    plot_one(METHOD_RIGHT, ax_std_R, ax_mean_R, collect_legend=True)

    # =========================
    # Axis labels + shared y-limits
    # =========================
    ax_std_L.set_ylabel("std")
    ax_mean_L.set_ylabel("ARI mean" if "ari" in score_type else "mean")

    ax_mean_L.set_xlabel(r"$\mu$")
    ax_mean_R.set_xlabel(r"$\mu$")

    ax_std_R.set_ylabel("")
    ax_mean_R.set_ylabel("")

    ax_std_L.set_ylim(*std_ylim)
    ax_std_R.set_ylim(*std_ylim)
    ax_mean_L.set_ylim(*mean_ylim)
    ax_mean_R.set_ylim(*mean_ylim)

    ax_mean_R.legend(
        handles, labels,
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        ncol=1,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3
    )

    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    required = ["method", "mu", "seed", "alpha"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available columns: {list(df.columns)}")

    for sc in SCORE_COLS:
        if sc not in df.columns:
            raise ValueError(f"Score column '{sc}' not found. Available columns: {list(df.columns)}")

    df["mu"] = pd.to_numeric(df["mu"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    for sc in SCORE_COLS:
        df[sc] = pd.to_numeric(df[sc], errors="coerce")

    df_long = df.melt(
        id_vars=["method", "mu", "seed", "alpha"],
        value_vars=SCORE_COLS,
        var_name="score_type",
        value_name="score",
    )

    g = (
        df_long.groupby(["method", "score_type", "mu", "alpha"])["score"]
              .agg(["mean", "std", "count"])
              .reset_index()
              .sort_values(["score_type", "method", "alpha", "mu"])
    )
    g["std"] = g["std"].fillna(0.0)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

    with PdfPages(OUT_PDF) as pdf:
        for score_type in SCORE_COLS:
            plot_two_methods_one_page(pdf, g, score_type=score_type)

    print("saved:", OUT_PDF)

if __name__ == "__main__":
    main()
