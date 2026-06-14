import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === input files ===
in_csv  = "./results/lfr_src_p_ablation.csv"
out_pdf = "./results/result_lfr_src_p_ablation_mu_series_method_p.pdf"


df = pd.read_csv(in_csv)

df["mu"] = df["mu"].astype(float)
df["src_p"] = df["src_p"].astype(float)

score_cols = ["best_ari_louvain", "best_ari_length"]
for c in score_cols:
    df[c] = df[c].astype(float)

df_long = df.melt(
    id_vars=["method", "mu", "src_p"],
    value_vars=score_cols,
    var_name="score_type",
    value_name="score"
)

g = (df_long.groupby(["method", "score_type", "src_p", "mu"])["score"]
       .agg(["mean", "std", "count"])
       .reset_index()
       .sort_values(["method", "score_type", "src_p", "mu"]))

g["std"] = g["std"].fillna(0.0)

def plot_onefig_mu_series_method_p(g, out_path):
    fig, axes = plt.subplots(
        2, 2,
        sharex=True,
        figsize=(11.5, 6.2),
        gridspec_kw={"height_ratios": [1, 3], "wspace": 0.18, "hspace": 0.10},
        constrained_layout=True
    )

    ax_std_luv, ax_std_len = axes[0, 0], axes[0, 1]
    ax_mean_luv, ax_mean_len = axes[1, 0], axes[1, 1]

    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", ">", "<"]
    linestyles = ["-", "--", "-.", ":"]

    series_list = sorted(
        g.groupby(["method", "src_p"]).groups.keys(),
        key=lambda x: (x[0], x[1])
    )

    max_std = 0.0
    for _, sub in g.groupby(["score_type"]):
        max_std = max(max_std, float(sub["std"].max()))
    max_std = max_std * 1.05

    for i, (method, pval) in enumerate(series_list):
        color = cmap(i % 20)
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        method_disp = "SRC(SPT)" if method == "src-spt" else method
        label = f"{method_disp} | p={pval:g}"

        # ---- Louvain ----
        sub = g[(g["method"] == method) &
                (g["src_p"] == pval) &
                (g["score_type"] == "best_ari_louvain")].sort_values("mu")

        if len(sub) > 0:
            x = sub["mu"].to_numpy()
            ax_std_luv.plot(
                x, sub["std"].to_numpy(),
                color=color, marker=marker, linestyle=ls,
                linewidth=2.0, markersize=5, alpha=0.85
            )
            ax_mean_luv.plot(
                x, sub["mean"].to_numpy(),
                color=color, marker=marker, linestyle=ls,
                linewidth=2.0, markersize=5, alpha=0.9,
                label=label
            )

        # ---- Length cut ----
        sub = g[(g["method"] == method) &
                (g["src_p"] == pval) &
                (g["score_type"] == "best_ari_length")].sort_values("mu")

        if len(sub) > 0:
            x = sub["mu"].to_numpy()
            ax_std_len.plot(
                x, sub["std"].to_numpy(),
                color=color, marker=marker, linestyle=ls,
                linewidth=2.0, markersize=5, alpha=0.85
            )
            ax_mean_len.plot(
                x, sub["mean"].to_numpy(),
                color=color, marker=marker, linestyle=ls,
                linewidth=2.0, markersize=5, alpha=0.9
            )

    ax_std_luv.set_title("Louvain", fontsize=12)
    ax_std_len.set_title("Length-cut", fontsize=12)

    ax_std_luv.set_ylabel("std")
    ax_std_len.set_ylabel("std")

    ax_std_luv.set_ylim(0, max_std)
    ax_std_len.set_ylim(0, max_std)

    ax_mean_luv.set_ylabel("ARI mean")
    ax_mean_len.set_ylabel("ARI mean")

    ax_mean_luv.set_xlabel(r"$\mu$")
    ax_mean_len.set_xlabel(r"$\mu$")

    ax_mean_luv.set_ylim(-0.02, 1.02)
    ax_mean_len.set_ylim(-0.02, 1.02)

    for ax in [ax_std_luv, ax_std_len, ax_mean_luv, ax_mean_len]:
        ax.grid(True, alpha=0.25)

    ax_mean_luv.legend(
        loc="upper right",
        fontsize=12,
        frameon=True,
        framealpha=0.85,
        borderpad=0.3,
        handlelength=1.3,
        labelspacing=0.3,
        ncol=1
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved:", out_path)

plot_onefig_mu_series_method_p(g, out_pdf)
