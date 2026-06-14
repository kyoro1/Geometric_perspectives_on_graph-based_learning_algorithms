import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.plot_style import METHOD_ORDER, COLOR, method_rank, style_for, normalize_method

# === input files ===
in_csv  = "./results/lfr_orc_src.csv"  
inf_csv = "./results/lfr_igraph.csv"   

out_pdf1 = "./results/result_lfr_src_orc_only.pdf"
out_pdf2 = "./results/result_lfr_louvain_plus_igraph.pdf"
out_pdf3 = "./results/result_lfr_src_mst_vs_spt.pdf"

DISPLAY_NAME = {
    "orc": "ORC",
    "src-mst": "SRC(MST)",
    "src-spt": "SRC(SPT)",
}

df = pd.read_csv(in_csv)
df["mu"] = df["mu"].astype(float)

score_cols = ["best_ari_louvain", "best_ari_length"]
for c in score_cols:
    df[c] = df[c].astype(float)

df_long = df.melt(
    id_vars=["method", "mu"],
    value_vars=score_cols,
    var_name="score_type",
    value_name="score"
)

df_ig = pd.read_csv(inf_csv)
df_ig["mu"] = df_ig["mu"].astype(float)
df_ig["ari"]   = df_ig["ari"].astype(float)

igraph_methods = ["infomap", "spinglass", "fast_greedy", "edgebetweenness", "label_propagation"]
df_ig = df_ig[df_ig["method"].isin(igraph_methods)].copy()

df_ig_long = df_ig.rename(columns={"ari": "score"})[["method", "mu", "score"]]
df_ig_long["score_type"] = "ari"

df_all = pd.concat([df_long, df_ig_long], ignore_index=True)

g = (df_all.groupby(["method", "score_type", "mu"])["score"]
       .agg(["mean", "std", "count"])
       .reset_index()
       .sort_values(["method", "score_type", "mu"]))

g["std"] = g["std"].fillna(0.0)


def plot_pdf_louvain_plus_igraph(g, out_path, igraph_methods):
    mask_louvain = (
        g["method"].isin(["src-mst", "src-spt", "orc"]) &
        (g["score_type"] == "best_ari_louvain")
    )

    mask_igraph = (
        g["method"].isin(igraph_methods) &
        (g["score_type"] == "ari")
    )

    g2 = g[mask_louvain | mask_igraph].copy()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(7.2, 6.0),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.10},
        constrained_layout=True
    )

    series_list = list(g2.groupby(["method", "score_type"]).groups.keys())
    series_list = sorted(series_list, key=lambda k: (method_rank(normalize_method(k[0])), k[1]))

    for i, (method, score_type) in enumerate(series_list):
        m_norm = normalize_method(method)

        sub = g2[(g2["method"] == method) & (g2["score_type"] == score_type)].sort_values("mu")
        x = sub["mu"].to_numpy()
        y_mean = sub["mean"].to_numpy()
        y_std  = sub["std"].to_numpy()

        disp = DISPLAY_NAME.get(m_norm, m_norm)
        label = f"{disp}"

        base_style = style_for(m_norm)
        style = dict(
            **base_style,
            linewidth=2.0,
            markersize=7,
        )

        ax_top.plot(x, y_std, label=label, alpha=0.75, **style)
        ax_bot.plot(x, y_mean, label=label, alpha=0.9,  **style)

    ax_top.set_ylabel("std")
    ax_top.grid(True)
    ax_top.set_title("LFR, 500 nodes", y=1.02)

    ax_bot.set_ylabel("ARI mean")
    ax_bot.set_xlabel(r"$\mu$")
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.grid(True)

    ax_bot.legend(
        loc="upper right",
        fontsize=13,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="0.8",
        ncol=1,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3
    )

    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print("saved:", out_path)


plot_pdf_louvain_plus_igraph(g, out_pdf2, igraph_methods)
