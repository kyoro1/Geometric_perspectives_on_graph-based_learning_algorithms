import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.plot_style import METHOD_ORDER, COLOR, method_rank, style_for, normalize_method

DISPLAY_NAME = {
    "orc": "ORC",
    "src-mst": "SRC(MST)",
    "src-spt": "SRC(SPT)",
}

df_src = pd.read_csv("./results/real_src_orc.csv")
df_ig  = pd.read_csv("./results/real_igraph.csv")

src_louvain = df_src[["name", "method", "best_ari_louvain"]].copy()
src_louvain = src_louvain.rename(columns={"best_ari_louvain": "ari"})
src_louvain["method_plot"] = src_louvain["method"] + " (louvain)"
df_src_plot = src_louvain[["name", "method_plot", "ari"]]

df_ig_plot = df_ig[["name", "method", "ari"]].copy()
df_ig_plot["method"] = df_ig_plot["method"].replace({"fastgreedy": "fast_greedy"})  # ←追加
df_ig_plot["method_plot"] = df_ig_plot["method"]
df_ig_plot = df_ig_plot[["name", "method_plot", "ari"]]

df_all = pd.concat([df_src_plot, df_ig_plot], ignore_index=True)

## Aggregation
stats = (
    df_all
    .groupby(["name", "method_plot"])["ari"]
    .agg(["mean", "std"])
    .reset_index()
)

pivot_mean = stats.pivot(index="name", columns="method_plot", values="mean")
pivot_std  = stats.pivot(index="name", columns="method_plot", values="std")

methods_order = [
    "orc (louvain)",
    "src-mst (louvain)",
    "src-spt (louvain)",
    "infomap",
    "spinglass",
    "fast_greedy",
    "edgebetweenness",
    "label_propagation",
]
methods_order = [m for m in methods_order if m in pivot_mean.columns]

name_order = ["karate", "football", "polbooks", "polblogs", "email-eu-core"]

pivot_mean = pivot_mean.reindex(index=name_order, columns=methods_order)
pivot_std  = pivot_std.reindex(index=name_order, columns=methods_order)

x = np.arange(len(pivot_mean.index))
n_methods = len(pivot_mean.columns)
bar_width = min(0.8 / n_methods, 0.12)

fig, ax = plt.subplots(figsize=(max(10, n_methods*1.1), 4.8))

for i, method_plot in enumerate(pivot_mean.columns):
    y = pivot_mean[method_plot].values
    e = pivot_std[method_plot].values

    # method_plot -> base method (color key)
    base = method_plot.replace(" (louvain)", "")
    base = normalize_method(base)

    disp = DISPLAY_NAME.get(base, base)
    label = disp

    ax.bar(
        x + (i - n_methods/2)*bar_width + bar_width/2,
        y,
        width=bar_width,
        label=label,
        color=COLOR.get(base, "black"),
        yerr=e,
        capsize=3,
        ecolor="black",
        linewidth=0.8
    )


# =========================
# 7) Styling
# =========================
ax.set_ylabel("ARI")
ax.set_ylim(0, 1.05)

ax.yaxis.grid(True, linestyle="--", linewidth=1)
ax.set_axisbelow(True)

ax.set_xticks(x)
ax.set_xticklabels(pivot_mean.index, rotation=0)

ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig("./results/summary_real_with_errorbar.pdf", dpi=300)
