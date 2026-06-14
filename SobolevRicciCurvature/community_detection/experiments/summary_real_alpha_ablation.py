import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.plot_style import METHOD_ORDER, COLOR, normalize_method

DISPLAY_NAME = {
    "src-mst": "SRC(MST)",
    "src-spt": "SRC(SPT)",
    "orc": "ORC",
}

df = pd.read_csv("./results/real_alpha_ablation.csv")

# ===== load =====
df["alpha"] = df["alpha"].astype(float)
df["best_ari_louvain"] = df["best_ari_louvain"].astype(float)

METHOD_COLOR_OVERRIDES = {
    "src-mst": "#ff7f0e",  # tab10 orange
    "src-spt": "#2ca02c",  # tab10 green
}

for name, df_name in df.groupby("name"):
    # ===== aggregate (method x alpha) =====
    g = (
        df_name.groupby(["method", "alpha"])["best_ari_louvain"]
              .agg(["mean", "std", "count"])
              .reset_index()
    )
    g["std"] = g["std"].fillna(0.0)

    # ===== plotting =====
    methods = sorted(g["method"].unique())
    alphas  = sorted(g["alpha"].unique())

    x = np.arange(len(alphas))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for i, method in enumerate(methods):
        m_norm = normalize_method(method)
        sub = g[g["method"] == method].sort_values("alpha")

        color = METHOD_COLOR_OVERRIDES.get(m_norm, None)
        if color is None:
            color = COLOR.get(m_norm, None)

        label = DISPLAY_NAME.get(m_norm, method)

        ax.bar(
            x + i * width,
            sub["mean"],
            width=width,
            yerr=sub["std"],
            capsize=4,
            label=label,
            alpha=0.9,
            color=color,
            edgecolor="none",
        )

    # ===== axis & style =====
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("ARI (mean ± std)")
    ax.set_title(f"{name}")
    ax.set_ylim(0.0, 1.0)

    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)

    plt.tight_layout()

    out_path = f"./results/summary_real_alpha_ablation_{name}.pdf"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("saved:", out_path)
