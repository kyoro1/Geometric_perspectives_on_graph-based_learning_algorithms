import os
import numpy as np
import pandas as pd
from tools.plot_style import normalize_method

# === input files ===
in_csv  = "./results/lfr_orc_src_root.csv"  

out_md  = "./results/lfr_table.md"
out_tex = "./results/lfr_table.tex"


# === 表示名（論文用）===
DISPLAY_NAME = {
    "orc": "ORC",
    "src-mst": "SRC(MST)",
    "src-spt": "SRC(SPT)",
}


def make_openreview_table(df):
    df = df.copy()

    # --- 型 ---
    df["mu"] = df["mu"].astype(float)

    score_cols = ["best_ari_louvain", "best_ari_length"]
    for c in score_cols:
        df[c] = df[c].astype(float)

    # --- melt ---
    df_all = df.melt(
        id_vars=["method", "mu"],
        value_vars=score_cols,
        var_name="score_type",
        value_name="score"
    )

    # --- 集計 ---
    g = (df_all.groupby(["method", "score_type", "mu"])["score"]
            .agg(["mean", "std"])
            .reset_index())

    g["std"] = g["std"].fillna(0.0)

    # --- Louvainだけ ---
    mask = (
        g["method"].isin(["src-mst", "src-spt", "orc"]) &
        (g["score_type"] == "best_ari_louvain")
    )
    g_plot = g[mask].copy()

    # --- 表用 ---
    g_plot["method_norm"] = g_plot["method"].apply(normalize_method)
    g_plot["display"] = g_plot["method_norm"].map(DISPLAY_NAME)

    g_plot["mean_std"] = g_plot.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1
    )

    table = g_plot.pivot(
        index="mu",
        columns="display",
        values="mean_std"
    ).sort_index()

    return table

# === optional: best強調 ===
def highlight_best(table):
    table = table.copy()

    for idx in table.index:
        row = table.loc[idx]

        means = row.apply(lambda x: float(x.split("±")[0]))
        best_col = means.idxmax()

        table.loc[idx, best_col] = f"**{table.loc[idx, best_col]}**"

    return table


# === main ===
if __name__ == "__main__":
    df = pd.read_csv(in_csv)

    table = make_openreview_table(df)

    # bestを太字にする場合
    table_highlight = highlight_best(table)

    # === Markdown（OpenReview用）===
    md = table_highlight.to_markdown()
    with open(out_md, "w") as f:
        f.write(md)

    print("=== Markdown Table ===")
    print(md)

    # === LaTeX（論文用）===
    tex = table.to_latex(escape=False)
    with open(out_tex, "w") as f:
        f.write(tex)

    print("\nSaved:")
    print(out_md)
    print(out_tex)