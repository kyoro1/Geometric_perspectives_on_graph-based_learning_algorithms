# tools/plot_lfr_beta_ari_and_res_2panel.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

in_csv  = "./results/lfr_src_beta.csv"
out_pdf = "./results/result_lfr_beta_ari_and_res_2panel.pdf"

df = pd.read_csv(in_csv)

# --- filter: only SRC(SPT) ---
df = df[df["method"].astype(str).str.lower().str.contains("src-spt")].copy()

# types
df["mu"] = df["mu"].astype(float)
df["beta"] = df["beta"].astype(float)
df["best_ari_louvain"] = df["best_ari_louvain"].astype(float)
df["best_res_louvain"] = df["best_res_louvain"].astype(float)

# --- aggregate: (mu, beta) -> mean/std ---
g_ari = (df.groupby(["mu", "beta"])["best_ari_louvain"]
           .agg(["mean", "std", "count"])
           .reset_index()
           .sort_values(["beta", "mu"]))
g_ari["std"] = g_ari["std"].fillna(0.0)

g_res = (df.groupby(["mu", "beta"])["best_res_louvain"]
           .agg(["mean", "std", "count"])
           .reset_index()
           .sort_values(["beta", "mu"]))
g_res["std"] = g_res["std"].fillna(0.0)

betas = sorted(df["beta"].unique())

# =========================
# plot
# =========================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)

ax1, ax2 = axes

# ---- Left: ARI ----
for b in betas:
    sub = g_ari[g_ari["beta"] == b]
    x = sub["mu"].to_numpy()
    y = sub["mean"].to_numpy()
    yerr = sub["std"].to_numpy()
    ax1.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4, label=rf"$\beta={b:g}$")

ax1.set_title("LFR: ARI (mean ± std)")
ax1.set_xlabel(r"LFR mixing parameter $\mu$")
ax1.set_ylabel("ARI")
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)

# ---- Right: best resolution ----
for b in betas:
    sub = g_res[g_res["beta"] == b]
    x = sub["mu"].to_numpy()
    y = sub["mean"].to_numpy()
    yerr = sub["std"].to_numpy()
    ax2.errorbar(x, y, yerr=yerr, marker="s", linewidth=2.0, capsize=4, label=rf"$\beta={b:g}$")

ax2.set_title("LFR: Best Louvain resolution (mean ± std)")
ax2.set_xlabel(r"LFR mixing parameter $\mu$")
ax2.set_ylabel("best resolution")
ax2.grid(True, alpha=0.3)

# y-limits (auto with small padding)
ymin = float(g_res["mean"].min() - 2.0 * g_res["std"].max())
ymax = float(g_res["mean"].max() + 2.0 * g_res["std"].max())
ax2.set_ylim(ymin, ymax)

# fig.suptitle(r"$\beta$ ablation (SRC-SPT + Louvain): performance vs selected resolution", y=1.02)
plt.savefig(out_pdf, dpi=300)
plt.close(fig)
print("saved:", out_pdf)
