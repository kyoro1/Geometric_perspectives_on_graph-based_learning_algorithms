# tools/plot_sbm_lfr_root_ratio_mean_std.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sbm_csv = "./results/sbm_src_root_diff.csv"
lfr_csv = "./results/lfr_src_root_diff.csv"
out_pdf = "./results/result_sbm_lfr_root_ratio_mean_std.pdf"


def load_and_aggregate(csv_path, x_col, method_key="src-spt"):
    df = pd.read_csv(csv_path)

    # --- filter: only SRC(SPT) ---
    df = df[df["method"].astype(str).str.lower().str.contains(method_key)].copy()

    df[x_col] = df[x_col].astype(float)
    df["kappa_over_tree_mean"] = df["kappa_over_tree_mean"].astype(float)

    # --- aggregate across trials ---
    g = (df.groupby(x_col)["kappa_over_tree_mean"]
           .agg(["mean", "std", "count"])
           .reset_index()
           .sort_values(x_col))

    g["std"] = g["std"].fillna(0.0)
    return g


g_sbm = load_and_aggregate(sbm_csv, x_col="ratio")
g_lfr = load_and_aggregate(lfr_csv, x_col="mu")

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 3.8),
    sharey=True,
    constrained_layout=True
)

# --- SBM (left) ---
ax = axes[0]
x = g_sbm["ratio"].to_numpy()
y = g_sbm["mean"].to_numpy()
yerr = g_sbm["std"].to_numpy()

ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4)
ax.set_title(r"SBM")
ax.set_xlabel("ratio")
ax.set_ylabel(r"$\|\Delta \kappa\|_1 / |\Delta(T)|$")
ax.grid(True, alpha=0.3)

# --- LFR (right) ---
ax = axes[1]
x = g_lfr["mu"].to_numpy()
y = g_lfr["mean"].to_numpy()
yerr = g_lfr["std"].to_numpy()

ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4)
ax.set_title(r"LFR")
ax.set_xlabel(r"$\mu$")
ax.grid(True, alpha=0.3)

# --- unified y-range (optional but recommended) ---
y_all = np.concatenate([g_sbm["mean"].to_numpy(), g_lfr["mean"].to_numpy()])
err_all = np.concatenate([g_sbm["std"].to_numpy(), g_lfr["std"].to_numpy()])
ymin = max(0.0, float(y_all.min() - 2.0 * err_all.max()))
ymax = float(y_all.max() + 2.0 * err_all.max())
axes[0].set_ylim(ymin, ymax)

plt.savefig(out_pdf, dpi=300)
plt.close(fig)
print("saved:", out_pdf)
