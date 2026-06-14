import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DISPLAY_NAME = {
    "orc": "ORC",
    "src-mst": "SRC(MST)",
    "src-spt": "SRC(SPT)",
}

csv_path = "./results/sbm_src_orc.csv" 
df = pd.read_csv(csv_path)

agg = (
    df.groupby(["ratio", "method"])["time_flow_sec"]
      .agg(["mean", "std", "count"])
      .reset_index()
)

agg["sem"] = agg["std"] / np.sqrt(agg["count"])

ratios = sorted(df["ratio"].unique())
methods = sorted(df["method"].unique())

x = np.arange(len(ratios))

n_methods = len(methods)
bar_width = min(0.8 / n_methods, 0.25)

fig, ax = plt.subplots(figsize=(10, 5))

for i, method in enumerate(methods):
    sub = agg[agg["method"] == method].set_index("ratio").reindex(ratios)

    means = sub["mean"].values
    errs  = sub["std"].values

    offset = (i - n_methods/2) * bar_width + bar_width/2

    label = DISPLAY_NAME.get(method, method)

    ax.bar(
        x + offset,
        means,
        width=bar_width,
        yerr=errs,
        capsize=3,
        label=label,
        alpha=0.9
    )

ax.set_xlabel("ratio")
ax.set_ylabel("Process Time (sec)")
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in ratios])
ax.legend(loc="upper left", frameon=True, )
ax.yaxis.grid(True, linestyle="--", linewidth=1)
ax.set_yscale("log")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("./results/summary_sbm_process_time.pdf", dpi=300)