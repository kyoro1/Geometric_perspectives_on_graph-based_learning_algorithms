import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/discrete_mm_momentum_sweep/costs_discrete_mm_momentum_sweep.csv")

plt.figure(figsize=(8, 5))
for m, g in df.groupby("ee_momentum"):
    g = g.sort_values("iteration")
    plt.plot(g["iteration"], g["cost"], label=f"momentum={m}")

plt.xlabel("Iteration", fontsize=14)
plt.ylabel("KL divergence", fontsize=14)
plt.title("KL divergence trajectory by EE momentum", fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("outputs/discrete_mm_momentum_sweep/kl_trajectory.pdf", dpi=200)
print("saved: outputs/discrete_mm_momentum_sweep/kl_trajectory.pdf")
