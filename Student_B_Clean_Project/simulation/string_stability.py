
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS / "results_with_mcth.csv")

t = df["time"].to_numpy()
errors = np.column_stack([
    df[f"e{i}"].to_numpy() for i in range(1, 7)
])

if len(t) < 2 or not np.all(np.diff(t) > 0):
    raise ValueError("Time must be strictly increasing.")

duration = t[-1] - t[0]

rmse = np.array([
    np.sqrt(np.trapezoid(errors[:, i]**2, t) / duration)
    for i in range(6)
])

peak = np.max(np.abs(errors), axis=0)

print("\nSTRING STABILITY DIAGNOSTICS")
print("Vehicle | RMSE (cm) | Peak error (m)")

for i in range(6):
    print(
        f"{i+1:^7} | "
        f"{rmse[i]*100:^9.4f} | "
        f"{peak[i]:^14.5f}"
    )

print("\nAdjacent vehicle RMSE ratios:")

ratios = rmse[1:] / rmse[:-1]

for i, ratio in enumerate(ratios, start=2):
    print(f"Vehicle {i} / Vehicle {i-1}: {ratio:.4f}")

print("\nInterpretation:")
if np.all(ratios <= 1):
    print("No adjacent RMSE amplification observed.")
else:
    print("At least one adjacent RMSE amplification observed.")

# Figure 1: RMSE across vehicles
fig, ax = plt.subplots(figsize=(9, 5))

vehicles = np.arange(1, 7)

ax.plot(
    vehicles, rmse * 100,
    "ko-", linewidth=2, markersize=7
)

ax.set_xticks(vehicles)
ax.set_xlabel("Follower Vehicle")
ax.set_ylabel("Spacing Error RMSE (cm)")
ax.set_title("Spacing Error Propagation Along the Platoon")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES / "string_stability_rmse.png", dpi=300)
plt.close(fig)

# Figure 2: Adjacent RMSE ratios
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    np.arange(2, 7),
    ratios,
    color="white",
    edgecolor="black",
    hatch="//"
)

ax.axhline(
    1.0, color="black",
    linestyle="--", linewidth=1.5,
    label="Unity ratio"
)

ax.set_xticks(np.arange(2, 7))
ax.set_xlabel("Follower Vehicle")
ax.set_ylabel("RMSE Ratio to Previous Vehicle")
ax.set_title("Adjacent Vehicle Spacing Error Ratios")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(FIGURES / "string_stability_ratios.png", dpi=300)
plt.close(fig)

print("\nSaved:")
print("string_stability_rmse.png")
print("string_stability_ratios.png")
