
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "ftre_results.csv")

t = df["time"].to_numpy()
mask = t >= 1.0

styles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]

def make_plot(prefix, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(1, 7):
        ax.plot(
            t[mask],
            df[f"{prefix}{i}"].to_numpy()[mask],
            color="black",
            linestyle=styles[i - 1],
            linewidth=1.2,
            label=f"Vehicle {i}"
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        RESULTS / filename,
        dpi=300,
        facecolor="white"
    )
    plt.close(fig)

make_plot(
    "a_error",
    "Acceleration Estimation Errors (After Initialization)",
    "Estimation error (m/s²)",
    "14_ftre_acceleration_errors_zoom.png"
)

make_plot(
    "dv_error",
    "Mismatched Disturbance Estimation Errors (After Initialization)",
    "Estimation error (m/s)",
    "15_ftre_disturbance_errors_zoom.png"
)

print("\nEstimator RMSE, evaluated from 1 to 20 seconds")
print("Vehicle | Acceleration (m/s²) | Disturbance (m/s)")

for i in range(1, 7):
    ae = df[f"a_error{i}"].to_numpy()[mask]
    de = df[f"dv_error{i}"].to_numpy()[mask]

    a_rmse = np.sqrt(np.mean(ae ** 2))
    d_rmse = np.sqrt(np.mean(de ** 2))

    print(f"{i:7d} | {a_rmse:19.6f} | {d_rmse:17.6f}")

print("\nSaved both zoomed graphs to:", RESULTS)


fig, ax = plt.subplots(figsize=(10, 5))

vehicle = 2
column = f"a_error{vehicle}"

ax.plot(
    t,
    df[column],
    color="black",
    linewidth=0.8
)

ax.set_xlim(2, 4)
ax.set_title(
    f"Vehicle {vehicle}: Acceleration Estimation Error"
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Estimation error (m/s²)")
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(
    RESULTS / "16_ftre_vehicle2_error_detail.png",
    dpi=300,
    facecolor="white"
)
plt.close(fig)