
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locate the project results directory
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Prefer the most recently processed spacing-error data
candidates = [
    RESULTS / "results_with_mcth.csv",
    RESULTS / "results.csv",
]

data_file = next((p for p in candidates if p.exists()), None)

if data_file is None:
    raise FileNotFoundError(
        f"No results CSV found in {RESULTS}"
    )

df = pd.read_csv(data_file)
print("Reading:", data_file.name)
print("Available columns:", list(df.columns))

# Find time column
time_col = next(
    (c for c in ["time", "t", "time_s"] if c in df.columns),
    None
)

if time_col is None:
    raise ValueError("Time column not found.")

# Detect the spacing-error columns
possible_sets = [
    [f"e{i}" for i in range(1, 7)],
    [f"e_{i}" for i in range(1, 7)],
    [f"spacing_error_{i}" for i in range(1, 7)],
]

error_cols = next(
    (cols for cols in possible_sets
     if all(c in df.columns for c in cols)),
    None
)

if error_cols is None:
    raise ValueError(
        "Cannot identify all six spacing-error columns. "
        "Check the printed column names."
    )

t = df[time_col].to_numpy(dtype=float)

if len(t) < 2 or not np.all(np.diff(t) > 0):
    raise ValueError("Time must be strictly increasing.")

duration = t[-1] - t[0]
if duration <= 0:
    raise ValueError("Invalid simulation duration.")

rows = []

for i, col in enumerate(error_cols, start=1):
    e = df[col].to_numpy(dtype=float)

    if not np.all(np.isfinite(e)):
        raise ValueError(f"Non-finite values in {col}")

    # Time-weighted RMSE
    integral = np.trapezoid(e**2, t)
    rmse_m = np.sqrt(integral / duration)

    rows.append({
        "Vehicle": i,
        "RMSE (m)": rmse_m,
        "RMSE (cm)": rmse_m * 100,
        "Peak absolute error (m)": np.max(np.abs(e))
    })

summary = pd.DataFrame(rows)

print("\nSPACING ERROR RESULTS")
print(summary.to_string(index=False))

output_csv = RESULTS / "rmse_summary.csv"
summary.to_csv(output_csv, index=False)

# Publication-style black and white figure
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(1, 7)
ax.bar(
    x,
    summary["RMSE (cm)"],
    color="white",
    edgecolor="black",
    linewidth=1.5,
    hatch="//"
)

ax.set_xticks(x)
ax.set_xticklabels([f"Vehicle {i}" for i in x])
ax.set_ylabel("RMSE (cm)")
ax.set_title("Spacing Error RMSE for Six Vehicles")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()

png = FIGURES / "rmse_six_vehicles.png"
pdf = FIGURES / "rmse_six_vehicles.pdf"

fig.savefig(png, dpi=300, facecolor="white")
fig.savefig(pdf, facecolor="white")
plt.close(fig)

print("\nSaved:", output_csv)
print("Saved:", png)
print("Saved:", pdf)
