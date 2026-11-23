
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 3: MODIFIED CONSTANT TIME HEADWAY (MCTH)
# Post-processing of preliminary simulation data
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CSV_FILE = RESULTS / "results.csv"

N = 6
H = 1.0
ZETA = 1.0

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {CSV_FILE}\n"
        "Run run_six_vehicles.py first."
    )

df = pd.read_csv(CSV_FILE)

t = df["time"].to_numpy()

if len(t) < 2:
    raise ValueError("CSV must contain at least two time samples.")

if not np.all(np.diff(t) > 0):
    raise ValueError("Time values must increase strictly.")

# Load conventional CTH spacing errors
e = np.column_stack([
    df[f"e{i}"].to_numpy()
    for i in range(1, N + 1)
])

# Load velocities and accelerations
v = np.column_stack([
    df[f"v{i}"].to_numpy()
    for i in range(N + 1)
])

a = np.column_stack([
    df[f"a{i}"].to_numpy()
    for i in range(N + 1)
])

# Initial spacing error
e0 = e[0, :]

# For e_i = x_(i-1) - x_i - d - h*v_i:
#
# e_dot_i = v_(i-1) - v_i - h*a_i
#
# This assumes x_dot = v, as in our preliminary model.
e_dot0 = (
    v[0, :-1]
    - v[0, 1:]
    - H * a[0, 1:]
)

# psi_i(t) = [e_i(0) + kappa_i*t] exp(-zeta*t)
#
# psi_dot_i(0) = kappa_i - zeta*e_i(0)
#
# Set psi_dot_i(0) = e_dot_i(0).
kappa = e_dot0 + ZETA * e0

exp_term = np.exp(-ZETA * t[:, None])

psi = (
    e0[None, :]
    + kappa[None, :] * t[:, None]
) * exp_term

# Modified spacing error
ebar = e - psi

# Exact derivative of correction function
psi_dot = (
    kappa[None, :]
    - ZETA * (
        e0[None, :]
        + kappa[None, :] * t[:, None]
    )
) * exp_term

# Conventional error derivative from stored dynamics
e_dot = (
    v[:, :-1]
    - v[:, 1:]
    - H * a[:, 1:]
)

ebar_dot = e_dot - psi_dot

# Export combined results
output = df.copy()

for i in range(N):
    output[f"ebar{i + 1}"] = ebar[:, i]
    output[f"psi{i + 1}"] = psi[:, i]

output.to_csv(
    RESULTS / "results_with_mcth.csv",
    index=False
)

# ============================================================
# GRAPH 1: CONVENTIONAL CTH ERRORS
# ============================================================

plt.figure(figsize=(10, 6))

for i in range(N):
    plt.plot(
        t, e[:, i],
        label=f"Vehicle {i + 1}"
    )

plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Spacing Error (m)")
plt.title("Conventional CTH Spacing Errors")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS / "01_cth_errors.png",
    dpi=300
)
plt.close()

# ============================================================
# GRAPH 2: MODIFIED MCTH ERRORS
# ============================================================

plt.figure(figsize=(10, 6))

for i in range(N):
    plt.plot(
        t, ebar[:, i],
        label=f"Vehicle {i + 1}"
    )

plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Modified Spacing Error (m)")
plt.title("MCTH Error: Post-Processed Trajectories")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS / "02_mcth_errors.png",
    dpi=300
)
plt.close()

# ============================================================
# GRAPH 3: CTH VS MCTH FOR VEHICLE 1
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    t, e[:, 0],
    color="black",
    linewidth=1.8,
    label="Conventional CTH error"
)

plt.plot(
    t, ebar[:, 0],
    color="black",
    linestyle="--",
    linewidth=1.8,
    label="Modified MCTH error"
)

plt.axhline(0, color="gray", linestyle=":")
plt.xlabel("Time (s)")
plt.ylabel("Spacing Error (m)")
plt.title("Vehicle 1: CTH vs MCTH Error Definition")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS / "03_cth_vs_mcth_vehicle1.png",
    dpi=300
)
plt.close()

# ============================================================
# VALIDATION
# ============================================================

print("\nMCTH calculation completed successfully!")

print("\nInitial modified spacing errors:")
for i in range(N):
    print(
        f"Vehicle {i + 1}: "
        f"{ebar[0, i]:.10f} m"
    )

print("\nInitial modified error derivatives:")
for i in range(N):
    print(
        f"Vehicle {i + 1}: "
        f"{ebar_dot[0, i]:.10f} m/s"
    )

assert np.allclose(ebar[0], 0.0, atol=1e-10)
assert np.allclose(ebar_dot[0], 0.0, atol=1e-10)

print("\nSUCCESS: Both MCTH initial conditions verified.")
print("Three graphs saved to the results folder.")
print("CSV saved as results_with_mcth.csv")
