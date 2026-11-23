
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 4: LOCAL AND COUPLED INTEGRAL SLIDING SURFACES
#
# Diagnostic calculation from existing MCTH trajectories.
# This is NOT yet a closed-loop STISMC simulation.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RESULTS / "results_with_mcth.csv"
OUTPUT_CSV = RESULTS / "results_with_surfaces.csv"

N = 6

# Reference-paper design parameters
K_I = 10.0
Q = 0.9

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Cannot find: {INPUT_CSV}\n"
        "Run mcth_comparison.py first."
    )

df = pd.read_csv(INPUT_CSV)

t = df["time"].to_numpy(dtype=float)

ebar = np.column_stack([
    df[f"ebar{i}"].to_numpy(dtype=float)
    for i in range(1, N + 1)
])

if len(t) < 2:
    raise ValueError("At least two time samples are required.")

dt = np.diff(t)

if not np.all(dt > 0):
    raise ValueError("Time must be strictly increasing.")

if not np.isfinite(ebar).all():
    raise ValueError("Modified spacing errors contain invalid values.")

# ============================================================
# 1. NUMERICAL INTEGRAL OF MODIFIED SPACING ERROR
#
# I_i(t) = integral from 0 to t of ebar_i(tau) d(tau)
#
# Trapezoidal integration
# ============================================================

integral_ebar = np.zeros_like(ebar)

increments = (
    0.5
    * (ebar[:-1, :] + ebar[1:, :])
    * dt[:, None]
)

integral_ebar[1:, :] = np.cumsum(
    increments,
    axis=0
)

# ============================================================
# 2. LOCAL INTEGRAL SLIDING SURFACES
#
# s_i = ebar_i + K_I * integral(ebar_i)
# ============================================================

s = ebar + K_I * integral_ebar

# ============================================================
# 3. BIDIRECTIONALLY COUPLED SLIDING SURFACES
#
# S_i = Q*s_i - s_(i+1), for i = 1,...,5
#
# Correct tail boundary:
# S_6 = Q*s_6
# ============================================================

S = np.zeros_like(s)

S[:, :-1] = (
    Q * s[:, :-1]
    - s[:, 1:]
)

S[:, -1] = Q * s[:, -1]

# ============================================================
# 4. SAVE ALL RESULTS
# ============================================================

output = df.copy()

for i in range(N):
    output[f"int_ebar{i + 1}"] = integral_ebar[:, i]
    output[f"s{i + 1}"] = s[:, i]
    output[f"S{i + 1}"] = S[:, i]

output.to_csv(OUTPUT_CSV, index=False)

# ============================================================
# 5. GRAPH: LOCAL SLIDING SURFACES
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

styles = [
    "-", "--", "-.", ":",
    (0, (5, 2)), (0, (3, 1, 1, 1))
]

for i in range(N):
    ax.plot(
        t,
        s[:, i],
        color="black",
        linestyle=styles[i],
        linewidth=1.4,
        label=f"Vehicle {i + 1}"
    )

ax.axhline(
    0,
    color="gray",
    linewidth=0.8
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Local Sliding Surface")
ax.set_title(
    "Local Integral Sliding Surfaces: Diagnostic Calculation"
)
ax.legend()
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(
    RESULTS / "04_local_sliding_surfaces.png",
    dpi=300,
    facecolor="white"
)
plt.close(fig)

# ============================================================
# 6. GRAPH: COUPLED SLIDING SURFACES
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(N):
    ax.plot(
        t,
        S[:, i],
        color="black",
        linestyle=styles[i],
        linewidth=1.4,
        label=f"Vehicle {i + 1}"
    )

ax.axhline(
    0,
    color="gray",
    linewidth=0.8
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Coupled Sliding Surface")
ax.set_title(
    "Bidirectionally Coupled Surfaces: Diagnostic Calculation"
)
ax.legend()
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(
    RESULTS / "05_coupled_sliding_surfaces.png",
    dpi=300,
    facecolor="white"
)
plt.close(fig)

# ============================================================
# 7. MATHEMATICAL VALIDATION
# ============================================================

# All modified errors begin at zero.
assert np.allclose(ebar[0, :], 0.0, atol=1e-9)

# All integral states begin at zero.
assert np.allclose(integral_ebar[0, :], 0.0)

# All local surfaces begin at zero.
assert np.allclose(s[0, :], 0.0, atol=1e-9)

# All coupled surfaces begin at zero.
assert np.allclose(S[0, :], 0.0, atol=1e-9)

# Verify the correct tail boundary.
assert np.allclose(S[:, -1], Q * s[:, -1])

# Verify the coupled surface equation.
assert np.allclose(
    S[:, :-1],
    Q * s[:, :-1] - s[:, 1:]
)

print("\nSUCCESS: Sliding surface calculations completed!")
print(f"Integral gain K_I = {K_I}")
print(f"Coupling parameter Q = {Q}")

print("\nInitial local surfaces:")
print(np.round(s[0, :], 10))

print("\nInitial coupled surfaces:")
print(np.round(S[0, :], 10))

print("\nFinal local surfaces:")
print(np.round(s[-1, :], 6))

print("\nFinal coupled surfaces:")
print(np.round(S[-1, :], 6))

print("\nFiles generated:")
print("04_local_sliding_surfaces.png")
print("05_coupled_sliding_surfaces.png")
print("results_with_surfaces.csv")
