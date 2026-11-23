
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STAGE 6: FTRE COMPONENT VALIDATION
#
# Test A: Acceleration estimation from velocity
# Test B: Known mismatched-disturbance estimation
#
# These are separate estimator tests, not the complete
# closed-loop vehicle/FTRE simulation.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS / "results.csv"

N = 6

# Test gains, not a verified reproduction of paper gains
LAMBDA_0 = 1.5
LAMBDA_1 = 1.1
ACC_BOUND = 20.0
DIST_BOUND = 5.0

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Missing {CSV_FILE}\n"
        "Run run_six_vehicles.py first."
    )

df = pd.read_csv(CSV_FILE)

print("CSV columns:")
print(df.columns.tolist())

# Accept common time-column names
time_col = next(
    (c for c in ["time", "t", "Time"] if c in df.columns),
    None
)

if time_col is None:
    raise ValueError("Cannot find time column.")

t = df[time_col].to_numpy(dtype=float)

if len(t) < 3 or np.any(np.diff(t) <= 0):
    raise ValueError("Time must contain increasing samples.")

# Detect follower velocity columns
velocity_names = []

for i in range(1, N + 1):
    candidates = [
        f"v{i}", f"v_{i}",
        f"velocity{i}", f"velocity_{i}"
    ]

    found = next(
        (name for name in candidates if name in df.columns),
        None
    )

    if found is None:
        raise ValueError(
            f"Cannot find velocity for vehicle {i}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    velocity_names.append(found)

velocity = df[velocity_names].to_numpy(dtype=float)

if not np.isfinite(velocity).all():
    raise ValueError("Velocity data contains NaN or infinity.")

# ============================================================
# SUPER-TWISTING DIFFERENTIATOR
#
# z0 estimates measured signal y
# z1 estimates derivative of y
#
# z0_dot = z1 - lambda0*sqrt(L)*sqrt(abs(z0-y))*sign(z0-y)
# z1_dot = -lambda1*L*sign(z0-y)
# ============================================================

def differentiate(time, measured, bound):
    measured = np.asarray(measured, dtype=float)

    if measured.ndim == 1:
        measured = measured[:, None]

    samples, channels = measured.shape

    z0 = np.zeros((samples, channels))
    z1 = np.zeros((samples, channels))

    z0[0] = measured[0]

    for k in range(samples - 1):
        dt = time[k + 1] - time[k]

        error = z0[k] - measured[k]

        EPSILON = 0.001
        sign_error = error / (
            np.abs(error) + EPSILON
        )
        

        z0_dot = (
            z1[k]
            - LAMBDA_0 * np.sqrt(bound)
            * np.sqrt(np.abs(error)) * sign_error
        )

        z1_dot = (
            -LAMBDA_1 * bound * sign_error
        )

        z0[k + 1] = z0[k] + dt * z0_dot
        z1[k + 1] = z1[k] + dt * z1_dot

    return z0, z1

# ============================================================
# TEST A: FOLLOWER ACCELERATION ESTIMATION
#
# Reference = numerical derivative of simulated velocity.
# This is not an independently measured ground truth.
# ============================================================

_, acceleration_hat = differentiate(
    t, velocity, ACC_BOUND
)

acceleration_ref = np.gradient(
    velocity, t, axis=0
)

acceleration_error = (
    acceleration_hat - acceleration_ref
)

# ============================================================
# TEST B: KNOWN MISMATCHED DISTURBANCE
#
# Artificial test plant:
# x_dot = v + delta_v
#
# Therefore derivative(x) - v = delta_v.
#
# Use synthetic signals so ground truth is known.
# ============================================================

phase = np.arange(N) * 0.35

delta_true = (
    0.08 * np.sin(0.8 * t[:, None] + phase[None, :])
    + 0.03 * np.sin(1.6 * t[:, None] + phase[None, :])
)

# Synthetic measured position for the test
x_test = np.zeros_like(velocity)

for k in range(len(t) - 1):
    dt = t[k + 1] - t[k]

    xdot_now = velocity[k] + delta_true[k]
    xdot_next = velocity[k + 1] + delta_true[k + 1]

    x_test[k + 1] = (
        x_test[k] + 0.5 * dt * (xdot_now + xdot_next)
    )

_, xdot_hat = differentiate(
    t, x_test, DIST_BOUND
)

delta_hat = xdot_hat - velocity

delta_error = delta_hat - delta_true

# ============================================================
# EXPORT CSV
# ============================================================

output = {"time": t}

for i in range(N):
    j = i + 1

    output[f"a_ref{j}"] = acceleration_ref[:, i]
    output[f"ahat{j}"] = acceleration_hat[:, i]
    output[f"a_error{j}"] = acceleration_error[:, i]

    output[f"dv_ref{j}"] = delta_true[:, i]
    output[f"dvhat{j}"] = delta_hat[:, i]
    output[f"dv_error{j}"] = delta_error[:, i]

pd.DataFrame(output).to_csv(
    RESULTS / "ftre_results.csv",
    index=False
)

# ============================================================
# BLACK AND WHITE GRAPH FUNCTION
# ============================================================

styles = [
    "-", "--", "-.", ":",
    (0, (5, 2)), (0, (3, 1, 1, 1))
]

def plot_six(values, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(N):
        ax.plot(
            t, values[:, i],
            color="black",
            linestyle=styles[i],
            linewidth=1.2,
            label=f"Vehicle {i+1}"
        )

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

# Acceleration estimate comparison for one vehicle
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    t, acceleration_ref[:, 1],
    color="black", linewidth=1.5,
    label="Numerical reference"
)

ax.plot(
    t, acceleration_hat[:, 1],
    color="black", linestyle="--",
    linewidth=1.3,
    label="FTRE estimate"
)

ax.set_title("Vehicle 2: Acceleration Estimation Test")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (m/s²)")
ax.grid(True, alpha=0.2)
ax.legend()

fig.tight_layout()
fig.savefig(
    RESULTS / "10_ftre_acceleration_estimates.png",
    dpi=300,
    facecolor="white"
)
plt.close(fig)

plot_six(
    acceleration_error,
    "Acceleration Estimation Errors",
    "Estimation error (m/s²)",
    "11_ftre_acceleration_errors.png"
)

# Disturbance estimate comparison for one vehicle
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    t, delta_true[:, 0],
    color="black", linewidth=1.5,
    label="Known injected disturbance"
)

ax.plot(
    t, delta_hat[:, 0],
    color="black", linestyle="--",
    linewidth=1.3,
    label="Estimated disturbance"
)

ax.set_title(
    "Vehicle 1: Synthetic Mismatched-Disturbance Test"
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Disturbance (m/s)")
ax.grid(True, alpha=0.2)
ax.legend()

fig.tight_layout()
fig.savefig(
    RESULTS / "12_ftre_disturbance_estimates.png",
    dpi=300,
    facecolor="white"
)
plt.close(fig)

plot_six(
    delta_error,
    "Synthetic Disturbance Estimation Errors",
    "Estimation error (m/s)",
    "13_ftre_disturbance_errors.png"
)

# ============================================================
# RMSE OF ESTIMATOR TESTS
# ============================================================

# Ignore the first second for steady-state diagnostics
mask = t >= (t[0] + 1.0)

print("\nFTRE COMPONENT TEST COMPLETED")
print("\nAcceleration estimation RMSE (m/s²):")

for i in range(N):
    rmse = np.sqrt(
        np.mean(acceleration_error[mask, i] ** 2)
    )
    print(f"Vehicle {i+1}: {rmse:.6f}")

print("\nSynthetic disturbance RMSE (m/s):")

for i in range(N):
    rmse = np.sqrt(
        np.mean(delta_error[mask, i] ** 2)
    )
    print(f"Vehicle {i+1}: {rmse:.6f}")

print("\nGenerated:")
print("10_ftre_acceleration_estimates.png")
print("11_ftre_acceleration_errors.png")
print("12_ftre_disturbance_estimates.png")
print("13_ftre_disturbance_errors.png")
print("ftre_results.csv")
print("\nIMPORTANT: These are separate estimator tests.")
print("They are not full closed-loop paper reproductions.")
