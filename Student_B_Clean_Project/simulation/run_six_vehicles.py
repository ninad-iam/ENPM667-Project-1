
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 2: SIX-VEHICLE PLATOON DYNAMICS TEST
# Preliminary controller, NOT the paper's adaptive STISMC
# ============================================================

N = 6
DT = 0.001
T_FINAL = 20.0

D = 0.5        # Standstill spacing (m)
H = 1.0        # Time headway (s)

KP = 0.8       # Preliminary proportional gain
KD = 1.5       # Preliminary relative-velocity gain
A_LIMIT = 5.0  # Illustrative acceleration limit (m/s^2)

# Paper reference initial conditions
X_INITIAL = np.array(
    [16.0, 14.0, 12.0, 10.0, 7.0, 4.2, 2.4]
)

V_INITIAL = np.array(
    [2.0, 2.3, 1.7, 1.6, 2.8, 2.4, 2.3]
)

# Output directory: project/results
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_initial_condition_test"
OUT.mkdir(parents=True, exist_ok=True)


def leader_acceleration(t):
    if 2.0 <= t < 6.0:
        return 1.0
    return 0.0


def run_simulation():
    t = np.arange(0.0, T_FINAL + DT / 2, DT)
    steps = len(t)

    # Vehicle index 0 = leader; 1..6 = followers
    x = np.zeros((steps, N + 1))
    v = np.zeros((steps, N + 1))
    a = np.zeros((steps, N + 1))
    e = np.zeros((steps, N))

    x[0] = X_INITIAL
    v[0] = V_INITIAL

    for k in range(steps):
        # CTH spacing error for each follower
        e[k] = (
            x[k, :-1]
            - x[k, 1:]
            - D
            - H * v[k, 1:]
        )

        a[k, 0] = leader_acceleration(t[k])

        # Preliminary controller:
        # spacing error + relative velocity feedback
        relative_v = v[k, :-1] - v[k, 1:]

        a[k, 1:] = np.clip(
            KP * e[k] + KD * relative_v,
            -A_LIMIT,
            A_LIMIT
        )

        if k == steps - 1:
            break

        # Constant acceleration integration over one step
        x[k + 1] = (
            x[k]
            + v[k] * DT
            + 0.5 * a[k] * DT**2
        )

        v[k + 1] = v[k] + a[k] * DT

    # Save CSV compatible with graph generator
    data = {"time": t}

    for i in range(N + 1):
        data[f"x{i}"] = x[:, i]
        data[f"v{i}"] = v[:, i]
        data[f"a{i}"] = a[:, i]

    for i in range(1, N + 1):
        data[f"e{i}"] = e[:, i - 1]

    df = pd.DataFrame(data)
    csv_path = OUT / "results.csv"
    df.to_csv(csv_path, index=False)

    # Vehicle positions
    plt.figure(figsize=(10, 6))
    for i in range(N + 1):
        plt.plot(t, x[:, i], label=f"Vehicle {i}")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Platoon Positions: Preliminary Dynamics Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "vehicle_positions.png", dpi=300)
    plt.close()

    # Vehicle velocities
    plt.figure(figsize=(10, 6))
    for i in range(N + 1):
        plt.plot(t, v[:, i], label=f"Vehicle {i}")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Platoon Velocities: Preliminary Dynamics Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "vehicle_velocities.png", dpi=300)
    plt.close()

    # Spacing errors
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(t, e[:, i], label=f"Vehicle {i + 1}")

    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("CTH Spacing Error (m)")
    plt.title("Preliminary Spacing Error Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "spacing_errors.png", dpi=300)
    plt.close()

    # Numerical checks
    assert np.isfinite(x).all()
    assert np.isfinite(v).all()
    assert np.isfinite(e).all()

    print("SUCCESS: Six-vehicle dynamics simulation completed!")
    print(f"Leader final velocity: {v[-1, 0]:.3f} m/s")
    print(f"Results saved to: {csv_path}")
    print("Graphs saved in the results folder.")

    print("\nSpacing error RMSE (preliminary controller):")
    for i in range(N):
        rmse = np.sqrt(np.mean(e[:, i] ** 2))
        print(f"Vehicle {i + 1}: {rmse:.5f} m")


if __name__ == "__main__":
    run_simulation()
