
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# ADAPTIVE VS FIXED-GAIN SUPER-TWISTING COMPARISON
# Reduced-order sliding-surface dynamics only
# NOT the full six-vehicle platoon plant
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

N = 6
DT = 0.001
T = 20.0

KB2 = 2.0
KB1 = 2.0 * np.sqrt(2.0 * KB2)

OMEGA_INITIAL = 2.0
OMEGA_FIXED = 2.0
OMEGA_MIN = 0.2
OMEGA_MAX = 30.0

RHO_INITIAL = 0.5
RHO_MAX = 20.0

OMEGA_FACTOR = 1.5
EPS_B = 0.1
GAMMA = 0.5
TAU = 0.05

S_INITIAL = np.array([
    -0.8, -0.2, -0.1, -0.3, -0.1, -1.0
])

t = np.arange(0, T + DT / 2, DT)
steps = len(t)


def disturbance_function(time):
    return 0.12 * np.sin(
        0.8 * time + np.arange(N) * 0.3
    )


def simulate(adaptive=True):
    S = np.zeros((steps, N))
    theta = np.zeros((steps, N))
    Omega = np.zeros((steps, N))
    rho = np.zeros((steps, N))
    nu_filter = np.zeros((steps, N))
    control = np.zeros((steps, N))

    S[0] = S_INITIAL
    Omega[0] = OMEGA_INITIAL if adaptive else OMEGA_FIXED
    rho[0] = RHO_INITIAL

    chi0 = Omega[0] - EPS_B

    for k in range(steps):
        current_s = S[k]
        current_omega = Omega[k]

        sign_s = np.sign(current_s)

        k1 = np.sqrt(current_omega) * KB1
        k2 = current_omega * KB2

        if adaptive:
            nu = k2 * sign_s
            filter_dot = (
                nu - nu_filter[k]
            ) / TAU

            chi = (
                current_omega
                - OMEGA_FACTOR * np.abs(nu_filter[k])
                - EPS_B
            )

            omega_dot = -rho[k] * np.sign(chi)

            omega_dot = np.where(
                (current_omega <= OMEGA_MIN)
                & (omega_dot < 0),
                0.0,
                omega_dot
            )

            omega_dot = np.where(
                (current_omega >= OMEGA_MAX)
                & (omega_dot > 0),
                0.0,
                omega_dot
            )

            rho_dot = np.where(
                np.abs(chi) > np.abs(chi0),
                GAMMA * np.abs(chi),
                0.0
            )

            rho_dot = np.where(
                rho[k] >= RHO_MAX,
                0.0,
                rho_dot
            )

            phi = -(
                omega_dot / current_omega
            ) * current_s

        else:
            omega_dot = np.zeros(N)
            rho_dot = np.zeros(N)
            filter_dot = np.zeros(N)
            phi = np.zeros(N)

        ubar = (
            -k1 * np.sqrt(np.abs(current_s))
            * sign_s
            + theta[k]
            + phi
        )

        theta_dot = -k2 * sign_s
        L = disturbance_function(t[k])

        control[k] = ubar

        if k == steps - 1:
            break

        S_dot = ubar + L

        S[k + 1] = (
            S[k] + DT * S_dot
        )

        theta[k + 1] = (
            theta[k] + DT * theta_dot
        )

        if adaptive:
            Omega[k + 1] = np.clip(
                current_omega + DT * omega_dot,
                OMEGA_MIN,
                OMEGA_MAX
            )

            rho[k + 1] = np.clip(
                rho[k] + DT * rho_dot,
                0.0,
                RHO_MAX
            )

            nu_filter[k + 1] = (
                nu_filter[k] + DT * filter_dot
            )
        else:
            Omega[k + 1] = OMEGA_FIXED
            rho[k + 1] = rho[k]
            nu_filter[k + 1] = nu_filter[k]

    return S, Omega, control


S_ad, Omega_ad, U_ad = simulate(adaptive=True)
S_fx, Omega_fx, U_fx = simulate(adaptive=False)

# ============================================================
# PERFORMANCE METRICS
# ============================================================

def rmse(values):
    return np.sqrt(np.mean(values**2, axis=0))


def control_rms(values):
    return np.sqrt(np.mean(values**2, axis=0))


def settling_time(signal, threshold=0.01):
    """
    Last-exit settling time.
    Returns NaN if not settled by simulation end.
    """
    outside = np.where(np.abs(signal) > threshold)[0]

    if len(outside) == 0:
        return 0.0

    if outside[-1] == len(signal) - 1:
        return np.nan

    return t[outside[-1] + 1]


rows = []

for i in range(N):
    rows.append({
        "Vehicle": i + 1,
        "Adaptive_RMSE": rmse(S_ad)[i],
        "Fixed_RMSE": rmse(S_fx)[i],
        "Adaptive_Control_RMS": control_rms(U_ad)[i],
        "Fixed_Control_RMS": control_rms(U_fx)[i],
        "Adaptive_Settling_s": settling_time(S_ad[:, i]),
        "Fixed_Settling_s": settling_time(S_fx[:, i])
    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "adaptive_fixed_comparison.csv",
    index=False
)

print("\nADAPTIVE VS FIXED-GAIN COMPARISON")
print(summary.round(5).to_string(index=False))

# ============================================================
# BLACK AND WHITE FIGURES
# ============================================================

def plot_comparison(vehicle_index, filename):
    i = vehicle_index - 1

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        t, S_ad[:, i],
        "k-", linewidth=1.5,
        label="Adaptive STISMC"
    )

    ax.plot(
        t, S_fx[:, i],
        "k--", linewidth=1.5,
        label="Fixed-gain STISMC"
    )

    ax.axhline(0, color="gray", linewidth=0.8)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sliding surface")
    ax.set_title(
        f"Vehicle {vehicle_index}: "
        "Adaptive vs Fixed-Gain STISMC"
    )
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUT / filename,
        dpi=300,
        facecolor="white"
    )
    plt.close(fig)


plot_comparison(
    1, "10_adaptive_vs_fixed_vehicle1.png"
)

plot_comparison(
    6, "11_adaptive_vs_fixed_vehicle6.png"
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    t, Omega_ad[:, 0],
    "k-", linewidth=1.5,
    label="Adaptive gain"
)

ax.plot(
    t, Omega_fx[:, 0],
    "k--", linewidth=1.5,
    label="Fixed gain"
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Gain Omega")
ax.set_title("Adaptive and Fixed Gain Evolution")
ax.grid(True, alpha=0.2)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "12_adaptive_vs_fixed_gain.png",
    dpi=300,
    facecolor="white"
)
plt.close(fig)

assert np.isfinite(S_ad).all()
assert np.isfinite(S_fx).all()
assert np.isfinite(U_ad).all()
assert np.isfinite(U_fx).all()

print("\nSaved:")
print("adaptive_fixed_comparison.csv")
print("10_adaptive_vs_fixed_vehicle1.png")
print("11_adaptive_vs_fixed_vehicle6.png")
print("12_adaptive_vs_fixed_gain.png")

print("\nNOTE: Reduced-order comparison only.")
print("No full platoon plant is simulated here.")
