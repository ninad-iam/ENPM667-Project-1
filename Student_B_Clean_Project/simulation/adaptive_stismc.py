
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STAGE 5: ADAPTIVE DUAL-LAYER SUPER-TWISTING CONTROLLER
#
# Reduced sliding-surface dynamics test.
# Based on paper Eqs. (30), (31), (34), (35).
#
# NOT YET THE COMPLETE VEHICLE PLATOON SIMULATION.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

N = 6
DT = 0.001
T = 20.0

KB2 = 2.0
KB1 = 2.0 * np.sqrt(2.0 * KB2)

# Initial test settings for dual-layer adaptation
OMEGA_INITIAL = 2.0
OMEGA_MIN = 0.2
OMEGA_MAX = 30.0
RHO_INITIAL = 0.5
RHO_MAX = 20.0

OMEGA_FACTOR = 1.5  # omega > 1 in paper Eq. (35)
EPS_B = 0.1
GAMMA = 0.5
TAU = 0.05

# Initial surfaces deliberately nonzero for testing
S_INITIAL = np.array([
    -0.8, -0.2, -0.1, -0.3, -0.1, -1.0
])

t = np.arange(0, T + DT / 2, DT)
steps = len(t)

S = np.zeros((steps, N))
theta = np.zeros((steps, N))
Omega = np.zeros((steps, N))
rho = np.zeros((steps, N))
nu_filter = np.zeros((steps, N))
control = np.zeros((steps, N))
disturbance = np.zeros((steps, N))

S[0] = S_INITIAL
Omega[0] = OMEGA_INITIAL
rho[0] = RHO_INITIAL

# Initial chi from Eq. (35)
chi0 = (
    Omega[0]
    - OMEGA_FACTOR * nu_filter[0]
    - EPS_B
)

def disturbance_function(time):
    """
    Smooth illustrative disturbance in reduced dynamics.
    NOT the reference paper's vehicle disturbance model.
    """
    return 0.12 * np.sin(
        0.8 * time + np.arange(N) * 0.3
    )

for k in range(steps):
    current_s = S[k]
    current_omega = Omega[k]
    current_rho = rho[k]
    current_filter = nu_filter[k]

    sign_s = np.sign(current_s)

    # Paper Eq. (30):
    # k1 = sqrt(Omega) * kb1
    # k2 = Omega * kb2
    k1 = np.sqrt(current_omega) * KB1
    k2 = current_omega * KB2

    # Equivalent-control approximation, Eq. (34)
    nu = k2 * sign_s

    filter_dot = (
        nu - current_filter
    ) / TAU

    # Dual-layer adaptation, Eq. (35)
    chi = (
        current_omega
        - OMEGA_FACTOR * np.abs(current_filter)
        - EPS_B
    )

    omega_dot = -current_rho * np.sign(chi)

    # Keep gain within numerical test bounds
    omega_dot = np.where(
        (current_omega <= OMEGA_MIN) & (omega_dot < 0),
        0.0,
        omega_dot
    )

    omega_dot = np.where(
        (current_omega >= OMEGA_MAX) & (omega_dot > 0),
        0.0,
        omega_dot
    )

    rho_dot = np.where(
        np.abs(chi) > np.abs(chi0),
        GAMMA * np.abs(chi),
        0.0
    )

    rho_dot = np.where(
        current_rho >= RHO_MAX,
        0.0,
        rho_dot
    )

    # Paper's adaptive compensation term
    phi = -(
        omega_dot / current_omega
    ) * current_s

    # Super-twisting control, Eq. (30)
    ubar = (
        -k1 * np.sqrt(np.abs(current_s)) * sign_s
        + theta[k]
        + phi
    )

    theta_dot = -k2 * sign_s

    L = disturbance_function(t[k])

    control[k] = ubar
    disturbance[k] = L

    if k == steps - 1:
        break

    # Reduced dynamics, Eq. (31)
    S_dot = ubar + L

    S[k + 1] = S[k] + DT * S_dot
    theta[k + 1] = theta[k] + DT * theta_dot

    Omega[k + 1] = np.clip(
        current_omega + DT * omega_dot,
        OMEGA_MIN,
        OMEGA_MAX
    )

    rho[k + 1] = np.clip(
        current_rho + DT * rho_dot,
        0.0,
        RHO_MAX
    )

    nu_filter[k + 1] = (
        current_filter + DT * filter_dot
    )

# ============================================================
# EXPORT RESULTS
# ============================================================

data = {"time": t}

for i in range(N):
    data[f"S{i+1}"] = S[:, i]
    data[f"Omega{i+1}"] = Omega[:, i]
    data[f"theta{i+1}"] = theta[:, i]
    data[f"ubar{i+1}"] = control[:, i]
    data[f"L{i+1}"] = disturbance[:, i]

pd.DataFrame(data).to_csv(
    OUT / "adaptive_stismc_test.csv",
    index=False
)

# ============================================================
# BLACK AND WHITE PLOTTING
# ============================================================

styles = [
    "-", "--", "-.", ":",
    (0, (5, 2)), (0, (3, 1, 1, 1))
]

def save_graph(values, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(N):
        ax.plot(
            t,
            values[:, i],
            color="black",
            linestyle=styles[i],
            linewidth=1.3,
            label=f"Vehicle {i+1}"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUT / filename,
        dpi=300,
        facecolor="white"
    )
    plt.close(fig)

save_graph(
    S,
    "Sliding surface",
    "Adaptive STISMC: Reduced Surface Dynamics",
    "06_adaptive_stismc_surfaces.png"
)

save_graph(
    Omega,
    "Adaptive gain Omega",
    "Adaptive Dual-Layer Gain Evolution",
    "07_adaptive_omega.png"
)

save_graph(
    control,
    "Transformed control input",
    "Adaptive STISMC Control Signals",
    "08_adaptive_control.png"
)

# ============================================================
# CHECKS
# ============================================================

assert np.isfinite(S).all()
assert np.isfinite(Omega).all()
assert np.isfinite(control).all()

print("\nSUCCESS: Adaptive STISMC reduced test completed!")
print(f"KB1 = {KB1:.3f}")
print(f"KB2 = {KB2:.3f}")

print("\nInitial sliding surfaces:")
print(np.round(S[0], 4))

print("\nFinal sliding surfaces:")
print(np.round(S[-1], 6))

print("\nFinal adaptive gains:")
print(np.round(Omega[-1], 4))

print("\nGenerated files:")
print("06_adaptive_stismc_surfaces.png")
print("07_adaptive_omega.png")
print("08_adaptive_control.png")
print("adaptive_stismc_test.csv")

print("\nNOTE: This is a reduced controller test.")
print("It does not yet simulate the complete vehicle plant.")


fig, ax = plt.subplots(figsize=(10, 6))

for i in range(N):
    ax.plot(
        t,
        S[:, i],
        color="black",
        linestyle=styles[i],
        linewidth=1.4,
        label=f"Vehicle {i+1}"
    )

ax.set_xlim(0, 2)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Sliding surface")
ax.set_title("Adaptive STISMC: Initial Convergence")
ax.grid(True, alpha=0.2)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "09_stismc_convergence_zoom.png",
    dpi=300,
    facecolor="white"
)
plt.close(fig)
