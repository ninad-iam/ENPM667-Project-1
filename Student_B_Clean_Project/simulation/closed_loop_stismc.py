from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simplified closed-loop MCTH + coupled integral sliding + adaptive ST controller.
# NOT a complete reproduction of the reference paper's nonlinear plant / FTRE.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_closed_loop'
OUT.mkdir(parents=True, exist_ok=True)
N, DT, T = 6, 0.001, 20.0
D, H, ZETA, KI, Q = 0.5, 1.0, 1.0, 10.0, 0.9
KB2 = 2.0
KB1 = 2.0 * np.sqrt(2.0 * KB2)
OMEGA_MIN, OMEGA_MAX = 0.2, 30.0
RHO_MAX, TAU, GAMMA, EPS_B, OMEGA_FACTOR = 20.0, 0.05, 0.5, 0.1, 1.5
X0 = np.array([16., 14., 12., 10., 7., 4.2, 2.4])
V0 = np.array([2., 2.3, 1.7, 1.6, 2.8, 2.4, 2.3])
TIME = np.arange(0., T + DT / 2., DT)
STEPS = len(TIME)

# B maps local sliding derivatives to coupled sliding derivatives.
B = Q * np.eye(N) - np.diag(np.ones(N - 1), 1)
assert np.isclose(np.linalg.det(B), Q**N)

def leader_a(t):
    return 1.0 if 2.0 <= t < 6.0 else 0.0

def simulate(adaptive=True):
    x = np.zeros((STEPS, N + 1))
    v = np.zeros_like(x)
    a = np.zeros_like(x)
    e = np.zeros((STEPS, N))
    eb = np.zeros_like(e)
    s = np.zeros_like(e)
    S = np.zeros_like(e)
    omega = np.zeros_like(e)
    ubar = np.zeros_like(e)
    integral = np.zeros(N)
    theta = np.zeros(N)
    rho = np.full(N, 0.5)
    filt = np.zeros(N)
    x[0], v[0] = X0, V0
    omega[0] = 2.0

    # Consistent initialization: choose initial desired e_dot = 0.
    # Then psi_dot(0) = 0, and initial follower acceleration is
    # (v_predecessor - v_follower)/H. This is a design choice, not paper's.
    e_initial = x[0, :-1] - x[0, 1:] - D - H * v[0, 1:]
    kappa = ZETA * e_initial
    chi_initial = np.full(N, 2.0 - EPS_B)

    for k, tk in enumerate(TIME):
        e[k] = x[k, :-1] - x[k, 1:] - D - H * v[k, 1:]
        decay = np.exp(-ZETA * tk)
        psi = (e_initial + kappa * tk) * decay
        psi_dot = (kappa - ZETA * (e_initial + kappa * tk)) * decay
        eb[k] = e[k] - psi
        s[k] = eb[k] + KI * integral
        S[k] = B @ s[k]
        current_omega = omega[k]
        sg = np.sign(S[k])
        k1 = KB1 * np.sqrt(current_omega)
        k2 = KB2 * current_omega
        nu = k2 * sg
        if adaptive:
            chi = current_omega - OMEGA_FACTOR * np.abs(filt) - EPS_B
            omega_dot = -rho * np.sign(chi)
            omega_dot = np.where((current_omega <= OMEGA_MIN) & (omega_dot < 0), 0., omega_dot)
            omega_dot = np.where((current_omega >= OMEGA_MAX) & (omega_dot > 0), 0., omega_dot)
            rho_dot = np.where(np.abs(chi) > np.abs(chi_initial), GAMMA * np.abs(chi), 0.)
            phi = -(omega_dot / current_omega) * S[k]
        else:
            omega_dot = np.zeros(N)
            rho_dot = np.zeros(N)
            phi = np.zeros(N)
        w = -k1 * np.sqrt(np.abs(S[k])) * sg + theta + phi
        ubar[k] = w
        relative_v = v[k, :-1] - v[k, 1:]
        # s_dot = relative_v - H*a_follower - psi_dot + KI*eb
        # S_dot = B@s_dot = w (ideal acceleration-actuated plant)
        a[k, 0] = leader_a(tk)
        a[k, 1:] = (relative_v - psi_dot + KI * eb[k] - np.linalg.solve(B, w)) / H
        if k == STEPS - 1:
            break
        x[k + 1] = x[k] + DT * v[k] + 0.5 * DT**2 * a[k]
        v[k + 1] = v[k] + DT * a[k]
        integral += DT * eb[k]
        theta += DT * (-k2 * sg)
        if adaptive:
            omega[k + 1] = np.clip(current_omega + DT * omega_dot, OMEGA_MIN, OMEGA_MAX)
            rho = np.clip(rho + DT * rho_dot, 0., RHO_MAX)
            filt += DT * (nu - filt) / TAU
        else:
            omega[k + 1] = current_omega
    for arr in (x, v, a, e, eb, s, S, omega, ubar):
        if not np.isfinite(arr).all():
            raise RuntimeError('Non-finite values: simulation unstable')
    return dict(x=x, v=v, a=a, e=e, eb=eb, s=s, S=S, omega=omega, ubar=ubar)

def save_fig(t, data, title, ylabel, filename, first=0, labels=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (7, 2, 1, 2))]
    for j in range(data.shape[1]):
        ax.plot(t, data[:, j], color='black', linestyle=styles[j % len(styles)],
                linewidth=1.2, label=(labels[j] if labels else f'Vehicle {j + first}'))
    ax.set(xlabel='Time (s)', ylabel=ylabel, title=title)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, facecolor='white')
    plt.close(fig)

ad = simulate(True)
fx = simulate(False)
records = {'time': TIME}
for name, result in [('adaptive', ad), ('fixed', fx)]:
    for quantity, values in result.items():
        for j in range(values.shape[1]):
            records[f'{name}_{quantity}{j if quantity in ("x", "v", "a") else j+1}'] = values[:, j]
pd.DataFrame(records).to_csv(OUT / 'closed_loop_results.csv', index=False)

save_fig(TIME, ad['x'], 'Closed-loop Adaptive: Vehicle Positions', 'Position (m)', '01_positions.png', first=0)
save_fig(TIME, ad['v'], 'Closed-loop Adaptive: Vehicle Velocities', 'Velocity (m/s)', '02_velocities.png', first=0)
save_fig(TIME, ad['e'], 'Closed-loop Adaptive: CTH Spacing Errors', 'Error (m)', '03_spacing_errors.png', first=1)
save_fig(TIME, ad['eb'], 'Closed-loop Adaptive: MCTH Errors', 'Modified error (m)', '04_mcth_errors.png', first=1)
save_fig(TIME, ad['S'], 'Closed-loop Adaptive: Coupled Surfaces', 'Coupled surface', '05_coupled_surfaces.png', first=1)
save_fig(TIME, ad['omega'], 'Closed-loop Adaptive: Gain Evolution', 'Omega', '06_gains.png', first=1)
save_fig(TIME, ad['a'][:, 1:], 'Closed-loop Adaptive: Follower Accelerations', 'Acceleration (m/s^2)', '07_accelerations.png', first=1)

rows = []
for j in range(N):
    rows.append({'Vehicle': j+1,
                 'Adaptive_CTH_RMSE_m': np.sqrt(np.mean(ad['e'][:, j]**2)),
                 'Fixed_CTH_RMSE_m': np.sqrt(np.mean(fx['e'][:, j]**2)),
                 'Adaptive_MCTH_RMSE_m': np.sqrt(np.mean(ad['eb'][:, j]**2)),
                 'Fixed_MCTH_RMSE_m': np.sqrt(np.mean(fx['eb'][:, j]**2)),
                 'Adaptive_Max_Abs_Accel': np.max(np.abs(ad['a'][:, j+1])),
                 'Fixed_Max_Abs_Accel': np.max(np.abs(fx['a'][:, j+1]))})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'closed_loop_summary.csv', index=False)
print('\nCLOSED-LOOP COMPARISON (simplified acceleration-actuated model)')
print(summary.round(5).to_string(index=False))
print('\nAdaptive final coupled surfaces:', np.round(ad['S'][-1], 6))
print('Fixed final coupled surfaces:   ', np.round(fx['S'][-1], 6))
print('Adaptive minimum gaps (m):', np.round(np.min(ad['x'][:, :-1] - ad['x'][:, 1:], axis=0), 4))
print('Adaptive peak acceleration (m/s^2):', np.max(np.abs(ad['a'][:, 1:])))
print('\nOutputs:', OUT)
print('NOTE: Ideal acceleration-actuated illustrative model; no FTRE or nonlinear plant.')
