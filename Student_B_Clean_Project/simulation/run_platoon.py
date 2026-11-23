"""Stage 1: verify the reference paper leader trajectory.
This is NOT yet the six-follower FTRE/STISMC simulation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
dt, duration = 0.001, 20.0
t = np.arange(0.0, duration + dt / 2, dt)
a0 = np.where((t >= 2.0) & (t < 6.0), 1.0, 0.0)
x0, v0 = np.empty_like(t), np.empty_like(t)
x0[0], v0[0] = 16.0, 2.0
for k in range(len(t) - 1):
    v0[k + 1] = v0[k] + a0[k] * dt
    x0[k + 1] = x0[k] + v0[k] * dt + 0.5 * a0[k] * dt**2
pd.DataFrame({"time": t, "x0": x0, "v0": v0, "a0": a0}).to_csv(OUT / "leader_reference.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(t, v0, color="black", lw=1.8)
ax.set(xlabel="Time (s)", ylabel="Leader velocity (m/s)", title="Reference leader velocity (stage 1)")
ax.grid(True, color="0.85")
fig.tight_layout()
fig.savefig(OUT / "leader_velocity.png", dpi=300)
plt.close(fig)
print("SUCCESS: leader trajectory created")
print(f"Final leader velocity: {v0[-1]:.3f} m/s")
print(f"Saved: {OUT / 'leader_velocity.png'}")
print(f"Saved: {OUT / 'leader_reference.csv'}")
print("NOTE: This is a leader-only test, not a validated platoon simulation.")
