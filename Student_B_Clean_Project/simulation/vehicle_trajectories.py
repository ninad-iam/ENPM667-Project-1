
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS / "results.csv")

t = df["time"]

styles = [
    "-", "--", "-.", ":", (0, (5, 2, 1, 2)),
    (0, (3, 1, 1, 1, 1, 1)), (0, (8, 3))
]

def plot_vehicles(prefix, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i in range(7):
        column = f"{prefix}{i}"
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

        label = "Leader" if i == 0 else f"Vehicle {i}"
        ax.plot(
            t, df[column],
            color="black",
            linestyle=styles[i],
            linewidth=1.5,
            label=label
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    fig.savefig(FIGURES / f"{filename}.png", dpi=300)
    fig.savefig(FIGURES / f"{filename}.pdf")
    plt.close(fig)

plot_vehicles(
    "x",
    "Position (m)",
    "Leader and Follower Vehicle Positions",
    "vehicle_positions"
)

plot_vehicles(
    "v",
    "Velocity (m/s)",
    "Leader and Follower Vehicle Velocities",
    "vehicle_velocities"
)

print("Successfully generated:")
print("vehicle_positions.png")
print("vehicle_velocities.png")
