import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
from lib.plotstyle import apply_style, save_fig

apply_style()


def _light_color(color, percent: float = 0.55):
    rgb = np.array(to_rgb(color))
    return tuple((1.0 - percent) * np.ones(3) + percent * rgb)

# Margin z = y * f(x). For a correct, confident prediction z >> 0; for a wrong one z < 0.
z = np.linspace(-3, 3, 800)

# Hinge loss (SVM):            max(0, 1 - z)
hinge = np.maximum(0.0, 1.0 - z)

# Logistic loss:               log2(1 + exp(-z))   (base-2 so it passes through (0, 1))
logistic = np.log2(1.0 + np.exp(-z))

# Modified Huber loss:         (1 - z)^2   for z >= -1
#                              -4 z        for z <  -1
modified_huber = np.where(z >= -1.0, np.maximum(0.0, 1.0 - z) ** 2, -4.0 * z)

fig, ax = plt.subplots()

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
curves = [
    (hinge, "Hinge loss"),
    (logistic, "Logistic loss"),
    (modified_huber, "Modified Huber loss"),
]
for (y, label), color in zip(curves, cycle):
    # Dark-outside, light-inside — matches the line style in run_default_gap.py.
    ax.plot(z, y, color=_light_color(color), linewidth=2.0, label=label,
            path_effects=[
                path_effects.Stroke(linewidth=3.0, foreground=color),
                path_effects.Normal(),
            ])

ax.axhline(0, color="black", ls="--", lw=0.9, zorder=1)
ax.axvline(0, color="black", ls="--", lw=0.9, zorder=1)

ax.set_xlim(-3, 3)
ax.set_ylim(-0.2, 8)
ax.set_xlabel(r"margin   $z = y \cdot f(x)$")
ax.set_ylabel("loss")
ax.set_title("Surrogate loss functions for binary classification")
ax.legend(loc="upper right")

save_fig(fig, Path(__file__).resolve().parent, "loss_functions")
