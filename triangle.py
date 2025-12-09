import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sawtooth

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6

# default parametres
alpha_base = 10
A = 0.5
T = 5.92


def get_trapezoidal_wave(t, T):
    triangle = sawtooth(2 * np.pi * t / T + np.pi, width=0.5)
    steepness = 2.0
    amplified_triangle = triangle * steepness
    return np.clip(amplified_triangle, -1, 1)


def alpha_forced(t):
    return alpha_base * (1 + A * get_trapezoidal_wave(t, T))


t = np.linspace(0, 4 * T, 1000)

alpha_values = alpha_forced(t)

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    t, alpha_values, color="#DC143C", linewidth=2, label=r"$\alpha(t)_{\text{trapez}}$"
)

ax.axhline(
    y=alpha_base,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label=rf"$\alpha_{{\text{{base}}}} = {alpha_base}$",
)
ax.axhline(
    y=alpha_base * (1 + A), color="lightgray", linestyle=":", linewidth=2, alpha=0.7
)
ax.axhline(
    y=alpha_base * (1 - A), color="lightgray", linestyle=":", linewidth=1.5, alpha=0.7
)

ax.text(
    4 * T + 0.3,
    alpha_base * (1 + A),
    r"$\alpha_{\mathrm{max}} = \alpha_{\mathrm{base}}(1+A)$",
    fontsize=12,
    va="center",
    ha="left",
)

ax.text(
    4 * T + 0.3,
    alpha_base * (1 - A),
    r"$\alpha_{\mathrm{min}} = \alpha_{\mathrm{base}}(1-A)$",
    fontsize=12,
    va="center",
    ha="left",
)

first_peak_time = T
second_peak_time = 2 * T
y_bracket = alpha_base * (1 + A) * 1.02

ax.annotate(
    "",
    xy=(second_peak_time, y_bracket),
    xytext=(first_peak_time, y_bracket),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
)
ax.text(
    (first_peak_time + second_peak_time) / 2,
    y_bracket + 0.3,
    "T",
    fontsize=14,
    ha="center",
    va="bottom",
    fontweight="bold",
)

ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)

ax.set_xlabel("Time [mRNA lifetimes]", fontsize=16)
ax.set_ylabel(r"$\alpha(t)_{\text{trapez}}$ [dimensionless]", fontsize=16)

# legend
ax.legend(
    bbox_to_anchor=(1.25, 0.5),
    loc="center right",
    frameon=True,
    fancybox=True,
    shadow=False,
    fontsize=13,
    framealpha=0.95,
    edgecolor="gray",
    borderpad=0.8,
)

ax.set_xlim([0, 4 * T])
ax.set_ylim([alpha_base * (1 - A) * 0.9, alpha_base * (1 + A) * 1.15])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_linewidth(1.0)

ax.tick_params(axis="both", which="major", labelsize=12)
plt.tight_layout()
plt.show()
