import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6

alpha_0 = 0.01
n = 2
beta = 5
alpha_base = 10
T_forcing = 5.92


def repressilator_odes_forced(y, t, A):
    m_lacI, m_tetR, m_cI, p_lacI, p_tetR, p_cI = y

    alpha_lacI = alpha_base * (1 + A * np.cos(2 * np.pi * t / T_forcing))
    alpha_tetR = alpha_base
    alpha_cI = alpha_base

    dm_lacI_dt = -m_lacI + alpha_lacI / (1 + p_cI**n) + alpha_0
    dm_tetR_dt = -m_tetR + alpha_tetR / (1 + p_lacI**n) + alpha_0
    dm_cI_dt = -m_cI + alpha_cI / (1 + p_tetR**n) + alpha_0

    dp_lacI_dt = -beta * (p_lacI - m_lacI)
    dp_tetR_dt = -beta * (p_tetR - m_tetR)
    dp_cI_dt = -beta * (p_cI - m_cI)

    return [dm_lacI_dt, dm_tetR_dt, dm_cI_dt, dp_lacI_dt, dp_tetR_dt, dp_cI_dt]


A_values = np.linspace(0, 1.0, 100)
amps_LacI = []
amps_TetR = []
amps_cI = []

# simulation settings
t_max = 300
points = 50000
t = np.linspace(0, t_max, points)

steady_start_time = 49

y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]

for i, A in enumerate(A_values):
    solution = odeint(repressilator_odes_forced, y0, t, args=(A,))

    p_lacI = solution[:, 3]
    p_tetR = solution[:, 4]
    p_cI = solution[:, 5]

    mask = t >= steady_start_time

    p_lacI_steady = p_lacI[mask]
    p_tetR_steady = p_tetR[mask]
    p_cI_steady = p_cI[mask]

    amps_LacI.append((np.max(p_lacI_steady) - np.min(p_lacI_steady)) / 2)
    amps_TetR.append((np.max(p_tetR_steady) - np.min(p_tetR_steady)) / 2)
    amps_cI.append((np.max(p_cI_steady) - np.min(p_cI_steady)) / 2)

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(A_values, amps_TetR, color="#FF8C00", linewidth=1.8, label="TetR", zorder=3)
ax.plot(
    A_values, amps_LacI, color="#DC143C", linewidth=1.8, label="LacI (Forced)", zorder=3
)
ax.plot(A_values, amps_cI, color="#0066CC", linewidth=1.8, label="λcI", zorder=3)

ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=2)
ax.text(
    0.535,
    ax.get_ylim()[1] * 0.69,
    r"$A=0.5$",
    ha="center",
    va="top",
    fontsize=12,
    color="gray",
)

ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.5, zorder=1)
ax.set_axisbelow(True)

ax.set_xlabel(r"Forcing Amplitude $A$", fontsize=16)
ax.set_ylabel(r"Mean Amplitude $[K_M$]", fontsize=16)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_linewidth(1.0)

ax.tick_params(axis="both", which="major", labelsize=12)

ax.legend(
    bbox_to_anchor=(1.012, 1),
    loc="upper left",
    frameon=True,
    fancybox=True,
    fontsize=13,
    framealpha=0.95,
    edgecolor="gray",
    borderpad=0.8,
)

ax.set_xlim(0, 1.0)

plt.tight_layout()
plt.show()
