from multiprocessing import Pool

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

# default parameters
alpha_0 = 0.01
n = 2
beta = 5
T_natural = 5.92

# forcing parameters
alpha_base = 10

# on off
force_lacI = True
force_tetR = False
force_cI = False


def alpha_forced(t, A, T, gene_forced=True):
    if gene_forced:
        return alpha_base * (1 + A * np.cos(2 * np.pi * t / T))
    else:
        return alpha_base


def repressilator_odes_forced(y, t, A, T):
    m_lacI, m_tetR, m_cI, p_lacI, p_tetR, p_cI = y

    alpha_lacI = alpha_forced(t, A, T, force_lacI)
    alpha_tetR = alpha_forced(t, A, T, force_tetR)
    alpha_cI = alpha_forced(t, A, T, force_cI)

    # mRNA
    dm_lacI_dt = -m_lacI + alpha_lacI / (1 + p_cI**n) + alpha_0
    dm_tetR_dt = -m_tetR + alpha_tetR / (1 + p_lacI**n) + alpha_0
    dm_cI_dt = -m_cI + alpha_cI / (1 + p_tetR**n) + alpha_0

    # protein
    dp_lacI_dt = -beta * (p_lacI - m_lacI)
    dp_tetR_dt = -beta * (p_tetR - m_tetR)
    dp_cI_dt = -beta * (p_cI - m_cI)

    return [dm_lacI_dt, dm_tetR_dt, dm_cI_dt, dp_lacI_dt, dp_tetR_dt, dp_cI_dt]


def simulate_point(args):
    A, T, y0, t = args
    solution = odeint(repressilator_odes_forced, y0, t, args=(A, T))

    # transient be gone !
    cutoff = len(t) // 2

    p_lacI = solution[cutoff:, 3]
    p_tetR = solution[cutoff:, 4]
    p_cI = solution[cutoff:, 5]

    amp_lacI = (np.max(p_lacI) - np.min(p_lacI)) / 2
    amp_tetR = (np.max(p_tetR) - np.min(p_tetR)) / 2
    amp_cI = (np.max(p_cI) - np.min(p_cI)) / 2

    return np.mean([amp_lacI, amp_tetR, amp_cI])


if __name__ == "__main__":
    y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]

    t = np.linspace(0, 300, 50000)

    T_values = np.linspace(1, 15, 200)
    A_values = np.linspace(0, 1, 200)

    # simulation
    param_grid = [(A, T, y0, t) for A in A_values for T in T_values]

    with Pool() as pool:
        results = pool.map(simulate_point, param_grid)

    synchronization_strength = np.array(results).reshape(len(A_values), len(T_values))

    # figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # plot
    contour = ax.contourf(
        T_values,
        A_values,
        synchronization_strength,
        levels=50,
        cmap="viridis",
        zorder=0,
    )

    # entrainment lines
    ax.axvline(
        x=T_natural,
        color="white",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="Natural Period $T_0$",
        zorder=2,
    )
    ax.axvline(
        x=T_natural / 2,
        color="white",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
        label="1:2 Resonance",
        zorder=2,
    )
    ax.axvline(
        x=T_natural * 2,
        color="white",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
        label="2:1 Resonance",
        zorder=2,
    )

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Mean Response Amplitude [$K_M$]", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel(r"Forcing Period $T$", fontsize=16)
    ax.set_ylabel(r"Forcing Amplitude $A$", fontsize=16)

    ax.grid(True, alpha=0.1, linestyle="-", linewidth=0.5, zorder=1, color="white")
    ax.set_axisbelow(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", labelsize=12)

    # legend
    ax.legend(
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        frameon=True,
        fancybox=True,
        fontsize=12,
        framealpha=0.2,
        edgecolor="white",
        labelcolor="white",
    )

    plt.tight_layout()
    plt.show()
