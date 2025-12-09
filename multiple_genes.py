import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks

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

# forcing parameters
alpha_base = 10
A = 0.5
T = 5.92


def alpha_forced(t, gene_forced=True):
    if gene_forced:
        return alpha_base * (1 + A * np.cos(2 * np.pi * t / T))
    else:
        return alpha_base


def repressilator_odes_forced(y, t, force_lacI, force_tetR, force_cI):
    m_lacI, m_tetR, m_cI, p_lacI, p_tetR, p_cI = y

    alpha_lacI = alpha_forced(t, force_lacI)
    alpha_tetR = alpha_forced(t, force_tetR)
    alpha_cI = alpha_forced(t, force_cI)

    # mRNA
    dm_lacI_dt = -m_lacI + alpha_lacI / (1 + p_cI**n) + alpha_0
    dm_tetR_dt = -m_tetR + alpha_tetR / (1 + p_lacI**n) + alpha_0
    dm_cI_dt = -m_cI + alpha_cI / (1 + p_tetR**n) + alpha_0

    # protein
    dp_lacI_dt = -beta * (p_lacI - m_lacI)
    dp_tetR_dt = -beta * (p_tetR - m_tetR)
    dp_cI_dt = -beta * (p_cI - m_cI)

    return [dm_lacI_dt, dm_tetR_dt, dm_cI_dt, dp_lacI_dt, dp_tetR_dt, dp_cI_dt]


def detect_steady_state(t, p_lacI):
    peaks_indices, _ = find_peaks(p_lacI, distance=30)
    peak_times = t[peaks_indices]
    peak_heights = p_lacI[peaks_indices]

    steady_start_time = t[-1]
    check_window = 10
    tolerance = 0.0005

    if len(peak_heights) > check_window:
        for i in range(len(peak_heights) - check_window):
            window = peak_heights[i : i + check_window]
            relative_variation = np.std(window) / np.mean(window)

            if relative_variation < tolerance:
                steady_start_time = peak_times[i]
                break

    return steady_start_time


def simulate_and_plot(ax, force_lacI, force_tetR, force_cI, title):
    y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]
    t = np.linspace(0, 300, 50000)

    # solve
    solution = odeint(
        repressilator_odes_forced, y0, t, args=(force_lacI, force_tetR, force_cI)
    )

    p_lacI = solution[:, 3]
    p_tetR = solution[:, 4]
    p_cI = solution[:, 5]

    steady_start_time = detect_steady_state(t, p_lacI)

    if steady_start_time < t[-1]:
        ax.axvspan(
            0,
            steady_start_time,
            alpha=0.12,
            color="orange",
            label="Transient",
            zorder=0,
        )
        ax.axvspan(
            steady_start_time,
            t[-1],
            alpha=0.12,
            color="lightblue",
            label="Steady state",
            zorder=0,
        )
    else:
        ax.axvspan(
            0,
            t[-1],
            alpha=0.12,
            color="orange",
            label="Transient (Not Settled)",
            zorder=0,
        )

    ax.plot(
        t,
        p_tetR,
        color="#FF8C00",
        linewidth=1.8,
        label="TetR" + (" (forced)" if force_tetR else ""),
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        t,
        p_lacI,
        color="#DC143C",
        linewidth=1.8,
        label="LacI" + (" (forced)" if force_lacI else ""),
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        t,
        p_cI,
        color="#0066CC",
        linewidth=1.8,
        label="λcI" + (" (forced)" if force_cI else ""),
        alpha=0.9,
        zorder=3,
    )

    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time [mRNA lifetimes]", fontsize=16)
    ax.set_ylabel(r"Protein Concentration [$K_M$]", fontsize=16)
    ax.set_title(title, fontsize=12, loc="left")

    # legend
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        fontsize=13,
        framealpha=0.95,
        edgecolor="gray",
        borderpad=0.8,
    )

    ax.set_xlim([0, 150])
    ax.set_ylim([0, 7])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=12)

    print(f"{title}: Steady state detected at t = {steady_start_time:.2f}")


fig, axes = plt.subplots(3, 1, figsize=(14, 18))

simulate_and_plot(
    axes[0], force_lacI=True, force_tetR=False, force_cI=False, title="(a) LacI forced"
)

simulate_and_plot(
    axes[1],
    force_lacI=True,
    force_tetR=True,
    force_cI=False,
    title="(b) LacI and TetR forced",
)

simulate_and_plot(
    axes[2],
    force_lacI=True,
    force_tetR=True,
    force_cI=True,
    title="(c) LacI, TetR and λcI forced",
)

plt.tight_layout()

output_path = "/Users/canjost/Downloads/multiple.pdf"
plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
print(f"\nPDF saved to: {output_path}")

plt.show()
