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
T = 2

# on off
force_lacI = True
force_tetR = False
force_cI = False


def alpha_forced(t, gene_forced=True):
    if gene_forced:
        return alpha_base * (1 + A * np.cos(2 * np.pi * t / T))
    else:
        return alpha_base


def repressilator_odes_forced(y, t, alpha_base, A, T, alpha_0, n, beta):
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


# initial conditions
y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]

t = np.linspace(0, 300, 50000)
solution = odeint(
    repressilator_odes_forced, y0, t, args=(alpha_base, A, T, alpha_0, n, beta)
)

# protein concentrations
p_lacI = solution[:, 3]
p_tetR = solution[:, 4]
p_cI = solution[:, 5]

window_start, window_end = 85, 98
window_mask = (t >= window_start) & (t <= window_end)
t_window = t[window_mask]

p_lacI_window = p_lacI[window_mask]
p_tetR_window = p_tetR[window_mask]
p_cI_window = p_cI[window_mask]

peaks_lacI, _ = find_peaks(p_lacI_window, distance=30)

# colors
c_tetR = "#FF8C00"
c_lacI = "#DC143C"
c_cI = "#0066CC"
c_annot = "#333333"

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(t, p_tetR, color=c_tetR, linewidth=1.0, label="TetR", alpha=0.9, zorder=3)
ax.plot(
    t, p_lacI, color=c_lacI, linewidth=1.8, label="LacI (forced)", alpha=0.9, zorder=3
)
ax.plot(t, p_cI, color=c_cI, linewidth=1.0, label="λcI", alpha=0.9, zorder=3)

if len(peaks_lacI) >= 4:

    def draw_period_dim(idx_start, idx_end, y_level, label):
        t_s = t_window[peaks_lacI[idx_start]]
        y_s = p_lacI_window[peaks_lacI[idx_start]]
        t_e = t_window[peaks_lacI[idx_end]]
        y_e = p_lacI_window[peaks_lacI[idx_end]]

        ax.plot(
            [t_s, t_s], [y_s, y_level], color="gray", linestyle=":", lw=1, alpha=0.8
        )
        ax.plot(
            [t_e, t_e], [y_e, y_level], color="gray", linestyle=":", lw=1, alpha=0.8
        )

        ax.annotate(
            "",
            xy=(t_s, y_level),
            xytext=(t_e, y_level),
            arrowprops=dict(arrowstyle="<->", color=c_annot, lw=1.5),
        )

        period = t_e - t_s
        txt = ax.text(
            (t_s + t_e) / 2,
            y_level + 0.05,
            f"{label} = {period:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color=c_annot,
        )
        txt.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.5))

    y_max = np.max(p_lacI_window)
    h1 = y_max + 0.8
    h2 = h1 + 0.8

    # period 1-3
    draw_period_dim(0, 2, h1, "$T_{3-1}$")

    # Period 2-4
    draw_period_dim(1, 3, h2, "$T_{4-2}$")

    # numbered peaks
    for i, idx in enumerate(peaks_lacI[:4]):
        t_p = t_window[idx]
        y_p = p_lacI_window[idx]
        ax.plot(
            t_p,
            y_p,
            "o",
            color="white",
            markeredgecolor=c_lacI,
            markeredgewidth=1.5,
            markersize=7,
            zorder=4,
        )
        ax.text(
            t_p,
            y_p + 0.2,
            str(i + 1),
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

# legend
ax.plot(
    [],
    [],
    "o",
    markeredgecolor=c_lacI,
    markerfacecolor="white",
    markersize=7,
    markeredgewidth=1.5,
    label="Peaks",
)

ax.set_xlabel("Time [mRNA lifetimes]", fontsize=16)
ax.set_ylabel(r"Protein Concentration [$K_M$]", fontsize=16)

ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

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

ax.set_xlim([window_start, window_end])
ax.set_ylim([0, 7.5])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_linewidth(1.0)

ax.tick_params(axis="both", which="major", labelsize=12)

plt.tight_layout()
plt.show()
