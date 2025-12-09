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

peaks_indices, _ = find_peaks(p_lacI, distance=30)
peak_times = t[peaks_indices]
peak_heights = p_lacI[peaks_indices]

steady_start_time = t[-1]
for i in range(len(peak_heights) - 10):
    window = peak_heights[i : i + 10]
    if np.std(window) / np.mean(window) < 0.0005:
        steady_start_time = peak_times[i]
        break

steady_mask = t >= steady_start_time
t_steady = t[steady_mask]
lacI_steady = p_lacI[steady_mask]

peaks_steady, _ = find_peaks(lacI_steady)
troughs_steady, _ = find_peaks(-lacI_steady)

if len(peaks_steady) > 1:
    global_mean_T = np.mean(np.diff(t_steady[peaks_steady]))
else:
    global_mean_T = 0

if len(peaks_steady) > 0 and len(troughs_steady) > 0:
    mean_max = np.mean(lacI_steady[peaks_steady])
    mean_min = np.mean(lacI_steady[troughs_steady])
    global_mean_A = (mean_max - mean_min) / 2
    global_center = (mean_max + mean_min) / 2
else:
    global_mean_A, global_center = 0, 0


window_start, window_end = 85, 98
window_mask = (t >= window_start) & (t <= window_end)
t_window = t[window_mask]

p_lacI_window = p_lacI[window_mask]
p_tetR_window = p_tetR[window_mask]
p_cI_window = p_cI[window_mask]

peaks_window, _ = find_peaks(p_lacI_window, distance=30)

if len(peaks_window) >= 2:
    peak1_idx = peaks_window[0]
    peak2_idx = peaks_window[1]

    t_p1 = t_window[peak1_idx]
    t_p2 = t_window[peak2_idx]
    y_p1 = p_lacI_window[peak1_idx]
    y_p2 = p_lacI_window[peak2_idx]
else:
    t_p1, t_p2, y_p1, y_p2 = None, None, None, None

c_tetR = "#FF8C00"
c_lacI = "#DC143C"
c_cI = "#0066CC"
c_annot = "#333333"

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(t, p_tetR, color="#FF8C00", linewidth=1.0, label="TetR", alpha=0.9, zorder=3)
ax.plot(
    t,
    p_lacI,
    color="#DC143C",
    linewidth=1.8,
    label="LacI (forced)",
    alpha=0.9,
    zorder=3,
)
ax.plot(t, p_cI, color="#0066CC", linewidth=1.0, label="λcI", alpha=0.9, zorder=3)

# centerline
if global_center != 0:
    ax.axhline(
        y=global_center, color="gray", linestyle=":", linewidth=1, alpha=0.6, zorder=2
    )
    ax.text(
        window_start + 0.2,
        global_center - 0.2,
        "Oscillation Center (LacI)",
        color="gray",
        fontsize=11,
    )

if t_p1 is not None:
    amp_x_range = t_window[peak1_idx - 30 : peak1_idx + 30]

    amp_x = t_p1
    ax.annotate(
        "",
        xy=(amp_x, y_p1),
        xytext=(amp_x, global_center),
        arrowprops=dict(arrowstyle="<->", color=c_annot, lw=1.5),
    )

    ax.text(
        amp_x + 0.3,
        (y_p1 + global_center) / 2,
        f"Amplitude\n$A = {global_mean_A:.2f}$",
        fontsize=12,
        color=c_annot,
        va="center",
    )

    y_dims = y_p1 + 0.4

    ax.plot([t_p1, t_p1], [y_p1, y_dims], color="gray", linestyle=":", lw=1, alpha=0.8)
    ax.plot([t_p2, t_p2], [y_p2, y_dims], color="gray", linestyle=":", lw=1, alpha=0.8)

    ax.annotate(
        "",
        xy=(t_p1, y_dims),
        xytext=(t_p2, y_dims),
        arrowprops=dict(arrowstyle="<->", color=c_annot, lw=1.5),
    )

    ax.text(
        (t_p1 + t_p2) / 2,
        y_dims + 0.05,
        f"Period $T = {global_mean_T:.2f}$",
        ha="center",
        va="bottom",
        fontsize=12,
        color=c_annot,
    )

    # peaks
    ax.plot(
        t_window[peaks_window],
        p_lacI_window[peaks_window],
        "o",
        color="white",
        markeredgecolor=c_lacI,
        markeredgewidth=1.5,
        markersize=7,
        zorder=4,
    )

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
ax.set_ylim([0, 7])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_linewidth(1.0)

ax.tick_params(axis="both", which="major", labelsize=12)

plt.tight_layout()
plt.show()
