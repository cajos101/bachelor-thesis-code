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
alpha = 10
alpha_0 = 0.01
n = 2
beta = 5


def repressilator_odes(y, t, alpha, alpha_0, n, beta):
    m_lacI, m_tetR, m_cI, p_lacI, p_tetR, p_cI = y

    # mRNA
    dm_lacI_dt = -m_lacI + alpha / (1 + p_cI**n) + alpha_0
    dm_tetR_dt = -m_tetR + alpha / (1 + p_lacI**n) + alpha_0
    dm_cI_dt = -m_cI + alpha / (1 + p_tetR**n) + alpha_0

    # protein
    dp_lacI_dt = -beta * (p_lacI - m_lacI)
    dp_tetR_dt = -beta * (p_tetR - m_tetR)
    dp_cI_dt = -beta * (p_cI - m_cI)

    return [dm_lacI_dt, dm_tetR_dt, dm_cI_dt, dp_lacI_dt, dp_tetR_dt, dp_cI_dt]


# initial conditions
y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]

t = np.linspace(0, 300, 50000)
solution = odeint(repressilator_odes, y0, t, args=(alpha, alpha_0, n, beta))

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

# peaks
peaks_window, _ = find_peaks(p_lacI_window, distance=30)

if len(peaks_window) >= 2:
    peak1_idx = peaks_window[0]
    peak2_idx = peaks_window[1]
    peak_val = p_lacI_window[peak1_idx]

    # troughs
    trough_region = p_lacI_window[peak1_idx:peak2_idx]
    trough_val = np.min(trough_region)
    mean_val = (peak_val + trough_val) / 2
    amplitude = peak_val - mean_val

    period_T = t_window[peak2_idx] - t_window[peak1_idx]
    t_start_period = t_window[peak1_idx]
    t_end_period = t_window[peak2_idx]
else:
    peak1_idx, period_T, mean_val = None, None, None

c_tetR = "#FF8C00"
c_lacI = "#DC143C"
c_cI = "#0066CC"
c_annot = "#333333"

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(t, p_tetR, color="#FF8C00", linewidth=1.8, label="TetR", alpha=0.9, zorder=3)
ax.plot(t, p_lacI, color="#DC143C", linewidth=1.8, label="LacI", alpha=0.9, zorder=3)
ax.plot(t, p_cI, color="#0066CC", linewidth=1.8, label="λcI", alpha=0.9, zorder=3)

# centerline
if mean_val is not None:
    ax.axhline(
        y=mean_val, color="gray", linestyle=":", linewidth=1, alpha=0.6, zorder=2
    )
    ax.text(
        window_start + 0.2,
        mean_val - 0.2,
        "Oscillation Center",
        color="gray",
        fontsize=11,
    )

if len(peaks_window) >= 2:
    amp_x_range = t_window[peak1_idx - 30 : peak1_idx + 30]
    amp_y_range = p_lacI_window[peak1_idx - 30 : peak1_idx + 30]

    mask_above = amp_y_range >= mean_val

    amp_x = t_window[peak1_idx]
    ax.annotate(
        "",
        xy=(amp_x, peak_val),
        xytext=(amp_x, mean_val),
        arrowprops=dict(arrowstyle="<->", color=c_annot, lw=1.5),
    )

    ax.text(
        amp_x + 0.3,
        (peak_val + mean_val) / 2,
        f"Amplitude\n$A = {amplitude:.2f}$",
        fontsize=12,
        color=c_annot,
        va="center",
    )

    y_dims = peak_val + 0.4

    ax.plot(
        [t_start_period, t_start_period],
        [peak_val, y_dims],
        color="gray",
        linestyle=":",
        lw=1,
        alpha=0.8,
    )
    ax.plot(
        [t_end_period, t_end_period],
        [peak_val, y_dims],
        color="gray",
        linestyle=":",
        lw=1,
        alpha=0.8,
    )

    ax.annotate(
        "",
        xy=(t_start_period, y_dims),
        xytext=(t_end_period, y_dims),
        arrowprops=dict(arrowstyle="<->", color=c_annot, lw=1.5),
    )

    ax.text(
        (t_start_period + t_end_period) / 2,
        y_dims + 0.05,
        f"Period $T = {period_T:.2f}$",
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
ax.set_ylim([0, 4])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_linewidth(1.0)

ax.tick_params(axis="both", which="major", labelsize=12)

plt.tight_layout()
plt.show()
