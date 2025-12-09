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


# Initial conditions
y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]

t = np.linspace(0, 300, 50000)
solution = odeint(repressilator_odes, y0, t, args=(alpha, alpha_0, n, beta))

p_lacI = solution[:, 3]
p_tetR = solution[:, 4]
p_cI = solution[:, 5]

steady_region_mask = t > 100
peaks_steady, _ = find_peaks(p_lacI[steady_region_mask])
if len(peaks_steady) > 1:
    global_T = np.mean(np.diff(t[steady_region_mask][peaks_steady]))
else:
    global_T = 0

window_start = 85
window_end = window_start + 2.2 * global_T
window_mask = (t >= window_start) & (t <= window_end)

t_window = t[window_mask]
p_lacI_window = p_lacI[window_mask]
p_tetR_window = p_tetR[window_mask]
p_cI_window = p_cI[window_mask]

# peak windows
peaks_lacI, _ = find_peaks(p_lacI_window, distance=20)
peaks_tetR, _ = find_peaks(p_tetR_window, distance=20)
peaks_cI, _ = find_peaks(p_cI_window, distance=20)

if len(peaks_lacI) >= 2 and len(peaks_tetR) >= 1 and len(peaks_cI) >= 1:
    t_peak_lacI_1 = t_window[peaks_lacI[0]]
    t_peak_lacI_2 = t_window[peaks_lacI[1]]

    p_peak_lacI_1 = p_lacI_window[peaks_lacI[0]]
    p_peak_lacI_2 = p_lacI_window[peaks_lacI[1]]

    ci_candidates = t_window[peaks_cI][t_window[peaks_cI] > t_peak_lacI_1]
    t_peak_cI = ci_candidates[0] if len(ci_candidates) > 0 else t_window[peaks_cI[0]]
    ci_peak_idx = np.where(t_window[peaks_cI] == t_peak_cI)[0][0]
    p_peak_cI = p_cI_window[peaks_cI[ci_peak_idx]]

    tet_candidates = t_window[peaks_tetR][t_window[peaks_tetR] > t_peak_lacI_1]
    t_peak_tetR = (
        tet_candidates[0] if len(tet_candidates) > 0 else t_window[peaks_tetR[0]]
    )
    tet_peak_idx = np.where(t_window[peaks_tetR] == t_peak_tetR)[0][0]
    p_peak_tetR = p_tetR_window[peaks_tetR[tet_peak_idx]]

    delay_cI_raw = t_peak_cI - t_peak_lacI_1
    delay_tetR_raw = t_peak_tetR - t_peak_lacI_1

    T_disp = round(global_T, 10)
    tau1_disp = round(delay_cI_raw, 10)
    tau2_disp = round(delay_tetR_raw, 10)

    phase_1_disp = (tau1_disp / T_disp) * 360
    phase_2_disp = (tau2_disp / T_disp) * 360

else:
    t_peak_lacI_1, t_peak_lacI_2, t_peak_cI, t_peak_tetR = None, None, None, None
    p_peak_lacI_1, p_peak_lacI_2, p_peak_cI, p_peak_tetR = None, None, None, None
    phase_1_disp, phase_2_disp = None, None
    tau1_disp, tau2_disp, T_disp = 0, 0, 0


# vis
fig, ax = plt.subplots(figsize=(14, 7))

c_lacI = "#DC143C"
c_tetR = "#FF8C00"
c_cI = "#0066CC"
c_annot = "#333333"

ax.plot(t, p_tetR, color=c_tetR, linewidth=1.8, label="TetR", alpha=0.9, zorder=3)
ax.plot(t, p_lacI, color=c_lacI, linewidth=1.8, label="LacI", alpha=0.9, zorder=3)
ax.plot(t, p_cI, color=c_cI, linewidth=1.8, label="λcI", alpha=0.9, zorder=3)

max_conc = max(np.max(p_lacI_window), np.max(p_tetR_window), np.max(p_cI_window))
arrow_y_low = max_conc * 1.1
arrow_y_mid = max_conc * 1.22
arrow_y_high = max_conc * 1.35


def draw_drop_line(t_val, y_start, y_end, color):
    ax.plot(
        [t_val, t_val],
        [y_start, y_end],
        linestyle=":",
        color=color,
        linewidth=1.5,
        alpha=0.6,
        zorder=0,
    )


if t_peak_lacI_1 is not None:
    draw_drop_line(t_peak_lacI_1, p_peak_lacI_1, arrow_y_high, c_lacI)
    draw_drop_line(t_peak_lacI_2, p_peak_lacI_2, arrow_y_high, c_lacI)
    draw_drop_line(t_peak_tetR, p_peak_tetR, arrow_y_mid, c_tetR)
    draw_drop_line(t_peak_cI, p_peak_cI, arrow_y_low, c_cI)

    arrow_props = dict(arrowstyle="<->", lw=1.5, color=c_annot)

    # arrow 1
    ax.annotate(
        "",
        xy=(t_peak_cI, arrow_y_low),
        xytext=(t_peak_lacI_1, arrow_y_low),
        arrowprops=arrow_props,
    )
    ax.text(
        (t_peak_lacI_1 + t_peak_cI) / 2,
        arrow_y_low + 0.05,
        f"$\\tau_1 = {tau1_disp:.2f}$",
        ha="center",
        va="bottom",
        fontsize=11,
        color=c_annot,
    )

    # arrow 2
    ax.annotate(
        "",
        xy=(t_peak_tetR, arrow_y_mid),
        xytext=(t_peak_lacI_1, arrow_y_mid),
        arrowprops=arrow_props,
    )
    ax.text(
        (t_peak_lacI_1 + t_peak_tetR) / 2,
        arrow_y_mid + 0.05,
        f"$\\tau_2 = {tau2_disp:.2f}$",
        ha="center",
        va="bottom",
        fontsize=11,
        color=c_annot,
    )

    # arrow 3
    ax.annotate(
        "",
        xy=(t_peak_lacI_2, arrow_y_high),
        xytext=(t_peak_lacI_1, arrow_y_high),
        arrowprops=arrow_props,
    )
    ax.text(
        (t_peak_lacI_1 + t_peak_lacI_2) / 2,
        arrow_y_high + 0.05,
        f"Period $T = {T_disp:.2f}$",
        ha="center",
        va="bottom",
        fontsize=12,
        color=c_annot,
    )

if phase_1_disp is not None:
    textstr = (
        r"$\theta_1 = \frac{360^\circ}{T} \cdot \tau_1$" + "\n"
        rf"$\ \ \ = {phase_1_disp:.1f}^\circ$" + "\n\n"
        r"$\theta_2 = \frac{360^\circ}{T} \cdot \tau_2$" + "\n"
        rf"$\ \ \ = {phase_2_disp:.1f}^\circ$"
    )

    ax.text(
        1.03,
        0.6,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.8", facecolor="white", edgecolor="gray", linewidth=1.0
        ),
    )

ax.set_xlabel("Time [mRNA lifetimes]", fontsize=16)
ax.set_ylabel(r"Protein Concentration [$K_M$]", fontsize=16)

ax.set_xlim([window_start, window_end])
ax.set_ylim([0, 5])

ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_linewidth(1.0)

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

plt.tight_layout()
plt.show()
