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

# on off
force_lacI = True
force_tetR = False
force_cI = False


def alpha_forced(t, T, gene_forced=True):
    if gene_forced:
        return alpha_base * (1 + A * np.cos(2 * np.pi * t / T))
    else:
        return alpha_base


def repressilator_odes_forced(y, t, T):
    m_lacI, m_tetR, m_cI, p_lacI, p_tetR, p_cI = y

    alpha_lacI = alpha_forced(t, T, force_lacI)
    alpha_tetR = alpha_forced(t, T, force_tetR)
    alpha_cI = alpha_forced(t, T, force_cI)

    # mRNA
    dm_lacI_dt = -m_lacI + alpha_lacI / (1 + p_cI**n) + alpha_0
    dm_tetR_dt = -m_tetR + alpha_tetR / (1 + p_lacI**n) + alpha_0
    dm_cI_dt = -m_cI + alpha_cI / (1 + p_tetR**n) + alpha_0

    # protein
    dp_lacI_dt = -beta * (p_lacI - m_lacI)
    dp_tetR_dt = -beta * (p_tetR - m_tetR)
    dp_cI_dt = -beta * (p_cI - m_cI)

    return [dm_lacI_dt, dm_tetR_dt, dm_cI_dt, dp_lacI_dt, dp_tetR_dt, dp_cI_dt]


# simulation settings
t_max = 300
t = np.linspace(0, t_max, 50000)
y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]
steady_idx = int(0.6 * len(t))

# sweep parameters
forcing_periods = np.linspace(3, 10, 80)
response_amplitudes = []
response_periods = []
natural_period = 5.92

for T_val in forcing_periods:
    solution = odeint(repressilator_odes_forced, y0, t, args=(T_val,))

    # extract steady state
    p_lacI_steady = solution[steady_idx:, 3]
    t_steady = t[steady_idx:]

    peaks, _ = find_peaks(p_lacI_steady)
    troughs, _ = find_peaks(-p_lacI_steady)

    if len(peaks) > 1 and len(troughs) > 1:
        # amplitude
        mean_peak = np.mean(p_lacI_steady[peaks])
        mean_trough = np.mean(p_lacI_steady[troughs])
        amp = (mean_peak - mean_trough) / 2
        response_amplitudes.append(amp)

        # period
        peak_times = t_steady[peaks]
        intervals = np.diff(peak_times)
        obs_period = np.mean(intervals)
        response_periods.append(obs_period)
    else:
        response_amplitudes.append(np.nan)
        response_periods.append(np.nan)

# fig
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# plot 1: Period
ax1.axhline(
    y=natural_period,
    color="gray",
    linestyle=":",
    linewidth=1.5,
    label="Natural Period $T_0$",
    alpha=0.6,
)

ax1.plot(
    forcing_periods,
    response_periods,
    "o-",
    color="#0066CC",
    linewidth=1.5,
    markersize=4,
    label="Period $T_{LacI}$",
)

ax1.set_ylabel("Period $T_{LacI} [mRNA lifetimes]$", fontsize=16)
ax1.legend(
    loc="upper left",
    frameon=True,
    fancybox=True,
    fontsize=13,
    framealpha=0.95,
    edgecolor="gray",
    borderpad=0.8,
)
ax1.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)

# plot 2: Amplitude
ax2.axvline(
    x=natural_period,
    color="gray",
    linestyle=":",
    linewidth=1.5,
    label="Natural Period $T_0$",
    alpha=0.6,
)
ax2.plot(
    forcing_periods,
    response_amplitudes,
    "o-",
    color="#DC143C",
    linewidth=1.5,
    markersize=4,
    label="Amplitude $A_{LacI}$",
)

ax2.set_ylabel("Amplitude $A_{LacI}$[$K_M$]", fontsize=16)
ax2.set_xlabel("Forcing Period $T$ [mRNA lifetimes]", fontsize=16)

ax2.legend(
    loc="upper right",
    frameon=True,
    fancybox=True,
    fontsize=13,
    framealpha=0.95,
    edgecolor="gray",
    borderpad=0.8,
)
ax2.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)

for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=12)

print("Sweep complete.")

plt.tight_layout()
plt.show()
