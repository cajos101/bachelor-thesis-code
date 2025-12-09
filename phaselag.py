import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.signal import hilbert

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
alpha_base = 10

# forcing Amplitude
A = 0.5

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


# initial conditions
y0 = [0.0, 0.0, 0.0, 1.0, 0.5, 0.1]

# range
T_values = np.linspace(3, 10, 80)
omega_values = 2 * np.pi / T_values

phase_diff = np.zeros(len(T_values))

steady_start_time = 49

# simulation
for i, T in enumerate(T_values):
    t = np.linspace(0, 300, 6000)

    # ODE
    solution = odeint(repressilator_odes_forced, y0, t, args=(T,))

    mask = t >= steady_start_time
    t_steady = t[mask]

    p_lacI_steady = solution[mask, 3]
    p_lacI_centered = p_lacI_steady - np.mean(p_lacI_steady)

    forcing_steady = np.array([A * np.cos(2 * np.pi * tt / T) for tt in t_steady])

    analytic_signal_protein = hilbert(p_lacI_centered)
    analytic_signal_forcing = hilbert(forcing_steady)

    phase_protein = np.angle(analytic_signal_protein)
    phase_forcing = np.angle(analytic_signal_forcing)

    phase_difference = phase_protein - phase_forcing

    phase_difference = np.unwrap(phase_difference)
    phase_diff[i] = np.mean(phase_difference)

phase_diff_deg = np.rad2deg(phase_diff)
phase_diff_deg = np.mod(phase_diff_deg + 180, 360) - 180

# fig
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(
    T_values,
    phase_diff_deg,
    "o-",
    color="#0E6BCE",
    linewidth=2,
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=2,
)

ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)

ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)

ax.set_xlabel("Forcing Period T [mRNA lifetimes]", fontsize=16)
ax.set_ylabel(" Phase Difference $\\phi$ [degrees]", fontsize=16)

ax.set_xlim([T_values[0], T_values[-1]])
ax.set_ylim([-180, 180])

ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_linewidth(1.0)

ax.tick_params(axis="both", which="major", labelsize=12)

plt.tight_layout()
plt.show()
