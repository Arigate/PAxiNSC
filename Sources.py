from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm

from NSC_cosmo import background, PertEMD_noAX, primordial_phi_ini, gstar, gstars
from ALP_NSC import NSC_PERT_axi as ALP_PERT
from QCD_axions import NSC_PERT_axi as QCD_PERT

plt.rcParams.update({
    "pdf.fonttype": 42,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

output_dir = Path("Plots/Sources")

T_RH = 0.100
Tini = 30.0
ep = 1e5
maxi = 70.0

theta_ini = 1.0
dtheta_ini = 0.0
kappas = np.array([100.0, 10.0, 1.0])
cpu_fraction = 0.22

alp_ma0_ev = 1e-8
Tl = 0.04
b = 4.0
smooth = 15.0
points_per_cycle = 300
max_substeps = 1_000_000


def smooth_log(x, y, window=3000, order=3, dens=4):
    lx = np.log(x)
    lx_u = np.linspace(lx[0], lx[-1], len(x) * dens)
    y_u = interp1d(lx, y, kind="linear")(lx_u)

    window = min(window, len(y_u) if len(y_u) % 2 else len(y_u) - 1)
    if window <= order:
        return np.asarray(y)

    y_s = savgol_filter(y_u, window, order)
    return interp1d(lx_u, y_s, kind="cubic")(lx)


def get_background():
    R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR, R_RH, k_RH = background(T_RH, Tini, maxi, ep)

    with np.errstate(divide="ignore", invalid="ignore"):
        c_r2 = dP_r_dR / drho_r_dR
        omega_r = P_r / rho_r

    return {
        "R": R,
        "rho_phi": rho_phi,
        "rho_r": rho_r,
        "Temp": Temp,
        "Hub": Hub,
        "GG": GG,
        "c_r2": c_r2,
        "omega_r": omega_r,
        "R_RH": R_RH,
        "k_RH": k_RH,
        "H_RH": k_RH / R_RH,
        "x": R / R_RH,
    }


def noax_mode_solver(args, rho_phi, rho_r, c_r2, omega_r, GG, R):
    k, Phi_ini = args
    delta_r, _, Phi, _, _, dPhi_dR = PertEMD_noAX(rho_phi, rho_r, c_r2, omega_r, GG, R, k, Phi_ini)
    return delta_r, Phi, dPhi_dR


def qcd_mode_worker(task, rho_phi, rho_r, Hub, Temp, R, ma0, fa):
    kappa, k, delta_r, Phi, dPhi_dR = task
    out = QCD_PERT(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, ma0, fa)
    R_osc, R_l = float(out[4][5]), float(out[4][6])
    return kappa, np.asarray(out[9], dtype=float), R_osc, R_l


def alp_mode_worker(task, rho_phi, rho_r, Hub, Temp, R, R_RH, ma0):
    kappa, k, delta_r, Phi, dPhi_dR = task
    out = ALP_PERT(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, R_RH, ma0, Tl, fa_input=None, b_val=b, smooth=smooth, points_per_cycle=points_per_cycle, max_substeps=max_substeps)
    R_osc, R_l = float(out[4][5]), float(out[4][6])
    return kappa, np.asarray(out[9], dtype=float), R_osc, R_l


def run_noax(bg, cores):
    Phi_ini = primordial_phi_ini(kappas, T_RH, bg["H_RH"], float(gstars(T_RH)))
    k_values = kappas * bg["k_RH"]
    task = partial(noax_mode_solver, rho_phi=bg["rho_phi"], rho_r=bg["rho_r"], c_r2=bg["c_r2"], omega_r=bg["omega_r"], GG=bg["GG"], R=bg["R"])

    with Pool(cores) as pool:
        results = list(tqdm(pool.imap(task, zip(k_values, Phi_ini), chunksize=1), total=len(kappas), desc="No-axion modes"))

    return k_values, results


def collect_sources(worker, tasks, cores, label):
    sources = {}
    markers = {}

    with Pool(cores) as pool:
        for kappa, S, R_osc, R_l in tqdm(pool.imap(worker, tasks, chunksize=1), total=len(tasks), desc=label):
            if S.ndim != 2:
                raise RuntimeError(f"Unexpected source array shape for kappa={kappa:g}: {S.shape}")
            if S.shape[0] != 3 and S.shape[1] == 3:
                S = S.T
            if S.shape[0] != 3:
                raise RuntimeError(f"Expected three sources for kappa={kappa:g}, got shape {S.shape}")

            sources[float(kappa)] = S
            markers[float(kappa)] = (R_osc, R_l)

    return sources, markers


def plot_sources(result, model, ma0_ev, x_end=None):
    R = result["R"]
    x = result["x"]
    Temp = result["Temp"]
    R_RH = result["R_RH"]
    sources = result["sources"]
    markers = result["markers"]
    colors = ["#4477AA", "#EE6677", "#228833"]

    if x_end is None:
        if model == "QCD":
            idx = np.argsort(Temp)
            x_end = 1.5 * float(np.interp(0.0101, Temp[idx], x[idx]))
        else:
            x_end = float(x[-1])

    fig, axes = plt.subplots(1, len(kappas), figsize=(17, 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for j, (ax, kappa) in enumerate(zip(axes, kappas)):
        S = sources[float(kappa)]
        R_osc, R_l = markers[float(kappa)]
        x_osc, x_l = R_osc / R_RH, R_l / R_RH

        S1 = smooth_log(R, np.abs(S[0]))
        S2 = smooth_log(R, np.abs(S[1]))
        S3 = smooth_log(R, np.abs(S[2]))

        ax.plot(x, S1 * R_RH**2, color=colors[0], lw=3, label=r"$|S_1|$", zorder=3)
        ax.plot(x, S2 * R_RH**2, color=colors[1], lw=3, label=r"$|S_2|$", zorder=4)
        ax.plot(x, S3 * R_RH**2, color=colors[2], lw=3, label=r"$|S_3|$", zorder=2)

        ax.axvline(x_osc, color="black", ls="-.", alpha=0.8, zorder=5)
        ax.axvline(x_l, color="black", ls="--", alpha=0.8, zorder=5)
        ax.axvline(1.0, color="red", ls=":", zorder=5)

        trans = ax.get_xaxis_transform()
        ax.text(x_osc * 0.8, 0.05, r"$x_{\mathrm{osc}}$", transform=trans, rotation=90, va="bottom", ha="right")
        ax.text(x_l * 0.8, 0.05, r"$x_{\Lambda}$", transform=trans, rotation=90, va="bottom", ha="right")
        ax.text(1.1, 0.05, r"$x_{\mathrm{RH}}$", transform=trans, rotation=90, va="bottom", ha="left")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\kappa={kappa:g}$")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(1e-3, x_end)
        ax.set_ylim(1e-11, 1e4)

        if j == 0:
            ax.set_ylabel(r"$|S_i(\kappa,x)|$")
            ax.legend(loc="upper right", frameon=True, framealpha=0.8, handlelength=2.5)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    ma_tag = f"{ma0_ev:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    kappa_tag = "-".join(f"{kappa:g}" for kappa in kappas)
    pdf_name = output_dir / f"{model}_sources_ma{ma_tag}eV_{kappa_tag}_average.pdf"
    fig.savefig(pdf_name, dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_name}")
    return pdf_name


def qcd_sources(make_plot=True):
    bg = get_background()
    cores = min(len(kappas), max(1, int((cpu_count() or 1) * cpu_fraction)))

    ma0 = 3.58e-7 * 1e-9 * float(gstar(T_RH))**0.5 / float(gstars(T_RH))**(2.0 / 3.0) * (T_RH / 0.150)**(4.0 / 3.0) * theta_ini**(4.0 / 3.0)
    fa = 5.69e-3 / ma0

    k_values, noax = run_noax(bg, cores)
    tasks = [(float(kappa), float(k_values[i]), noax[i][0], noax[i][1], noax[i][2]) for i, kappa in enumerate(kappas)]
    worker = partial(qcd_mode_worker, rho_phi=bg["rho_phi"], rho_r=bg["rho_r"], Hub=bg["Hub"], Temp=bg["Temp"], R=bg["R"], ma0=ma0, fa=fa)
    sources, markers = collect_sources(worker, tasks, cores, "QCD sources")

    result = {**bg, "sources": sources, "markers": markers, "ma0_ev": ma0 * 1e9, "fa": fa}
    if make_plot:
        result["pdf"] = plot_sources(result, "QCD", result["ma0_ev"])

    print(f"QCD ma0 = {result['ma0_ev']:.6e} eV")
    return result


def alp_sources(ma0_ev=alp_ma0_ev, make_plot=True):
    bg = get_background()
    cores = min(len(kappas), max(1, int((cpu_count() or 1) * cpu_fraction)))
    ma0 = ma0_ev * 1e-9

    k_values, noax = run_noax(bg, cores)
    tasks = [(float(kappa), float(k_values[i]), noax[i][0], noax[i][1], noax[i][2]) for i, kappa in enumerate(kappas)]
    worker = partial(alp_mode_worker, rho_phi=bg["rho_phi"], rho_r=bg["rho_r"], Hub=bg["Hub"], Temp=bg["Temp"], R=bg["R"], R_RH=bg["R_RH"], ma0=ma0)
    sources, markers = collect_sources(worker, tasks, cores, "ALP sources")

    result = {**bg, "sources": sources, "markers": markers, "ma0_ev": ma0_ev, "Tl": Tl, "b": b}
    if make_plot:
        result["pdf"] = plot_sources(result, "ALP", ma0_ev)

    return result


if __name__ == "__main__":
    qcd_sources()
