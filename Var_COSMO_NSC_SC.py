import gc
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.optimize import root_scalar
from tqdm import tqdm

from NSC_cosmo import background, PertEMD_noAX, primordial_phi_ini, gstars
from ALP_NSC import NSC_PERT_axi

summary_file = Path("Sim_Data/Cosmos/V_Cosmo.npz")

b_values = np.array([0.0, 4.0])
cosmo_names = np.array(["SC", "NSC"])

theta_ini = 1.0
dtheta_ini = 0.0
smooth = 15.0
points_per_cycle = 300
max_substeps = 1_000_000
cpu_fraction = 0.22

T_RH = 0.015
T_l = 0.025
ma0_ev = 1e-8
ma0 = ma0_ev * 1e-9
Tini_nsc = 3e1
ep_nsc = 1e5
ep_sc = 1e-50
maxi = 12.0
n_k_modes = 400
kappa_min = 0.1
kappa_max = 300.0
x_target = 10.0


def jeans_k(ma0, H_RH, R_l, R_RH, k_RH):
    return np.sqrt(ma0 / H_RH) * (R_l / R_RH)**0.25 * k_RH


def noax_mode_solver(args, rho_phi, rho_r, c_r2, omega_r, GG, R):
    k, Phi_ini = args
    out = PertEMD_noAX(rho_phi, rho_r, c_r2, omega_r, GG, R, k, Phi_ini)
    return out[0], out[2], out[5]


def axion_mode_worker(args, rho_phi, rho_r, Hub, Temp, R, R_RH, ma0, T_l, b, x_idx):
    mode_idx, k, delta_r, Phi, dPhi_dR = args
    out = NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, R_RH, ma0, T_l, fa_input=None, b_val=b, smooth=smooth, points_per_cycle=points_per_cycle, max_substeps=max_substeps)
    bg = out[4] if mode_idx == 0 else None
    return mode_idx, float(np.abs(out[0][x_idx])), bg


def get_matched_sc_Tini(target_R_RH):
    def residual(Tini_guess):
        return background(T_RH, Tini_guess, 10.0, ep_sc)[10] - target_R_RH

    Tini_approx = T_RH * target_R_RH
    solution = root_scalar(residual, bracket=[Tini_approx * 1e-4, Tini_approx * 1e4], method="brentq", xtol=1e-5)
    return float(solution.root)


def run_configuration(cosmo_name, b, kappa, Tini, cores):
    R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR, R_RH, k_RH = background(T_RH, Tini, maxi, kappa)

    with np.errstate(divide="ignore", invalid="ignore"):
        omega_r = P_r / rho_r
        c_r2 = dP_r_dR / drho_r_dR

    H_RH = k_RH / R_RH
    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), n_k_modes)
    k_values = kappas * k_RH
    Phi_ini = primordial_phi_ini(kappas, T_RH, H_RH, float(gstars(T_RH)))

    if cosmo_name == "SC":
        Phi_ini *= 10.0 / 9.0

    x = R / R_RH
    x_idx = int(np.argmin(np.abs(x - x_target)))
    x_spec = float(x[x_idx])

    if x[-1] <= x_target:
        raise RuntimeError(f"Integration endpoint x_max={x[-1]:.6g} does not extend beyond x={x_target:g}.")

    noax_task = partial(noax_mode_solver, rho_phi=rho_phi, rho_r=rho_r, c_r2=c_r2, omega_r=omega_r, GG=GG, R=R)
    noax_args = list(zip(k_values, Phi_ini))

    with Pool(cores) as pool:
        noax_results = list(tqdm(pool.imap(noax_task, noax_args, chunksize=1), total=n_k_modes, desc=f"No-axion ({cosmo_name}, b={b:g})"))

    mode_args = [(i, k_values[i], noax_results[i][0], noax_results[i][1], noax_results[i][2]) for i in range(n_k_modes)]
    worker = partial(axion_mode_worker, rho_phi=rho_phi, rho_r=rho_r, Hub=Hub, Temp=Temp, R=R, R_RH=R_RH, ma0=ma0, T_l=T_l, b=b, x_idx=x_idx)

    delta_a = np.full(n_k_modes, np.nan)
    bg = None

    with Pool(cores) as pool:
        iterator = pool.imap(worker, mode_args, chunksize=1)
        for mode_idx, delta_i, bg_i in tqdm(iterator, total=n_k_modes, desc=f"Axion ({cosmo_name}, b={b:g})"):
            delta_a[mode_idx] = delta_i
            if bg_i is not None:
                bg = bg_i

    if bg is None:
        raise RuntimeError(f"No axion background returned for {cosmo_name}, b={b:g}")

    R_osc = float(bg[5])
    R_l = float(bg[6])
    kappa_J = jeans_k(ma0, H_RH, R_l, R_RH, k_RH) / k_RH

    if np.isfinite(R_osc):
        idx = np.argsort(R)
        H_osc = float(np.interp(R_osc, R[idx], Hub[idx]))
        kappa_osc = R_osc * H_osc / k_RH
    else:
        kappa_osc = np.nan

    del noax_results, mode_args, R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR, bg
    gc.collect()

    return delta_a, kappa_J, kappa_osc, x_spec


def run():
    available = cpu_count() or 1
    cores = max(1, int(available * cpu_fraction))

    emd_bg = background(T_RH, Tini_nsc, 10.0, ep_nsc)
    R_RH_nsc = float(emd_bg[10])
    Tini_sc = get_matched_sc_Tini(R_RH_nsc)

    sc_bg = background(T_RH, Tini_sc, 10.0, ep_sc)
    R_RH_sc = float(sc_bg[10])
    rel_match = abs(R_RH_sc / R_RH_nsc - 1.0)

    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), n_k_modes)
    delta_a = np.full((len(b_values), len(cosmo_names), n_k_modes), np.nan)
    kappa_J = np.full((len(b_values), len(cosmo_names)), np.nan)
    kappa_osc = np.full((len(b_values), len(cosmo_names)), np.nan)
    x_spec = np.full((len(b_values), len(cosmo_names)), np.nan)

    print(f"CPU cores: {cores}/{available} (~{100 * cpu_fraction:.0f}%)")
    print(f"Matched Tini_SC = {Tini_sc:.8e} GeV")
    print(f"R_RH relative mismatch = {rel_match:.3e}")
    print()

    cosmologies = [("SC", ep_sc, Tini_sc), ("NSC", ep_nsc, Tini_nsc)]

    for b_idx, b in enumerate(b_values):
        for cosmo_idx, (name, ep, Tini) in enumerate(cosmologies):
            delta_a[b_idx, cosmo_idx], kappa_J[b_idx, cosmo_idx], kappa_osc[b_idx, cosmo_idx], x_spec[b_idx, cosmo_idx] = run_configuration(name, b, ep, Tini, cores)

    inputs = {
        "T_RH": T_RH,
        "T_l": T_l,
        "ma0_ev": ma0_ev,
        "Tini_nsc": Tini_nsc,
        "Tini_sc": Tini_sc,
        "ep_nsc": ep_nsc,
        "ep_sc": ep_sc,
        "maxi": maxi,
        "theta_ini": theta_ini,
        "dtheta_ini": dtheta_ini,
        "smooth": smooth,
        "points_per_cycle": points_per_cycle,
        "max_substeps": max_substeps,
        "cpu_fraction": cpu_fraction,
        "x_target": x_target,
        "R_RH_nsc": R_RH_nsc,
        "R_RH_sc": R_RH_sc,
        "R_RH_rel_match": rel_match,
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(summary_file, b_values=b_values, cosmo_names=cosmo_names, kappas=kappas, delta_a_abs=delta_a, kappa_J=kappa_J, kappa_osc=kappa_osc, x_spec=x_spec, inputs=inputs)
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    run()
