import gc
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from NSC_cosmo import background, PertEMD_noAX, primordial_phi_ini, gstars
from ALP_NSC import NSC_PERT_axi

output_root = Path("Sim_Data/Jeans_grad")
summary_file = output_root / "Jeans_summary.npz"

theta_ini = 1.0
dtheta_ini = 0.0
b = 4.0
smooth = 15.0
points_per_cycle = 300
max_substeps = 1_000_000
cpu_fraction = 0.22

Tini = 3e1
ep = 1e5
maxi = 10.0
Tl = 0.04

jeans_fac = np.array([1.0, 0.5], dtype=float)
T_arr = np.logspace(np.log10(0.025), np.log10(0.065), 30)
ma0_arr = np.logspace(np.log10(1e-9), np.log10(1e-7), 30)


def noax_mode_solver(args, rho_phi, rho_r, c_r2, omega_r, GG, R):
    k, Phi_ini = args
    out = PertEMD_noAX(rho_phi, rho_r, c_r2, omega_r, GG, R, k, Phi_ini)
    return out[0], out[2], out[5]


def axion_mode_worker(task, rho_phi, rho_r, Hub, Temp, R, R_RH):
    mass_idx, fac_idx, ma0_gev, k, delta_r, Phi, dPhi_dR = task
    out = NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, R_RH, ma0_gev, Tl, fa_input=None, b_val=b, smooth=smooth, points_per_cycle=points_per_cycle, max_substeps=max_substeps)
    return mass_idx, fac_idx, float(np.abs(out[0][-1]))


def run_temperature(T_RH, cores):
    R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR, R_RH, k_RH = background(T_RH, Tini, maxi, ep)

    with np.errstate(divide="ignore", invalid="ignore"):
        omega_r = P_r / rho_r
        c_r2 = dP_r_dR / drho_r_dR

    H_RH = k_RH / R_RH
    sort_idx = np.argsort(Temp)
    R_l = float(np.interp(Tl, Temp[sort_idx], R[sort_idx]))

    ma0_gev = ma0_arr * 1e-9
    k_J = np.sqrt(ma0_gev / H_RH) * (R_l / R_RH)**0.25 * k_RH

    target_k = []
    target_meta = []

    for mass_idx, (mass, k_J_i) in enumerate(zip(ma0_gev, k_J)):
        for fac_idx, fac in enumerate(jeans_fac):
            target_k.append(fac * k_J_i)
            target_meta.append((mass_idx, fac_idx, mass))

    target_k = np.asarray(target_k, dtype=float)
    target_kappa = target_k / k_RH
    Phi_ini = primordial_phi_ini(target_kappa, T_RH, H_RH, float(gstars(T_RH)))

    noax_task = partial(noax_mode_solver, rho_phi=rho_phi, rho_r=rho_r, c_r2=c_r2, omega_r=omega_r, GG=GG, R=R)
    noax_args = list(zip(target_k, Phi_ini))

    with Pool(cores) as pool:
        noax_results = list(tqdm(pool.imap(noax_task, noax_args, chunksize=1), total=len(noax_args), desc=f"No-axion T_RH={T_RH * 1e3:.2f} MeV"))

    axion_tasks = []
    for i, (mass_idx, fac_idx, mass) in enumerate(target_meta):
        delta_r, Phi, dPhi_dR = noax_results[i]
        axion_tasks.append((mass_idx, fac_idx, mass, target_k[i], delta_r, Phi, dPhi_dR))

    axion_task = partial(axion_mode_worker, rho_phi=rho_phi, rho_r=rho_r, Hub=Hub, Temp=Temp, R=R, R_RH=R_RH)
    delta_final = np.full((len(ma0_arr), len(jeans_fac)), np.nan, dtype=float)

    with Pool(cores) as pool:
        iterator = pool.imap(axion_task, axion_tasks, chunksize=1)
        for mass_idx, fac_idx, delta_a in tqdm(iterator, total=len(axion_tasks), desc=f"Axion T_RH={T_RH * 1e3:.2f} MeV"):
            delta_final[mass_idx, fac_idx] = delta_a

    del noax_results, axion_tasks, R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR
    gc.collect()
    return delta_final


def run_simulations():
    available = cpu_count() or 1
    cores = max(1, int(available * cpu_fraction))
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"CPU cores available : {available}")
    print(f"CPU cores used      : {cores} (~{100 * cpu_fraction:.0f}%)")
    print(f"Output              : {summary_file}")
    print(f"grid                : {len(T_arr)} temperatures x {len(ma0_arr)} masses")
    print(f"modes per mass      : {jeans_fac}")
    print()

    delta_a = np.full((len(ma0_arr), len(T_arr), len(jeans_fac)), np.nan, dtype=float)

    for T_idx, T_RH in enumerate(T_arr):
        delta_a[:, T_idx, :] = run_temperature(T_RH, cores)

    inputs = {
        "Tini": Tini,
        "ep": ep,
        "maxi": maxi,
        "Tl": Tl,
        "theta_ini": theta_ini,
        "dtheta_ini": dtheta_ini,
        "b": b,
        "smooth": smooth,
        "points_per_cycle": points_per_cycle,
        "max_substeps": max_substeps,
        "cpu_fraction": cpu_fraction,
    }

    np.savez_compressed(
        summary_file,
        T_RH=T_arr,
        ma0_ev=ma0_arr,
        jeans_fac=jeans_fac,
        delta_a_abs=delta_a,
        inputs=inputs,
    )

    print()
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    run_simulations()
