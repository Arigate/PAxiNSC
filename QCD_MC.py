import gc
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Axi_WKB import integrate_wkb
from NSC_cosmo import background, PertEMD_noAX, primordial_phi_ini, gstar, gstars
from QCD_axions import NSC_PERT_axi

output_root = Path("Sim_Data/QCD_MC")
summary_file = output_root / "qcd_mc.npz"

T_RH_min = 0.005
T_RH_max = 0.300
n_T_RH = 48


theta_ini = 1.1313708512275915
dtheta_ini = 0.0
cpu_fraction = 0.20

Tini = 3e1
ep = 1e5
maxi = 95.0

kappa_min = 1.0
kappa_max = 250.0
n_k_modes = 300

T_chi_end = 0.0101
num_extension = 1.09
x_num_min = 10.0

x_match_wkb = 90.0
n_wkb = 200_000


def get_x_eq(T_RH):
    T_eq = 0.79e-9
    return float((T_RH / T_eq) * (gstars(T_RH) / gstars(T_eq))**(1.0 / 3.0))


def get_jeans_eq(ma0, R_eq):
    T_eq = 0.79e-9
    gstar_eq = 3.36
    Mp = 2.435e18
    rho_rad = gstar_eq * np.pi**2 * T_eq**4 / 30.0
    rho_tot = 2.0 * rho_rad
    return ((2.0 * rho_tot * ma0**2 / Mp**2)**0.25) * R_eq


def qcd_ma0(T_RH):
    return 3.58e-7 * 1e-9 * float(gstar(T_RH))**0.5 / float(gstars(T_RH))**(2.0 / 3.0) * (T_RH / 0.150)**(4.0 / 3.0) * theta_ini**(4.0 / 3.0)


def noax_mode_solver(args, rho_phi, rho_r, c_r2, omega_r, GG, R):
    k, Phi_ini = args
    out = PertEMD_noAX(rho_phi, rho_r, c_r2, omega_r, GG, R, k, Phi_ini)
    return out[0], out[2], out[5]


def axion_mode_worker(args, rho_phi, rho_r, Hub, Temp, R, ma0, fa):
    mode_idx, k, delta_r, Phi, dPhi_dR = args
    out = NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, ma0, fa)

    bg = out[4] if mode_idx == 0 else None
    return mode_idx, float(out[2][-1]), float(out[3][-1]), bg


def run_point(T_RH, kappas, cores):
    ma0 = qcd_ma0(T_RH)
    ma0_ev = ma0 * 1e9
    fa = 5.69e-3 / ma0

    R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR, R_RH, k_RH = background(T_RH, Tini, maxi, ep)
    x = R / R_RH
    H_RH = k_RH / R_RH

    with np.errstate(divide="ignore", invalid="ignore"):
        c_r2 = dP_r_dR / drho_r_dR
        omega_r = P_r / rho_r

    k_values = kappas * k_RH
    Phi_ini = primordial_phi_ini(kappas, T_RH, H_RH, float(gstars(T_RH)))

    noax_task = partial(noax_mode_solver, rho_phi=rho_phi, rho_r=rho_r, c_r2=c_r2, omega_r=omega_r, GG=GG, R=R)
    noax_args = list(zip(k_values, Phi_ini))

    with Pool(cores) as pool:
        noax = list(tqdm(pool.imap(noax_task, noax_args, chunksize=1), total=n_k_modes, desc=f"No-axion T_RH={T_RH * 1e3:.2f} MeV"))

    delta_r = np.asarray([item[0] for item in noax])
    Phi = np.asarray([item[1] for item in noax])
    dPhi_dR = np.asarray([item[2] for item in noax])

    idx = np.argsort(Temp)
    x_chi_end = float(np.interp(T_chi_end, Temp[idx], x[idx]))
    x_num_end = max(x_num_min, num_extension * x_chi_end)
    num_mask = x <= x_num_end

    R_num = R[num_mask]
    rho_phi_num = rho_phi[num_mask]
    rho_r_num = rho_r[num_mask]
    Temp_num = Temp[num_mask]
    Hub_num = Hub[num_mask]

    mode_args = [(i, k_values[i], delta_r[i, num_mask], Phi[i, num_mask], dPhi_dR[i, num_mask]) for i in range(n_k_modes)]
    axion_task = partial(axion_mode_worker, rho_phi=rho_phi_num, rho_r=rho_r_num, Hub=Hub_num, Temp=Temp_num, R=R_num, ma0=ma0, fa=fa)

    ax1_ic = np.full(n_k_modes, np.nan)
    dax1_ic = np.full(n_k_modes, np.nan)
    bg = None

    with Pool(cores) as pool:
        iterator = pool.imap(axion_task, mode_args, chunksize=1)
        for mode_idx, ax1_end, dax1_end, bg_i in tqdm(iterator, total=n_k_modes, desc=f"QCD T_RH={T_RH * 1e3:.2f} MeV, ma0={ma0_ev:.2e} eV"):
            ax1_ic[mode_idx] = ax1_end
            dax1_ic[mode_idx] = dax1_end
            if bg_i is not None:
                bg = bg_i

    if bg is None:
        raise RuntimeError("No axion background returned")

    ax0, dax0, ma_arr, _, _, R_osc, R_l, _ = bg
    ax0_ic = float(ax0[-1])
    dax0_ic = float(dax0[-1])
    ma_ic = float(ma_arr[-1])
    x_ic = float(R_num[-1] / R_RH)

    R_osc = float(R_osc)
    R_l = float(R_l)
    order = np.argsort(R)
    H_osc = float(np.interp(R_osc, R[order], Hub[order])) if np.isfinite(R_osc) else np.nan
    H_l = float(np.interp(R_l, R[order], Hub[order])) if np.isfinite(R_l) else np.nan

    x_eq = get_x_eq(T_RH)
    R_eq = x_eq * R_RH
    kappa_J_eq = get_jeans_eq(ma0, R_eq) / k_RH
    k_osc = R_osc * H_osc if np.isfinite(H_osc) else np.nan
    k_l = R_l * H_l if np.isfinite(H_l) else np.nan
    kappa_osc = k_osc / k_RH
    kappa_l = k_l / k_RH

    delta_a_eq = integrate_wkb(x, kappas, ax1_ic, dax1_ic, Phi, dPhi_dR, ax0_ic, dax0_ic, ma_ic, Hub, R_RH, x_ic, x_eq, x_match=x_match_wkb, n_points=n_wkb)

    del noax, delta_r, Phi, dPhi_dR, mode_args, bg, R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR
    gc.collect()

    return delta_a_eq, x_eq, x_ic, R_eq, kappa_J_eq, k_osc, k_l, kappa_osc, kappa_l, R_RH, k_RH


def run():
    available = cpu_count() or 1
    cores = max(1, int(available * cpu_fraction))
    T_RH_arr = np.logspace(np.log10(T_RH_min), np.log10(T_RH_max), n_T_RH)
    ma0_arr = np.array([qcd_ma0(T_RH) * 1e9 for T_RH in T_RH_arr])
    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), n_k_modes)

    n_scan = len(T_RH_arr)
    delta_a_eq = np.full((n_scan, n_k_modes), np.nan)
    x_eq = np.full(n_scan, np.nan)
    x_ic = np.full(n_scan, np.nan)
    R_eq = np.full(n_scan, np.nan)
    kappa_J_eq = np.full(n_scan, np.nan)
    k_osc = np.full(n_scan, np.nan)
    k_l = np.full(n_scan, np.nan)
    kappa_osc = np.full(n_scan, np.nan)
    kappa_l = np.full(n_scan, np.nan)
    R_RH = np.full(n_scan, np.nan)
    k_RH = np.full(n_scan, np.nan)

    print(f"CPU cores: {cores}/{available} (~{100 * cpu_fraction:.0f}%)")
    print(f"T_RH grid: {T_RH_min * 1e3:g} - {T_RH_max * 1e3:g} MeV ({n_T_RH} log-spaced points)")
    print(f"kappa grid: {kappa_min:g} - {kappa_max:g} ({n_k_modes} modes)")
    print(f"WKB grid: {n_wkb} points, x_match={x_match_wkb:g}")
    print()

    for i, T_RH in enumerate(T_RH_arr):
        delta_a_eq[i], x_eq[i], x_ic[i], R_eq[i], kappa_J_eq[i], k_osc[i], k_l[i], kappa_osc[i], kappa_l[i], R_RH[i], k_RH[i] = run_point(T_RH, kappas, cores)

    inputs = {
        "T_RH_min": T_RH_min,
        "T_RH_max": T_RH_max,
        "n_T_RH": n_T_RH,
        "theta_ini": theta_ini,
        "dtheta_ini": dtheta_ini,
        "Tini": Tini,
        "ep": ep,
        "maxi": maxi,
        "kappa_min": kappa_min,
        "kappa_max": kappa_max,
        "n_k_modes": n_k_modes,
        "T_chi_end": T_chi_end,
        "num_extension": num_extension,
        "x_num_min": x_num_min,
        "x_match_wkb": x_match_wkb,
        "n_wkb": n_wkb,
        "cpu_fraction": cpu_fraction,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        summary_file,
        T_RH=T_RH_arr,
        ma0_ev=ma0_arr,
        kappas=kappas,
        delta_a_eq=delta_a_eq,
        x_eq=x_eq,
        x_ic=x_ic,
        R_eq=R_eq,
        kappa_J_eq=kappa_J_eq,
        k_osc=k_osc,
        k_l=k_l,
        kappa_osc=kappa_osc,
        kappa_l=kappa_l,
        R_RH=R_RH,
        k_RH=k_RH,
        inputs=inputs,
    )

    print()
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    run()
