import gc
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from NSC_cosmo import background, PertEMD_noAX, primordial_phi_ini, gstars
from ALP_NSC import NSC_PERT_axi

num_cores = 24

T_RH_arr = [0.015]
Tl_arr = [0.025]
ma0_arr = [1e-9]

Tini = 30.0
ep = 1e5
maxi = 11.0

theta_ini = 1.0
dtheta_ini = 0.0
b = 4.0
smooth = 15.0
points_per_cycle = 600
max_substeps = 1_000_000

kappa_min = 0.1
kappa_max = 450.0
n_k_modes = 250
kappa_extra_ma0 = [10.0, 80.0, 100.0]
kappa_extra_TRH = [10.0, 50.0, 100.0]
kappa_custom_max = 400.0
n_k_modes_custom = 400


def get_scan_type(T_RH_arr, Tl_arr, ma0_arr):
    if len(ma0_arr) > 1 and len(T_RH_arr) == len(Tl_arr) == 1:
        return "ma0"
    if len(T_RH_arr) > 1 and len(ma0_arr) == len(Tl_arr) == 1:
        return "T_RH"
    if len(Tl_arr) > 1 and len(T_RH_arr) == len(ma0_arr) == 1:
        return "Tl"
    return "single" if len(T_RH_arr) == len(Tl_arr) == len(ma0_arr) == 1 else "custom"


def get_kappas(scan_type):
    if scan_type in {"ma0", "T_RH"}:
        base = np.logspace(np.log10(kappa_min), np.log10(kappa_max), n_k_modes)
        extra = kappa_extra_ma0 if scan_type == "ma0" else kappa_extra_TRH
        return np.sort(np.unique(np.concatenate([base, extra])))
    return np.logspace(np.log10(kappa_min), np.log10(kappa_custom_max), n_k_modes_custom)


def get_output_root(scan_type):
    names = {"ma0": "Varying_MA0", "T_RH": "Varying_TRH", "Tl": "Varying_Tl", "single": "Single_Run", "custom": "Varying_Custom"}
    return Path("Sim_Data") / names[scan_type]


def get_run_folder(output_root, T_RH, Tl, ma0_ev):
    run_folder = output_root / f"TRH_{T_RH * 1e3:.3f}MeV"
    if len(Tl_arr) > 1:
        run_folder /= f"Tl_{Tl * 1e3:.3f}MeV"
    if len(ma0_arr) > 1:
        run_folder /= f"MA0_{ma0_ev:.2e}eV"
    return run_folder


def noax_mode_solver(args, rho_phi, rho_r, c_r2, omega_r, GG, R):
    k, Phi_ini = args
    return PertEMD_noAX(rho_phi, rho_r, c_r2, omega_r, GG, R, k, Phi_ini)


def axion_mode_worker(args, rho_phi, rho_r, Hub, Temp, R, R_RH, ma0, Tl):
    k, delta_r, Phi, dPhi_dR = args
    return NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, R_RH, ma0, Tl, fa_input=None, b_val=b, smooth=smooth, points_per_cycle=points_per_cycle, max_substeps=max_substeps)


def remove_scratch(paths, folder):
    for path in paths.values():
        if path.exists():
            path.unlink()
    if folder.exists() and not any(folder.iterdir()):
        folder.rmdir()


def run_parameter_sweep():
    scan_type = get_scan_type(T_RH_arr, Tl_arr, ma0_arr)
    kappas = get_kappas(scan_type)
    output_root = get_output_root(scan_type)
    output_root.mkdir(parents=True, exist_ok=True)

    print("====================================================")
    print(f"SCAN       = {scan_type}")
    print(f"OUTPUT     = {output_root}")
    print(f"N_modes    = {len(kappas)}")
    print(f"kappa_min  = {kappas.min():.6g}")
    print(f"kappa_max  = {kappas.max():.6g}")
    print(f"T_RH       = {T_RH_arr}")
    print(f"Tl         = {Tl_arr}")
    print(f"ma0 [eV]   = {ma0_arr}")
    print(f"theta_ini  = {theta_ini}")
    print(f"b          = {b}")
    print(f"smooth     = {smooth}")
    print(f"Tini       = {Tini} GeV")
    print("====================================================")

    for T_RH in T_RH_arr:
        trh_folder = output_root / f"TRH_{T_RH * 1e3:.3f}MeV"
        trh_folder.mkdir(parents=True, exist_ok=True)
        precomputed_file = trh_folder / f"noax_NSC_TRH_{T_RH:.6f}.npz"

        R, rho_phi, rho_r, P_r, Temp, Hub, GG, _, drho_r_dR, dP_r_dR, R_RH, k_RH = background(T_RH, Tini, maxi, ep)

        with np.errstate(divide="ignore", invalid="ignore"):
            omega_r = P_r / rho_r
            c_r2 = dP_r_dR / drho_r_dR

        k_values = kappas * k_RH
        H_RH = k_RH / R_RH
        Phi_ini_values = primordial_phi_ini(kappas, T_RH, H_RH, float(gstars(T_RH)))
        x = R / R_RH
        n_modes, n_times = len(k_values), len(R)

        scratch_dir = trh_folder / ".scratch"
        scratch_dir.mkdir(exist_ok=True)
        noax_paths = {name: scratch_dir / f"{name}.npy" for name in ["delta_r", "delta_phi", "Phi", "theta_phi", "theta_r", "dPhi_dR"]}

        delta_r_noax = np.lib.format.open_memmap(noax_paths["delta_r"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
        delta_phi_noax = np.lib.format.open_memmap(noax_paths["delta_phi"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
        Phi_noax = np.lib.format.open_memmap(noax_paths["Phi"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
        theta_phi_noax = np.lib.format.open_memmap(noax_paths["theta_phi"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
        theta_r_noax = np.lib.format.open_memmap(noax_paths["theta_r"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
        dPhi_dR_noax = np.lib.format.open_memmap(noax_paths["dPhi_dR"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))

        noax_task = partial(noax_mode_solver, rho_phi=rho_phi, rho_r=rho_r, c_r2=c_r2, omega_r=omega_r, GG=GG, R=R)
        with Pool(num_cores) as pool:
            iterator = pool.imap(noax_task, zip(k_values, Phi_ini_values), chunksize=1)
            for i, result in enumerate(tqdm(iterator, total=n_modes, desc=f"No-axion T_RH={T_RH * 1e3:.1f} MeV")):
                delta_r_noax[i], delta_phi_noax[i], Phi_noax[i], theta_phi_noax[i], theta_r_noax[i], dPhi_dR_noax[i] = result

        for arr in [delta_r_noax, delta_phi_noax, Phi_noax, theta_phi_noax, theta_r_noax, dPhi_dR_noax]:
            arr.flush()

        np.savez_compressed(
            precomputed_file,
            background={"R": R, "rho_phi": rho_phi, "rho_r": rho_r, "P_r": P_r, "Hub": Hub, "Temp": Temp, "GG": GG, "c_r2": c_r2, "omega_r": omega_r},
            noax={"delta_r": delta_r_noax, "delta_phi": delta_phi_noax, "Phi": Phi_noax, "theta_phi": theta_phi_noax, "theta_r": theta_r_noax, "dPhi_dR": dPhi_dR_noax},
            horizon={"k_vals": k_values, "kappas": kappas, "x": x, "Phi_ini": Phi_ini_values},
            modes={"R_RH": R_RH, "k_RH": k_RH},
            inputs={"T_RH": T_RH, "Tini": Tini, "EP": ep, "maxi": maxi, "b": b, "smooth": smooth, "scan_type": scan_type},
        )

        del delta_phi_noax, theta_phi_noax, theta_r_noax
        gc.collect()

        Hub_f = interp1d(R, Hub, kind="cubic", bounds_error=False, fill_value=np.nan)

        for ma0_ev in ma0_arr:
            ma0 = ma0_ev * 1e-9

            for Tl in Tl_arr:
                run_folder = get_run_folder(output_root, T_RH, Tl, ma0_ev)
                run_folder.mkdir(parents=True, exist_ok=True)
                tmp_dir = run_folder / ".scratch"
                tmp_dir.mkdir(exist_ok=True)

                ax_paths = {name: tmp_dir / f"{name}.npy" for name in ["delta_a_s", "delta_a_r", "ax1", "dax1", "S"]}
                delta_a_s = np.lib.format.open_memmap(ax_paths["delta_a_s"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
                delta_a_r = np.lib.format.open_memmap(ax_paths["delta_a_r"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
                ax1 = np.lib.format.open_memmap(ax_paths["ax1"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
                dax1 = np.lib.format.open_memmap(ax_paths["dax1"], mode="w+", dtype=np.float64, shape=(n_modes, n_times))
                S = np.lib.format.open_memmap(ax_paths["S"], mode="w+", dtype=np.float64, shape=(n_modes, 3, n_times))

                worker = partial(axion_mode_worker, rho_phi=rho_phi, rho_r=rho_r, Hub=Hub, Temp=Temp, R=R, R_RH=R_RH, ma0=ma0, Tl=Tl)
                mode_args = ((k_values[i], delta_r_noax[i], Phi_noax[i], dPhi_dR_noax[i]) for i in range(n_modes))

                bg_first = None
                with Pool(num_cores) as pool:
                    iterator = pool.imap(worker, mode_args, chunksize=1)
                    for i, result in enumerate(tqdm(iterator, total=n_modes, desc=f"Axion T_RH={T_RH * 1e3:.1f} MeV Tl={Tl * 1e3:.1f} MeV ma0={ma0_ev:.1e} eV")):
                        delta_a_s[i], delta_a_r[i], ax1[i], dax1[i], S[i] = result[0], result[1], result[2], result[3], result[9]
                        if bg_first is None:
                            bg_first = result[4]

                for arr in [delta_a_s, delta_a_r, ax1, dax1, S]:
                    arr.flush()

                axo = np.asarray(bg_first[0]).copy()
                daxo = np.asarray(bg_first[1]).copy()
                ma = np.asarray(bg_first[2]).copy()
                rho_theta = np.asarray(bg_first[3]).copy()
                b_arr = np.asarray(bg_first[4]).copy()
                R_osc = float(bg_first[5])
                R_l = float(bg_first[6])
                fa = float(bg_first[7])

                k_osc = R_osc * float(Hub_f(R_osc)) if np.isfinite(R_osc) else np.nan
                k_l = R_l * float(Hub_f(R_l)) if np.isfinite(R_l) else np.nan
                rho_a = rho_theta * fa**2

                if np.isfinite(R_osc):
                    idx_osc = int(np.argmin(np.abs(R - R_osc)))
                    rho_a_osc, ma_osc = rho_a[idx_osc], ma[idx_osc]
                else:
                    rho_a_osc = ma_osc = np.nan

                payload = {
                    "axiPerts": {
                        "delta_a_s": delta_a_s,
                        "delta_a_r": delta_a_r,
                        "ax1": ax1,
                        "dax1": dax1,
                        "S": S,
                    },
                    "axiBack": {
                        "rho_a": rho_a,
                        "fa": fa,
                        "axo": axo,
                        "daxo": daxo,
                        "ma": ma,
                        "rho_a_osc": rho_a_osc,
                        "ma_osc": ma_osc,
                        "b_arr": b_arr,
                    },
                    "scale_factors": {
                        "R_l": R_l,
                        "R_osc": R_osc,
                    },
                    "inputs": {
                        "MA0": ma0,
                        "T_RH": T_RH,
                        "Tini": Tini,
                        "EP": ep,
                        "maxi": maxi,
                        "Tl": Tl,
                        "b": b,
                        "smooth": smooth,
                        "theta_ini": theta_ini,
                        "x_num_end": x[-1],
                        "k_RH": k_RH,
                        "R_RH": R_RH,
                        "k_osc": k_osc,
                        "k_l": k_l,
                        "N_modes": len(kappas),
                        "kappa_min": kappas.min(),
                        "kappa_max": kappas.max(),
                        "scan_type": scan_type,
                        "precision_points_per_cycle": points_per_cycle,
                        "precision_max_substeps": max_substeps,
                        "savgol_mode": "mirror",
                    },
                }

                output_file = run_folder / f"AX_TRH_{T_RH:.6f}_Tl_{Tl:.6f}_MA0_{ma0_ev:.2e}.npz"
                np.savez_compressed(output_file, **payload)

                del payload, delta_a_s, delta_a_r, ax1, dax1, S, axo, daxo, ma, rho_theta, rho_a, b_arr, bg_first, worker
                gc.collect()
                remove_scratch(ax_paths, tmp_dir)

        del delta_r_noax, Phi_noax, dPhi_dR_noax
        gc.collect()
        remove_scratch(noax_paths, scratch_dir)

        del R, rho_phi, rho_r, P_r, Temp, Hub, GG, drho_r_dR, dP_r_dR, c_r2, omega_r
        gc.collect()


if __name__ == "__main__":
    run_parameter_sweep()
