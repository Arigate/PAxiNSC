import os
import gc
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from NSC_cosmo import background, PertEMD_noAX
from NSC_axions import NSC_PERT_axi

def noax_mode_solver(k, rho_phi, rho_r, GG, R):
    delta_r, delta_phi, Phi, theta_phi, theta_r, dPhi_dR = PertEMD_noAX(rho_phi, rho_r, GG, R, k)
    return delta_r, delta_phi, Phi, theta_phi, theta_r, dPhi_dR

def axion_mode_worker(
    args,
    rho_phi,
    rho_r,
    Hub,
    Temp,
    R,
    theta_ini,
    dtheta_ini,
    R_RH,
    MA0,
    Tl,
    fa,
    b
):
    k, delta_r_in, Phi_in, dPhi_dR_in = args
    out = NSC_PERT_axi(
        rho_phi,
        rho_r,
        Hub,
        Temp,
        R,
        k,
        theta_ini,
        dtheta_ini,
        delta_r_in,
        Phi_in,
        dPhi_dR_in,
        R_RH,
        MA0,
        Tl,
        fa,
        b
    )
    return out

def save_precomputed_base(path, background_data, noax_data, horizon_info, mode_info, input_info):
    payload = {
        "background": background_data,
        "noax": noax_data,
        "horizon": horizon_info,
        "modes": mode_info,
        "inputs": input_info,
    }
    np.savez_compressed(path, **payload)

def run_parameter_sweep(Tends, Tlambdas, MA0_eVs, num_cores):
    if len(Tlambdas) > 1:
        root_output_folder = "Varying_Tl"
    elif len(Tends) > 1:
        root_output_folder = "Varying_Tend"
    elif len(MA0_eVs) > 1:
        root_output_folder = "Varying_MA0"
    else:
        root_output_folder = "Single_Run"

    os.makedirs(root_output_folder, exist_ok=True)

    theta0 = 1
    dtheta0 = 0.0
    fa = None
    b = 4
    Tini = 3e1
    kappa = 1e5
    maxi = 10
    n_k_modes = 25

    for Tend in Tends:
        output_folder = os.path.join(root_output_folder, f"TRH_{Tend*1e3:.3f}MeV")
        os.makedirs(output_folder, exist_ok=True)
        
        precomputed_file = os.path.join(output_folder, f"noax_background_Tend_{Tend:.6f}.npz")

        R, rho_phi, rho_r, Temp, Hub, GG, R_RH, k_RH = background(Tend, Tini, maxi, kappa)
        H_RH = Hub[np.argmin(np.abs(R_RH - R))]

        k_values = np.logspace(np.log10(0.1 * k_RH), np.log10(250 * k_RH), n_k_modes)
        k_values = np.sort(np.append(k_values, [10.0 * k_RH, 50.0 * k_RH, 100.0 * k_RH]))

        noax_task = partial(noax_mode_solver, rho_phi=rho_phi, rho_r=rho_r, GG=GG, R=R)
        with Pool(num_cores) as pool:
            noax_results = list(tqdm(pool.imap(noax_task, k_values), total=len(k_values), desc=f"No-axion modes Tend={Tend:.3f}"))

        delta_r_noax, delta_phi_noax, Phi_noax, _, _, dPhi_dR_noax = map(list, zip(*noax_results))
        del noax_results
        gc.collect()

        save_precomputed_base(
            precomputed_file,
            background_data={"R": R, "rho_phi": rho_phi, "rho_r": rho_r, "Hub": Hub, "Temp": Temp, "GG": GG},
            noax_data={
                "delta_r": np.array(delta_r_noax),
                "delta_phi": np.array(delta_phi_noax),
                "Phi": np.array(Phi_noax),
                "dPhi_dR": np.array(dPhi_dR_noax),
            },
            horizon_info={"k_vals": k_values},
            mode_info={"R_RH": R_RH, "k_RH": k_RH},
            input_info={"Tend": Tend, "Tini": Tini, "kappa": kappa, "maxi": maxi, "b": b},
        )

        gc.collect()

        mode_args = list(zip(k_values, delta_r_noax, Phi_noax, dPhi_dR_noax))

        for MA0_eV in MA0_eVs:
            MA0 = MA0_eV * 1e-9
            for Tl in Tlambdas:
                with Pool(num_cores) as pool:
                    worker = partial(
                        axion_mode_worker,
                        rho_phi=rho_phi,
                        rho_r=rho_r,
                        Hub=Hub,
                        Temp=Temp,
                        R=R,
                        theta_ini=theta0,
                        dtheta_ini=dtheta0,
                        R_RH=R_RH,
                        MA0=MA0,
                        Tl=Tl,
                        fa=fa,
                        b=b
                    )

                    axion_results = list(tqdm(pool.imap(worker, mode_args), total=len(k_values), desc=f"Tend {Tend:.3f} Tl {Tl:.3f} MA0 {MA0_eV:.1e}"))

                    delta_a_s = [r[0] for r in axion_results]
                    delta_a_r = [r[1] for r in axion_results]
                    ax1 = [r[2] for r in axion_results]
                    dax1 = [r[3] for r in axion_results]
                    
                    bg_first = axion_results[0][4]
                    axo = np.array(bg_first[0])
                    daxo = np.array(bg_first[1])
                    ma = np.array(bg_first[2])
                    rho_theta = np.array(bg_first[3])
                    R_osc = bg_first[5]
                    R_l = bg_first[6]
                    fa_computed = bg_first[7]

                    rho_a = rho_theta * (fa_computed**2)
                    rho_a_osc = rho_a[np.argmin(np.abs(R - R_osc))]
                    ma_osc = ma[np.argmin(np.abs(R - R_osc))]
                    k_J_val = np.sqrt(MA0 / H_RH) * (R_l / R_RH)**0.25 * k_RH
                    
                    payload = {
                        "axiPerts": {
                            "delta_a_s": delta_a_s,
                            "delta_a_r": delta_a_r,
                            "ax1": ax1,
                            "dax1": dax1,
                        },
                        "axiBack": {
                            "rho_a": rho_a,
                            "fa": fa_computed,
                            "axo": axo,
                            "daxo": daxo,
                            "ma": ma,
                            "rho_a_osc": rho_a_osc,
                            "ma_osc": ma_osc,
                        },
                        "scale_factors": {
                            "R_l": R_l,
                            "R_osc": R_osc,
                        },
                        "inputs": {
                            "MA0": MA0,
                            "k_J": k_J_val,
                            "Tend": Tend,
                            "Tini": Tini,
                            "kappa": kappa,
                            "maxi": maxi,
                            "Tl": Tl,
                            "b": b,
                        },
                    }

                    output_file = os.path.join(output_folder, f"axion_Tend_{Tend:.6f}_Tl_{Tl:.6f}_MA0_{MA0_eV:.2e}.npz")
                    np.savez_compressed(output_file, **payload)

                    del axion_results, delta_a_s, delta_a_r, ax1, dax1, bg_first, axo, daxo, ma, rho_theta, rho_a, worker
                    gc.collect()

def run_simulations():
    num_cores = int(cpu_count() * 0.8)
    
    Tends = [0.005, 0.02, 0.05]
    Tlambdas = [0.025]
    MA0_eVs = [1e-10]
    
    run_parameter_sweep(Tends, Tlambdas, MA0_eVs, num_cores)

if __name__ == "__main__":
    run_simulations()