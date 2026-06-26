import os
import gc
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy.integrate import cumulative_trapezoid
import numpy as np
from tqdm import tqdm
from NSC_cosmo import background, PertEMD_noAX
from QCD_axions import NSC_PERT_axi
from scipy.interpolate import interp1d

data = np.loadtxt("Data/geffcbest.txt", skiprows=0)
T_tab = data[:, 0]
gstar = interp1d(T_tab, data[:, 1], bounds_error=False, fill_value="extrapolate")

data = np.loadtxt("Data/heffcbest.txt", skiprows=0)
T_tab = data[:, 0]
gstars = interp1d(T_tab, data[:, 1], bounds_error=False, fill_value="extrapolate")

def get_x_eq(Tend_mev):
    Teq_ev = 0.79
    Teq_mev = Teq_ev * 1e-6
    Teq_GEV = Teq_mev * 1e-3
    Tend_GEV = Tend_mev * 1e-3
    gs_end = gstars(Tend_GEV)
    gs_eq = gstars(Teq_GEV)
    return float((Tend_GEV / Teq_GEV) * (gs_end / gs_eq)**(1.0 / 3.0))

def get_jeans(ma0, R_eq):
    Teq_ev = 0.79
    Teq_gev = Teq_ev * 1e-9
    g_star_eq = 3.36
    Mp = 2.435e18
    rho_rad = g_star_eq * (np.pi**2 / 30.0) * (Teq_gev**4)
    rho_tot = 2.0 * rho_rad
    kj_phys = ((2/Mp**2)* rho_tot * ma0**2 )**0.25
    kj_comoving = kj_phys * R_eq
    return kj_comoving

def integrate_axion_bruteforce_live(x_full, kappa_arr, ax1_ic_vals, dax1_ic_vals, Phi_full, dPhi_dR_full, axo_ic, daxo_ic, ma_ic, Hub_full, R_RH, x_ic, x_end, N_ext=2000000):
    idx_ic_full = np.argmin(np.abs(x_full - x_ic))
    hub_ic = Hub_full[idx_ic_full]
    
    mu = ma_ic / (hub_ic * x_ic**2)
    M_ic = mu * x_ic
    omega_ic = 0.5 * mu * x_ic**2
    
    theta0_ic = axo_ic
    dtheta0_dx_ic = daxo_ic * R_RH
    psi0_ic = (x_ic * np.sqrt(2.0 * M_ic) / 2.0) * (theta0_ic + 1j * (dtheta0_dx_ic / M_ic)) * np.exp(1j * omega_ic)
    
    x_match_phi = 90.0
    idx_m_phi = np.argmin(np.abs(x_full - x_match_phi))
    x_m_val = x_full[idx_m_phi]
    
    x_ext = np.logspace(np.log10(x_ic), np.log10(x_end), N_ext)
    
    mask_num = x_ext < x_m_val
    mask_ana = x_ext >= x_m_val

    delta_wkb_end = np.zeros(len(kappa_arr))

    for j, k_idx in enumerate(range(len(kappa_arr))):
        kappa_val = kappa_arr[k_idx]
        
        theta1_ic = ax1_ic_vals[k_idx]
        dtheta1_dx_ic = dax1_ic_vals[k_idx] * R_RH
        
        phi_num = Phi_full[k_idx, :]
        dphi_dx_num = dPhi_dR_full[k_idx, :] * R_RH

        psik_ic = (x_ic * np.sqrt(2.0 * M_ic) / 2.0) * (theta1_ic + 1j * (dtheta1_dx_ic / M_ic)) * np.exp(1j * omega_ic)

        phi_m_val = phi_num[idx_m_phi]
        dphi_dx_m_val = dphi_dx_num[idx_m_phi]

        F_m = (kappa_val * x_m_val)**2 * phi_m_val
        dF_dx_m = 2.0 * kappa_val**2 * x_m_val * phi_m_val + (kappa_val * x_m_val)**2 * dphi_dx_m_val

        y_m = kappa_val * x_m_val / np.sqrt(3.0)
        deriv_term = (np.sqrt(3.0) / kappa_val) * dF_dx_m

        C1 = F_m * np.cos(y_m) - deriv_term * np.sin(y_m)
        C2 = F_m * np.sin(y_m) + deriv_term * np.cos(y_m)

        phi_interp = interp1d(x_full, phi_num, bounds_error=False, fill_value=0.0)
        dphi_interp = interp1d(x_full, dphi_dx_num, bounds_error=False, fill_value=0.0)

        phi_hybrid_ext = np.zeros_like(x_ext)
        dphi_hybrid_ext = np.zeros_like(x_ext)

        phi_hybrid_ext[mask_num] = phi_interp(x_ext[mask_num])
        dphi_hybrid_ext[mask_num] = dphi_interp(x_ext[mask_num])

        y_ext = kappa_val * x_ext[mask_ana] / np.sqrt(3.0)
        F_ext = C1 * np.cos(y_ext) + C2 * np.sin(y_ext)
        dF_ext = (kappa_val / np.sqrt(3.0)) * (-C1 * np.sin(y_ext) + C2 * np.cos(y_ext))

        phi_hybrid_ext[mask_ana] = F_ext / (kappa_val * x_ext[mask_ana])**2
        dphi_hybrid_ext[mask_ana] = (dF_ext - 2.0 * F_ext / x_ext[mask_ana]) / (kappa_val * x_ext[mask_ana])**2

        p_kappa_ext = (kappa_val**2 / (2.0 * mu)) * np.log(x_ext / x_ic)
        integrand_ext = np.exp(1j * p_kappa_ext) * (mu * x_ext * phi_hybrid_ext + 2.0j * dphi_hybrid_ext)
        
        integral_legal_final = cumulative_trapezoid(integrand_ext, x_ext, initial=0.0)[-1]

        psik_eq_legal = np.exp(-1j * p_kappa_ext[-1]) * (psik_ic - 1j * psi0_ic * integral_legal_final)
        
        delta_wkb_end[j] = 2.0 * np.real(np.conj(psi0_ic) * psik_eq_legal) / np.abs(psi0_ic)**2

    return np.abs(delta_wkb_end)

def noax_mode_solver(k, rho_phi, rho_r, GG, R):
    delta_r, delta_phi, Phi, theta_phi, theta_r, dPhi_dR = PertEMD_noAX(rho_phi, rho_r, GG, R, k)
    return delta_r, delta_phi, Phi, theta_phi, theta_r, dPhi_dR

def axion_mode_worker(args, rho_phi, rho_r, Hub, Temp, R, theta_ini, dtheta_ini, ma0, fa):
    k, delta_r_in, Phi_in, dPhi_dR_in = args
    out = NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r_in, Phi_in, dPhi_dR_in, ma0, fa)
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

def run_parameter_sweep(MA0_array, Tend_array, num_cores):
    root_output_folder = "WKB_QCD_data"
    os.makedirs(root_output_folder, exist_ok=True)
    
    Tini = 3e1
    kappa = 1e5
    maxi = 95
    theta0 = 0.8
    dtheta0 = 0.0
    
    n_k_modes = 300
    
    kappas_eqs = []
    delta_a_eqs = []
    x_eqs = []
    k_J_eqs = []
    k_RH_eqs = []
    Tend_list = []
    MA0_list = []
    R_RH_list = []
    k_osc_list = []
    k_L_list = []
    R_eq_list = []

    for idx in range(len(Tend_array)):
        Tend = Tend_array[idx]
        MA0 = MA0_array[idx]

        output_folder = os.path.join(root_output_folder, f"TRH_{Tend*1e3:.3f}MeV")
        os.makedirs(output_folder, exist_ok=True)
        
        precomputed_file = os.path.join(output_folder, f"noax_NSC_{Tend:.6f}.npz")

        R, rho_phi, rho_r, Temp, Hub, GG, R_RH, k_RH = background(Tend, Tini, maxi, kappa)

        k_values = np.logspace(np.log10(1 * k_RH), np.log10(250 * k_RH), n_k_modes)
        kappas = np.array(k_values) / k_RH
        x = R / R_RH
        
        noax_task = partial(noax_mode_solver, rho_phi=rho_phi, rho_r=rho_r, GG=GG, R=R)
        with Pool(num_cores) as pool:
            noax_results = list(tqdm(pool.imap(noax_task, k_values), total=len(k_values), desc=f"No-axion modes Tend={Tend:.3f}"))

        delta_r_noax, delta_phi_noax, Phi_noax, _, _, dPhi_dR_noax = map(list, zip(*noax_results))
        del noax_results
        gc.collect()

        save_precomputed_base(
            precomputed_file,
            background_data={"R": R, "Hub": Hub, "Temp": Temp},
            noax_data={"delta_r": np.array(delta_r_noax), "delta_phi": np.array(delta_phi_noax), "Phi": np.array(Phi_noax), "dPhi_dR": np.array(dPhi_dR_noax)},
            horizon_info={"kappas": kappas, "x": x},
            mode_info={"R_RH": R_RH, "k_RH": k_RH},
            input_info={"Tend": Tend},
        )

        gc.collect()
        
        x_end_chi = interp1d(Temp, x, bounds_error=False, fill_value=np.nan)(0.0101)
        if x_end_chi > 10:
            x_end_num = x_end_chi * 1.09
        else:
            x_end_num = 10
            
        mask_num = x <= x_end_num
        R_num = R[mask_num]
        rho_phi_num = rho_phi[mask_num]
        rho_r_num = rho_r[mask_num]
        Temp_num = Temp[mask_num]
        Hub_num = Hub[mask_num]
        
        delta_r_num = [dr[mask_num] for dr in delta_r_noax]
        Phi_num = [p[mask_num] for p in Phi_noax]
        dPhi_dR_num = [dp[mask_num] for dp in dPhi_dR_noax]

        mode_args = list(zip(k_values, delta_r_num, Phi_num, dPhi_dR_num))

        MA0_eV = MA0 * 1e9
        fa = 5.69 * 10**(-3) / MA0
        R_eq = get_x_eq(Tend*1e3) * R_RH
        k_Jeq = (get_jeans(MA0, R_eq) / k_RH)
        
        with Pool(num_cores) as pool:
            worker = partial(
                axion_mode_worker,
                rho_phi=rho_phi_num, rho_r=rho_r_num, Hub=Hub_num, Temp=Temp_num, R=R_num,
                theta_ini=theta0, dtheta_ini=dtheta0, ma0=MA0, fa=fa
            )

            axion_results = list(tqdm(pool.imap(worker, mode_args), total=len(k_values), desc=f"Tend {Tend:.3f} MA0 {MA0_eV:.1e}"))
            
        delta_a_s, delta_a_r, ax1, dax1 = zip(*[(r[0], r[1], r[2], r[3]) for r in axion_results])
        
        bg_first = axion_results[0][4]
        axo, daxo, ma_arr, rho_theta, dlnchi_dlnT_arr, R_osc, R_l, fa_input = bg_first

        hub_interp = interp1d(R, Hub, kind='cubic', bounds_error=False, fill_value=np.nan)
        
        R_osc_float = float(R_osc)
        R_l_float = float(R_l)
        
        k_osc = R_osc_float * hub_interp(R_osc_float) if not np.isnan(R_osc_float) else np.nan
        k_L = R_l_float * hub_interp(R_l_float) if not np.isnan(R_l_float) else np.nan

        rho_a = rho_theta * (fa**2)
        idx_osc = np.argmin(np.abs(R_num - R_osc_float))
        rho_a_osc = rho_a[idx_osc]
        ma_osc = ma_arr[idx_osc]

        payload = {
            "axiPerts": {"delta_a_s": delta_a_s, "delta_a_r": delta_a_r, "ax1": ax1, "dax1": dax1},
            "axiBack": {"rho_a": rho_a, "fa": fa, "axo": axo, "daxo": daxo, "ma": ma_arr, "rho_a_osc": rho_a_osc, "ma_osc": ma_osc},
            "scale_factors": {"R_l": R_l_float, "R_osc": R_osc_float},
            "inputs": {"MA0": MA0, "k_J_eq": k_Jeq, "Tend": Tend, "Tini": Tini},
        }

        output_file = os.path.join(output_folder, f"AX_Tend_{Tend:.6f}_MA0_{MA0_eV:.2e}.npz")
        np.savez_compressed(output_file, **payload)

        Phi_mat = np.array(Phi_noax)
        dPhi_dR_mat = np.array(dPhi_dR_noax)
        ax1_mat = np.array(ax1)
        dax1_mat = np.array(dax1)

        ax1_ic_vals = ax1_mat[:, -1]
        dax1_ic_vals = dax1_mat[:, -1]
        axo_ic = axo[-1]
        daxo_ic = daxo[-1]
        ma_ic = ma_arr[-1]
        x_ic_val = R_num[-1] / R_RH
        
        x_eq = get_x_eq(Tend*1e3)
        x_end = x_eq
        
        delta_a_eq = integrate_axion_bruteforce_live(
            x, kappas, ax1_ic_vals, dax1_ic_vals, Phi_mat, dPhi_dR_mat, 
            axo_ic, daxo_ic, ma_ic, Hub, R_RH, x_ic_val, x_end, N_ext=200000
        )

        kappas_eqs.append(kappas)
        delta_a_eqs.append(delta_a_eq)
        x_eqs.append(x_eq)
        k_J_eqs.append(k_Jeq)
        k_RH_eqs.append(k_RH)
        Tend_list.append(Tend)
        MA0_list.append(MA0)
        R_RH_list.append(R_RH)
        k_osc_list.append(k_osc)
        k_L_list.append(k_L)
        R_eq_list.append(R_eq)

        del axion_results, delta_a_s, delta_a_r, ax1, dax1, bg_first, axo, daxo, ma_arr, rho_theta, rho_a, worker
        gc.collect()

    master_file_path = os.path.join(root_output_folder, "qcd_mc.npz")
    np.savez_compressed(
        master_file_path,
        kappas=np.array(kappas_eqs),
        delta_a_eq=np.array(delta_a_eqs),
        x_eq=np.array(x_eqs),
        k_J_eq=np.array(k_J_eqs),
        k_RH=np.array(k_RH_eqs),
        Tend=np.array(Tend_list),
        MA0=np.array(MA0_list),
        R_RH=np.array(R_RH_list),
        k_osc=np.array(k_osc_list),
        k_L=np.array(k_L_list),
        R_eq=np.array(R_eq_list)
    )

import numpy as np
from multiprocessing import cpu_count

def run_simulations():
    num_cores = int(cpu_count() * 0.38)
    MA0_array = np.array([4.69858711e-18, 9.33410403e-18, 1.85429143e-17])
    Tend_array = np.array([0.005, 0.0083666, 0.014])

    run_parameter_sweep(MA0_array, Tend_array, num_cores)

if __name__ == "__main__":
    run_simulations()