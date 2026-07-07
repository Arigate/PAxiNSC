# BACKGROUND SOLVER 

from scipy.integrate import solve_ivp
import numpy as np
from scipy.interpolate import interp1d


def background(Tend, Tini, maxi, EP):
    Mp = 2.4e18

    data_gstar = np.loadtxt("Data/geffcbest.txt")
    gstar = interp1d(data_gstar[:, 0], data_gstar[:, 1], bounds_error=False, fill_value="extrapolate")

    data_gstars = np.loadtxt("Data/heffcbest.txt")
    gstars = interp1d(data_gstars[:, 0], data_gstars[:, 1], bounds_error=False, fill_value="extrapolate")

    data_fgstars = np.loadtxt("Data/gstarcbest.txt")
    fgstars = interp1d(data_fgstars[:, 0], data_fgstars[:, 1], bounds_error=False, fill_value="extrapolate")

    H_RH = (Tend**2) * np.pi * np.sqrt((gstar(Tend) / 10.0)) / (3.0 * Mp)
    Rini = 1.0
    Rmax = 1e16
    ws = 0.0
    bb = 5/2

    def rhoSM(T):
        T = np.maximum(T, 0.0)
        return (np.pi**2 / 30.0) * gstar(T) * T**4

    def ss(T):
        T = np.maximum(T, 0.0)
        return (2.0 * np.pi**2 / 45.0) * gstars(T) * T**3

    T_floor = 1e-30
    denom_floor = 1e-300

    def rhs(R, Y):
        T, rho_phi = Y
        T = max(T, T_floor)
        rho_phi = max(rho_phi, 0.0)

        rho_r_val = rhoSM(T)
        HH = np.sqrt((rho_r_val + rho_phi) / (3.0 * Mp**2))

        GG = bb * H_RH
        s_val = max(ss(T), denom_floor)
        fg_val = max(fgstars(T), denom_floor)

        dTdR = (-T / R + GG * rho_phi / (3.0 * HH * s_val * R)) * gstars(T) / np.sqrt(gstar(T)) / fg_val
        dPdR = -3.0 * (1.0 + ws) * rho_phi / R - GG * rho_phi / (HH * R)
        return [dTdR, dPdR]

    Y0 = [float(Tini), float(rhoSM(Tini) * EP)]
    R_eval = np.logspace(np.log10(Rini), np.log10(Rmax), 100000)

    sol = solve_ivp(
        rhs,
        [Rini, Rmax],
        Y0,
        method="LSODA",
        atol=1e-11,
        rtol=1e-11,
        t_eval=R_eval,
    )

    R = sol.t
    Temp = np.maximum(sol.y[0], 0.0)
    Rho_phi = np.maximum(sol.y[1], 0.0)
    Rho_r_arr = rhoSM(Temp)
    Hub = np.sqrt((Rho_phi + Rho_r_arr) / (3.0 * Mp**2))
    

    idx_RH = np.argmin(np.abs(Temp - Tend))
    R_RH = float(R[idx_RH])
    k_RH = float(Hub[idx_RH] * R_RH)


    limit_R = maxi * R_RH
    mask_cut = R <= limit_R
    
    R_cut = R[mask_cut]
    Temp_cut = Temp[mask_cut]
    Hub_cut = Hub[mask_cut]
    Rho_phi_cut = Rho_phi[mask_cut]
    Rho_r_cut = Rho_r_arr[mask_cut]
    
    Gamma_cut = np.full(len(R_cut), bb * H_RH)

    return (
        R_cut,
        Rho_phi_cut,
        Rho_r_cut,
        Temp_cut,
        Hub_cut,
        Gamma_cut,
        R_RH,
        k_RH,
    )



# PERTURBATION SOLVER / Charm version

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def perturbations_njit(R_val, Y, k, R_arr, rho_phi_arr, rho_r_arr, GG_arr):
    Mp = 2.4e18
    delta_phi, theta_phi, delta_r, theta_r, Phi = Y

    rho_phi_val = np.interp(R_val, R_arr, rho_phi_arr)
    rho_r_val = np.interp(R_val, R_arr, rho_r_arr)
    gamma_phi_val = np.interp(R_val, R_arr, GG_arr)

    eps = 1e-300
    rho_tot = rho_phi_val + rho_r_val
    E_val = np.sqrt(max(rho_tot, 0.0) / (3.0 * Mp**2)) + eps

    ratio = rho_phi_val / (rho_r_val + eps)
    gamma_ratio = ratio * (gamma_phi_val / (R_val * E_val))
    rho_delta_phi = rho_phi_val * delta_phi

    inv_R2_E = 1.0 / (R_val**2 * E_val)
    k2_inv_R2_E = (k**2) * inv_R2_E

    dPhi_dR = -(
        (1.0 / (6.0 * Mp**2 * R_val * E_val**2)) * (rho_delta_phi + rho_r_val * delta_r)
        + ((k**2) / (3.0 * R_val**3 * E_val**2) + 1.0 / R_val) * Phi
    )

    ddelta_phi_dR = -(gamma_phi_val / (R_val * E_val)) * Phi - inv_R2_E * theta_phi + 3.0 * dPhi_dR
    dtheta_phi_dR = k2_inv_R2_E * Phi - (1.0 / R_val) * theta_phi

    dtheta_r_dR = gamma_ratio * (3.0 / 4.0 * theta_phi - theta_r) + k2_inv_R2_E * (delta_r / 4.0 + Phi)
    ddelta_r_dR = gamma_ratio * (delta_phi - delta_r + Phi) - (4.0 / 3.0 * inv_R2_E) * theta_r + 4.0 * dPhi_dR

    return np.array([ddelta_phi_dR, dtheta_phi_dR, ddelta_r_dR, dtheta_r_dR, dPhi_dR])


def PertEMD_noAX(rho_phi, rho_r, GG, R, k):
    R = np.asarray(R, dtype=np.float64)
    rho_r = np.asarray(rho_r, dtype=np.float64)
    rho_phi = np.asarray(rho_phi, dtype=np.float64)
    GG = np.asarray(GG, dtype=np.float64)
    k = float(k)

    Rmax = float(R[-1])
    Rini = float(R[0])

    Phi_ini = np.sqrt((2.101)*10**(-9))
    Delta_phi_ini = -2.0 * Phi_ini
    Delta_r_ini = -2.0 * Phi_ini if rho_phi[0] < rho_r[0] else -Phi_ini

    Y0 = np.array([Delta_phi_ini, 0.0, Delta_r_ini, 0.0, Phi_ini], dtype=np.float64)

    sols = solve_ivp(
        perturbations_njit,
        [Rini, Rmax],
        Y0,
        method="Radau",
        atol=1e-13,
        rtol=1e-13,
        t_eval=R,
        args=(k, R, rho_phi, rho_r, GG)
    )

    if not sols.success:
        raise RuntimeError(f"The integrator failed: {sols.message}")

    delta_phi_res = sols.y[0]
    theta_phi_res = sols.y[1]
    delta_r_res = sols.y[2]
    theta_r_res = sols.y[3]
    Phi_res = sols.y[4]

    dPhi_dR_res = np.zeros(len(R), dtype=np.float64)
    for i in range(len(R)):
        Y_col = sols.y[:, i]
        dPhi_dR_res[i] = perturbations_njit(R[i], Y_col, k, R, rho_phi, rho_r, GG)[4]

    return delta_r_res, delta_phi_res, Phi_res, theta_phi_res, theta_r_res, dPhi_dR_res














