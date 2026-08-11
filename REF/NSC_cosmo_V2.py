# BACKGROUND SOLVER 

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def background(Tend, Tini, maxi, EP):
    Mp = 2.4e18

    data_gstar = np.loadtxt("Data/geffdhs.dat")
    gstar = interp1d(data_gstar[:, 0], data_gstar[:, 1], bounds_error=False, fill_value="extrapolate")

    data_gstars = np.loadtxt("Data/heffdhs.dat")
    gstars = interp1d(data_gstars[:, 0], data_gstars[:, 1], bounds_error=False, fill_value="extrapolate")

    data_dgstar = np.loadtxt("Data/geffderivdhs.dat")
    dgstar = interp1d(data_dgstar[:, 0], data_dgstar[:, 1], bounds_error=False, fill_value="extrapolate")

    data_dgstars = np.loadtxt("Data/heffderivdhs.dat")
    dgstars = interp1d(data_dgstars[:, 0], data_dgstars[:, 1], bounds_error=False, fill_value="extrapolate")

    H_RH = float((Tend**2) * np.pi * np.sqrt(gstar(Tend) / 10.0) / (3.0 * Mp))
    Rini = 1.0
    Rmax = 1e16
    ws = 0.0
    bb = 5/2
    GG = bb * H_RH
    T_floor = 1e-30
    denom_floor = 1e-300

    def rhoSM(T):
        T = np.maximum(T, T_floor)
        return (np.pi**2 / 30.0) * gstar(T) * T**4

    def pressureSM(T):
        T = np.maximum(T, T_floor)
        return (np.pi**2 / 90.0) * (4.0 * gstars(T) - 3.0 * gstar(T)) * T**4

    def drhoSMdT(T):
        T = np.maximum(T, T_floor)
        return (np.pi**2 / 30.0) * T**3 * (4.0 * gstar(T) + T * dgstar(T))

    def dpressureSMdT(T):
        T = np.maximum(T, T_floor)
        return (np.pi**2 / 90.0) * T**3 * (16.0 * gstars(T) - 12.0 * gstar(T) + T * (4.0 * dgstars(T) - 3.0 * dgstar(T)))

    def rhs(R, Y):
        T, rho_phi = Y
        T = max(T, T_floor)
        rho_phi = max(rho_phi, 0.0)
        rho_r = float(rhoSM(T))
        P_r = float(pressureSM(T))
        drho_dT = max(float(drhoSMdT(T)), denom_floor)
        HH = np.sqrt((rho_r + rho_phi) / (3.0 * Mp**2))
        dTdR = (GG * rho_phi - 3.0 * HH * (rho_r + P_r)) / (R * HH * drho_dT)
        dRhoPhidR = -3.0 * (1.0 + ws) * rho_phi / R - GG * rho_phi / (HH * R)
        return [dTdR, dRhoPhidR]

    Y0 = [float(Tini), float(rhoSM(Tini) * EP)]
    R_eval = np.logspace(np.log10(Rini), np.log10(Rmax), 100000)
    sol = solve_ivp(rhs, [Rini, Rmax], Y0, method="LSODA", atol=1e-13, rtol=1e-13, t_eval=R_eval)

    R = sol.t
    Temp = np.maximum(sol.y[0], T_floor)
    Rho_phi = np.maximum(sol.y[1], 0.0)
    Rho_r = rhoSM(Temp)
    P_r = pressureSM(Temp)
    Hub = np.sqrt((Rho_phi + Rho_r) / (3.0 * Mp**2))
    dRho_r_dT = np.maximum(drhoSMdT(Temp), denom_floor)
    dP_r_dT = dpressureSMdT(Temp)
    dT_dR = (GG * Rho_phi - 3.0 * Hub * (Rho_r + P_r)) / (R * Hub * dRho_r_dT)
    dRho_r_dR = dRho_r_dT * dT_dR
    dP_r_dR = dP_r_dT * dT_dR

    idx_RH = np.argmin(np.abs(Temp - Tend))
    R_RH = float(R[idx_RH])
    k_RH = float(Hub[idx_RH] * R_RH)

    limit_R = maxi * R_RH
    mask_cut = R <= limit_R
    R_cut = R[mask_cut]
    Temp_cut = Temp[mask_cut]
    Hub_cut = Hub[mask_cut]
    Rho_phi_cut = Rho_phi[mask_cut]
    Rho_r_cut = Rho_r[mask_cut]
    P_r_cut = P_r[mask_cut]
    dT_dR_cut = dT_dR[mask_cut]
    dRho_r_dR_cut = dRho_r_dR[mask_cut]
    dP_r_dR_cut = dP_r_dR[mask_cut]
    Gamma_cut = np.full(len(R_cut), GG)
    return (R_cut, Rho_phi_cut, Rho_r_cut, P_r_cut, Temp_cut, Hub_cut, Gamma_cut, dT_dR_cut, dRho_r_dR_cut, dP_r_dR_cut, R_RH, k_RH)


# PERTURBATION SOLVER / Charm version

@njit
def perturbations_njit(R_val, Y, k, R_arr, rho_phi_arr, rho_r_arr, GG_arr, c_r2_arr, omega_r_arr):
    Mp = 2.4e18
    delta_phi, theta_phi, delta_r, theta_r, Phi = Y
    rho_phi_val = np.interp(R_val, R_arr, rho_phi_arr)
    rho_r_val = np.interp(R_val, R_arr, rho_r_arr)
    gamma_phi_val = np.interp(R_val, R_arr, GG_arr)
    c_r2_val = np.interp(R_val, R_arr, c_r2_arr)
    w_r_val = np.interp(R_val, R_arr, omega_r_arr)
    eps = 1e-300
    rho_tot = rho_phi_val + rho_r_val
    E_val = np.sqrt(max(rho_tot, 0.0) / (3.0 * Mp**2)) + eps
    ratio = rho_phi_val / (rho_r_val + eps)
    gamma_ratio = ratio * (gamma_phi_val / (R_val * E_val))
    rho_delta_phi = rho_phi_val * delta_phi
    inv_R2_E = 1.0 / (R_val**2 * E_val)
    k2_inv_R2_E = k**2 * inv_R2_E
    dPhi_dR = -((rho_delta_phi + rho_r_val * delta_r) / (6.0 * Mp**2 * R_val * E_val**2) + (k**2 / (3.0 * R_val**3 * E_val**2) + 1.0 / R_val) * Phi)
    ddelta_phi_dR = -(gamma_phi_val / (R_val * E_val)) * Phi - inv_R2_E * theta_phi + 3.0 * dPhi_dR
    dtheta_phi_dR = k2_inv_R2_E * Phi - theta_phi / R_val
    den_r = 1.0 + w_r_val
    ddelta_r_dR = gamma_ratio * (delta_phi + Phi - delta_r) - (3.0 / R_val) * (c_r2_val - w_r_val) * delta_r - den_r * inv_R2_E * theta_r + 3.0 * den_r * dPhi_dR
    dtheta_r_dR = (gamma_ratio / den_r) * (theta_phi - (1.0 + c_r2_val) * theta_r) - ((1.0 - 3.0 * c_r2_val) / R_val) * theta_r + k2_inv_R2_E * ((c_r2_val / den_r) * delta_r + Phi)
    return np.array([ddelta_phi_dR, dtheta_phi_dR, ddelta_r_dR, dtheta_r_dR, dPhi_dR])


def PertEMD_noAX(rho_phi, rho_r, c_r2, omega_r, GG, R, k):
    R = np.asarray(R, dtype=np.float64)
    rho_r = np.asarray(rho_r, dtype=np.float64)
    rho_phi = np.asarray(rho_phi, dtype=np.float64)
    c_r2 = np.asarray(c_r2, dtype=np.float64)
    omega_r = np.asarray(omega_r, dtype=np.float64)
    GG = np.asarray(GG, dtype=np.float64)
    k = float(k)
    Rmax = float(R[-1])
    Rini = float(R[0])
    Phi_ini = np.sqrt(2.101e-9)
    Delta_phi_ini = -2.0 * Phi_ini
    Delta_r_ini = -2.0 * Phi_ini if rho_phi[0] < rho_r[0] else -Phi_ini
    Y0 = np.array([Delta_phi_ini, 0.0, Delta_r_ini, 0.0, Phi_ini], dtype=np.float64)
    sols = solve_ivp(perturbations_njit, [Rini, Rmax], Y0, method="Radau", atol=1e-13, rtol=1e-13, t_eval=R, args=(k, R, rho_phi, rho_r, GG, c_r2, omega_r))
    if not sols.success:
        raise RuntimeError(f"The integrator failed: {sols.message}")
    delta_phi_res = sols.y[0]
    theta_phi_res = sols.y[1]
    delta_r_res = sols.y[2]
    theta_r_res = sols.y[3]
    Phi_res = sols.y[4]
    dPhi_dR_res = np.zeros(len(R), dtype=np.float64)
    for i in range(len(R)):
        dPhi_dR_res[i] = perturbations_njit(R[i], sols.y[:, i], k, R, rho_phi, rho_r, GG, c_r2, omega_r)[4]
    return delta_r_res, delta_phi_res, Phi_res, theta_phi_res, theta_r_res, dPhi_dR_res
