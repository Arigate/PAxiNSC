import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.signal import savgol_filter
from scipy.special import expit

from Axi_RK4 import R_terms, integrate_bg, integrate_pert
from NSC_cosmo import gstar, gstars, dgstar, dgstars, gstars_0


def pressureSM(T):
    return (np.pi**2 / 90.0) * (4.0 * gstars(T) - 3.0 * gstar(T)) * T**4

def delta_r_DT(delta_r, Temp):
    return delta_r / (4.0 + Temp * dgstar(Temp) / gstar(Temp))

def get_mass(Temp, T_l, b, ma0, smooth=15.0):
    Temp = np.asarray(Temp, dtype=float)
    if b == 0:
        return np.full_like(Temp, ma0, dtype=float)

    x = np.log(Temp / T_l)
    return ma0 * np.exp(-(b / smooth) * np.logaddexp(0.0, smooth * x))

def get_b_profile(Temp, T_l, b, smooth=15.0):
    Temp = np.asarray(Temp, dtype=float)
    if b == 0:
        return np.zeros_like(Temp, dtype=float)
    return b * expit(smooth * np.log(Temp / T_l))

def R_axi(R, Hub, Temp, T_l, ma_arr):
    ma_f = interp1d(R, ma_arr, kind="cubic")
    Hub_f = interp1d(R, Hub, kind="cubic")
    Temp_f = interp1d(R, Temp, kind="cubic")

    f = lambda x: Temp_f(x) - T_l
    R_l = root_scalar(f, bracket=[R[0], R[-1]], method="brentq").root if f(R[0]) * f(R[-1]) < 0 else np.nan

    f = lambda x: ma_f(x) - 2.0 * Hub_f(x)
    R_osc = root_scalar(f, bracket=[R[0], R[-1]], method="brentq").root if f(R[0]) * f(R[-1]) < 0 else np.nan
    return R_l, R_osc

def compute_abundance(rho_theta, ma_arr, R, Temp, R_osc, R_RH):
    h = 0.67810
    rho_crit = 3.445e-47
    T0 = 2.33e-13
    omega_cdm = 0.1201075

    R = np.asarray(R, dtype=float)
    Temp = np.asarray(Temp, dtype=float)
    rho_theta = np.asarray(rho_theta, dtype=float)
    ma_arr = np.asarray(ma_arr, dtype=float)
    R_osc = float(np.asarray(R_osc))
    R_RH = float(np.asarray(R_RH))

    if not np.isfinite(R_osc) or not np.isfinite(R_RH):
        return np.nan

    Temp_f = interp1d(R, Temp, kind="linear", fill_value="extrapolate")
    T_osc = float(Temp_f(R_osc))

    if R_osc >= R_RH:
        dilution = (T0 / T_osc)**3 * float(gstars_0 / gstars(T_osc))
    else:
        R_post = 1.2 * R_RH
        T_post = float(Temp_f(R_post))
        dilution = (T0 / T_post)**3 * float(gstars_0 / gstars(T_post)) * (R_osc / R_post)**3

    idx_osc = int(np.argmin(np.abs(R - R_osc)))
    ma_osc = float(ma_arr[idx_osc])
    if ma_osc <= 0.0:
        return np.nan

    rho_today = h**2 * (float(ma_arr[-1]) / ma_osc) * (float(rho_theta[idx_osc]) / rho_crit) * dilution
    return np.sqrt(omega_cdm / rho_today) if rho_today > 0.0 and np.isfinite(rho_today) else np.nan

def axion_background_solver(rho_phi, rho_r, Hub, R, Temp, ma0, ax0, dax0, T_l, b, points_per_cycle, smooth=15.0, max_substeps=1000000):
    if len(R) < 5:
        z = np.zeros_like(R)
        return z, z, z, np.nan, np.nan

    ma_arr = get_mass(Temp, T_l, b, ma0, smooth)
    R_arr, damp, mass2 = R_terms(rho_phi, rho_r, pressureSM(Temp), Hub, R, ma_arr)
    y0 = np.array([float(np.atleast_1d(ax0)[0]), float(np.atleast_1d(dax0)[0])], dtype=np.float64)

    sol = integrate_bg(R_arr, damp, mass2, y0, int(points_per_cycle), 20, int(max_substeps), 0, True)
    R_l, R_osc = R_axi(R_arr, Hub, Temp, T_l, ma_arr)
    return sol[0], sol[1], ma_arr, R_osc, R_l

def axion_pert_coupled_solver(k, rho_phi, rho_r, Hub, Temp, R, delta_r, Phi, dPhi_dR, ma_arr, ax0, dax0, b_arr, points_per_cycle, max_substeps):
    if len(R) < 5:
        z = np.zeros_like(R)
        return z, z, z, z, np.zeros((3, len(R)), dtype=np.float64)

    R_arr, damp, mode, mass2 = R_terms(rho_phi, rho_r, pressureSM(Temp), Hub, R, ma_arr, perturb=True)
    src_phi = np.ascontiguousarray(dPhi_dR, dtype=np.float64)
    src_dr = np.ascontiguousarray(2.0 * b_arr * delta_r_DT(delta_r, Temp), dtype=np.float64)
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    y0 = np.array([float(ax0[0]), float(dax0[0]), 0.0, 0.0], dtype=np.float64)

    sol, sources = integrate_pert(R_arr, float(k), damp, mode, mass2, src_phi, src_dr, Phi, y0, int(points_per_cycle), 20, int(max_substeps), 0, True)
    return sol[0], sol[1], sol[2], sol[3], sources

def smooth_perturbations(R, delta_rho, rho_a, delta_p):
    log_R = np.log(R)
    log_R_uniform = np.linspace(log_R[0], log_R[-1], len(R))

    delta_rho = interp1d(log_R, delta_rho, kind="linear", fill_value="extrapolate")(log_R_uniform)
    rho_a = interp1d(log_R, rho_a, kind="linear", fill_value="extrapolate")(log_R_uniform)
    delta_p = interp1d(log_R, delta_p, kind="linear", fill_value="extrapolate")(log_R_uniform)

    window = min(2001, len(R) if len(R) % 2 else len(R) - 1)

    if window > 3:
        delta_rho = savgol_filter(delta_rho, window, 3, mode="mirror")
        rho_a = savgol_filter(rho_a, window, 3, mode="mirror")
        delta_p = savgol_filter(delta_p, window, 3, mode="mirror")

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_a = delta_rho / rho_a
        cs2 = delta_p / delta_rho

    delta_a = interp1d(log_R_uniform, delta_a, kind="cubic", fill_value="extrapolate")(log_R)
    cs2 = interp1d(log_R_uniform, cs2, kind="cubic", fill_value="extrapolate")(log_R)
    delta_rho = interp1d(log_R_uniform, delta_rho, kind="cubic", fill_value="extrapolate")(log_R)
    return delta_a, cs2, delta_rho

def NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, R_RH, ma0, T_l, fa_input=None, b_val=None, smooth=15.0, points_per_cycle=600, max_substeps=1000000):
    if b_val is None:
        if fa_input is None:
            raise ValueError("b_val must be provided")
        b_val, fa_input = fa_input, None

    ma_arr = get_mass(Temp, T_l, b_val, ma0, smooth)
    R_l, R_osc = R_axi(R, Hub, Temp, T_l, ma_arr)
    b_arr = get_b_profile(Temp, T_l, b_val, smooth)

    ax0 = np.full_like(R, theta_ini)
    dax0 = np.full_like(R, dtheta_ini)
    ax0, dax0, ax1, dax1, sources = axion_pert_coupled_solver(k, rho_phi, rho_r, Hub, Temp, R, delta_r, Phi, dPhi_dR, ma_arr, ax0, dax0, b_arr, points_per_cycle, max_substeps)

    rho_theta = 0.5 * (Hub * R * dax0)**2 + 2.0 * ma_arr**2 * np.sin(ax0 / 2.0)**2
    fa = compute_abundance(rho_theta, ma_arr, R, Temp, R_osc, R_RH)
    rho_a = fa**2 * rho_theta

    dT_T = delta_r_DT(delta_r, Temp)
    source_T = 2.0 * b_arr * ma_arr**2 * dT_T * (1.0 - np.cos(ax0))
    kinetic = (Hub * R)**2 * (dax0 * dax1 - Phi * dax0**2)
    potential = ma_arr**2 * np.sin(ax0) * ax1
    delta_rho = fa**2 * (kinetic + potential - source_T)
    delta_p = fa**2 * (kinetic - potential + source_T)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_a_r = delta_rho / rho_a
        cs2_r = delta_p / delta_rho

    delta_a_s, cs2_s, delta_rho_s = smooth_perturbations(R, delta_rho, rho_a, delta_p)
    background = [ax0, dax0, ma_arr, rho_theta, b_arr, R_osc, R_l, fa]
    return delta_a_s, delta_a_r, ax1, dax1, background, delta_rho_s, delta_rho, cs2_s, cs2_r, sources
