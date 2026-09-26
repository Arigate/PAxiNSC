import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.signal import savgol_filter

from Axi_RK4 import R_terms, integrate_bg, integrate_pert
from NSC_cosmo import gstar, gstars, dgstar, dgstars, gstars_0

logT, logchi = np.loadtxt("Data/chi_data.dat", unpack=True)
chi_f = interp1d(logT, logchi, kind="cubic")
T_low, T_high = 10**logT[0], 10**logT[-1]
chi_0 = 10**chi_f(logT.min())

def pressureSM(T):
    return (np.pi**2 / 90.0) * (4.0 * gstars(T) - 3.0 * gstar(T)) * T**4

def delta_r_DT(delta_r, Temp):
    return delta_r / (4.0 + Temp * dgstar(Temp) / gstar(Temp))

def Chi(T):
    T = np.atleast_1d(T)
    out = np.zeros_like(T, dtype=float)
    low, high = T < T_low, T > T_high
    mid = ~low & ~high

    if np.any(high):
        chi_high = 10**chi_f(logT.max())
        out[high] = chi_high * (10**logT.max() / T[high])**8.16
    if np.any(low):
        out[low] = chi_0
    if np.any(mid):
        out[mid] = 10**chi_f(np.log10(T[mid]))

    return float(out[0]) if out.size == 1 else out

def ma(T, ma0):
    return ma0 * np.sqrt(Chi(T) / chi_0)

def R_axi(R, Hub, Temp, ma_arr):
    ma_f = interp1d(R, ma_arr, kind="cubic")
    Hub_f = interp1d(R, Hub, kind="cubic")
    Temp_f = interp1d(R, Temp, kind="cubic")

    f = lambda x: Temp_f(x) - 0.150
    R_l = root_scalar(f, bracket=[R[0], R[-1]], method="brentq").root if f(R[0]) * f(R[-1]) < 0 else np.nan

    f = lambda x: ma_f(x) - 2.0 * Hub_f(x)
    R_osc = root_scalar(f, bracket=[R[0], R[-1]], method="brentq").root if f(R[0]) * f(R[-1]) < 0 else np.nan
    return R_l, R_osc

def axion_background_solver(rho_phi, rho_r, Hub, R, Temp, ma0, ax0, dax0, points_per_cycle):
    if len(R) < 5:
        z = np.zeros_like(R)
        return z, z, z, np.nan, np.nan

    ma_arr = ma(Temp, ma0)
    R_arr, damp, mass2 = R_terms(rho_phi, rho_r, pressureSM(Temp), Hub, R, ma_arr)
    y0 = np.array([float(np.atleast_1d(ax0)[0]), float(np.atleast_1d(dax0)[0])], dtype=np.float64)

    sol = integrate_bg(R_arr, damp, mass2, y0, int(points_per_cycle), 10, 100000, 10, False)
    R_l, R_osc = R_axi(R_arr, Hub, Temp, ma_arr)
    return sol[0], sol[1], ma_arr, R_osc, R_l

def axion_pert_coupled_solver(k, rho_phi, rho_r, Hub, Temp, R, delta_r, Phi, dPhi_dR, ma_arr, ax0, dax0, dlnchi_dlnT, points_per_cycle):
    if len(R) < 5:
        z = np.zeros_like(R)
        return z, z, z, z, np.zeros((3, len(R)), dtype=np.float64)

    R_arr, damp, mode, mass2 = R_terms(rho_phi, rho_r, pressureSM(Temp), Hub, R, ma_arr, perturb=True)
    src_phi = np.ascontiguousarray(dPhi_dR, dtype=np.float64)
    src_dr = np.ascontiguousarray(-dlnchi_dlnT * delta_r_DT(delta_r, Temp), dtype=np.float64)
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    y0 = np.array([float(ax0[0]), float(dax0[0]), 0.0, 0.0], dtype=np.float64)

    sol, sources = integrate_pert(R_arr, float(k), damp, mode, mass2, src_phi, src_dr, Phi, y0, int(points_per_cycle), 10, 100000, 10, False)
    return sol[0], sol[1], sol[2], sol[3], sources

def smooth_perturbations(R, delta_rho, rho_a, delta_p):
    log_R = np.log(R)
    log_R_uniform = np.linspace(log_R[0], log_R[-1], len(R))

    delta_rho = interp1d(log_R, delta_rho, kind="linear", fill_value="extrapolate")(log_R_uniform)
    rho_a = interp1d(log_R, rho_a, kind="linear", fill_value="extrapolate")(log_R_uniform)
    delta_p = interp1d(log_R, delta_p, kind="linear", fill_value="extrapolate")(log_R_uniform)

    window = min(2001, len(R) if len(R) % 2 else len(R) - 1)

    if window > 3:
        delta_rho = savgol_filter(delta_rho, window, 3)
        rho_a = savgol_filter(rho_a, window, 3)
        delta_p = savgol_filter(delta_p, window, 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_a = delta_rho / rho_a
        cs2 = delta_p / delta_rho

    delta_a = interp1d(log_R_uniform, delta_a, kind="cubic", fill_value="extrapolate")(log_R)
    cs2 = interp1d(log_R_uniform, cs2, kind="cubic", fill_value="extrapolate")(log_R)
    delta_rho = interp1d(log_R_uniform, delta_rho, kind="cubic", fill_value="extrapolate")(log_R)
    return delta_a, cs2, delta_rho

def NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, ma0, fa_input):
    ma_arr = ma(Temp, ma0)
    dlnchi_dlnT = np.gradient(np.log(Chi(Temp)), np.log(Temp))
    dlnchi_dlnT[Temp <= 0.0101] = 0.0
    R_l, R_osc = R_axi(R, Hub, Temp, ma_arr)

    ax0 = np.full_like(R, theta_ini)
    dax0 = np.full_like(R, dtheta_ini)
    points_per_cycle = 150
    ax0, dax0, ax1, dax1, sources = axion_pert_coupled_solver(k, rho_phi, rho_r, Hub, Temp, R, delta_r, Phi, dPhi_dR, ma_arr, ax0, dax0, dlnchi_dlnT, points_per_cycle)

    rho_theta = 0.5 * (Hub * R * dax0)**2 + 2.0 * ma_arr**2 * np.sin(ax0 / 2.0)**2
    rho_a = fa_input**2 * rho_theta

    dT_T = delta_r_DT(delta_r, Temp)
    source_T = -dlnchi_dlnT * ma_arr**2 * dT_T * (1.0 - np.cos(ax0))
    kinetic = (Hub * R)**2 * (dax0 * dax1 - Phi * dax0**2)
    potential = ma_arr**2 * np.sin(ax0) * ax1
    delta_rho = fa_input**2 * (kinetic + potential - source_T)
    delta_p = fa_input**2 * (kinetic - potential + source_T)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_a_r = delta_rho / rho_a
        cs2_r = delta_p / delta_rho

    delta_a_s, cs2_s, delta_rho_s = smooth_perturbations(R, delta_rho, rho_a, delta_p)
    background = [ax0, dax0, ma_arr, rho_theta, dlnchi_dlnT, R_osc, R_l, fa_input]
    return delta_a_s, delta_a_r, ax1, dax1, background, delta_rho_s, delta_rho, cs2_s, cs2_r, sources
