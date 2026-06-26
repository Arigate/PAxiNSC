import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import root_scalar

@njit
def _interp_linear(x, y, xi):
    n = x.shape[0]
    if xi <= x[0]:
        return y[0]
    if xi >= x[n - 1]:
        return y[n - 1]
    
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xi:
            lo = mid
        else:
            hi = mid
    
    x0 = x[lo]
    x1 = x[hi]
    y0 = y[lo]
    y1 = y[hi]
    
    return y0 + (y1 - y0) * (xi - x0) / (x1 - x0)

@njit
def _rhs_coupled_system(R_val, Y, R_grid, k, damp_arr, mode_arr, mass_sq_arr, src_phi_arr, src_dr_arr, phi_arr):
    t0 = Y[0]
    dt0 = Y[1]
    t1 = Y[2]
    dt1 = Y[3]

    damp_val = _interp_linear(R_grid, damp_arr, R_val)     
    mode_val = _interp_linear(R_grid, mode_arr, R_val)  
    mass_sq  = _interp_linear(R_grid, mass_sq_arr, R_val) 
    dphi = _interp_linear(R_grid, src_phi_arr, R_val)   
    phi_val  = _interp_linear(R_grid, phi_arr, R_val)            
    bb_dr = _interp_linear(R_grid, src_dr_arr, R_val)      

    sin_t0 = np.sin(t0)
    cos_t0 = np.cos(t0)

    ddt0 = -damp_val * dt0 - mass_sq * sin_t0
    omega_sq_pert = (k * k * mode_val) + (mass_sq * cos_t0)

    s1 = 4.0 * dphi * dt0
    s2 = -2.0 * phi_val * (mass_sq * sin_t0)
    s3 = bb_dr * (mass_sq * sin_t0)
    total_source = s1 + s2 + s3
    
    ddt1 = -damp_val * dt1 - omega_sq_pert * t1 + total_source

    out = np.empty(4, dtype=np.float64)
    out[0] = dt0
    out[1] = ddt0
    out[2] = dt1
    out[3] = ddt1
    return out

@njit
def _integrate_smart(R_grid, k, damp_arr, mode_arr, mass_sq_arr, src_phi, src_dr, phi_arr, R_eval, Y0, points_per_cycle):
    n_eval = R_eval.shape[0]
    sol = np.zeros((4, n_eval), dtype=np.float64)
    
    Y = Y0.astype(np.float64)
    curr_R = R_eval[0]
    sol[:, 0] = Y

    for i in range(1, n_eval):
        target_R = R_eval[i]
        delta_grid = target_R - R_eval[i-1]
        
        if delta_grid <= 0:
            sol[:, i] = Y
            continue

        mass_sq_val = _interp_linear(R_grid, mass_sq_arr, curr_R)
        mode_val = _interp_linear(R_grid, mode_arr, curr_R)
        omega_total = np.sqrt(k * k * mode_val + mass_sq_val)

        if omega_total > 1e-10:
            n_float = (delta_grid * omega_total * points_per_cycle) / (2.0 * np.pi)
            n_sub = int(n_float) + 10 
        else:
            n_sub = 10
            
        if n_sub > 100000: 
            n_sub = 100000

        h = delta_grid / n_sub
        
        for _ in range(n_sub):
            k1 = _rhs_coupled_system(curr_R, Y, R_grid, k, damp_arr, mode_arr, mass_sq_arr, src_phi, src_dr, phi_arr)
            k2 = _rhs_coupled_system(curr_R + 0.5 * h, Y + 0.5 * h * k1, R_grid, k, damp_arr, mode_arr, mass_sq_arr, src_phi, src_dr, phi_arr)
            k3 = _rhs_coupled_system(curr_R + 0.5 * h, Y + 0.5 * h * k2, R_grid, k, damp_arr, mode_arr, mass_sq_arr, src_phi, src_dr, phi_arr)
            k4 = _rhs_coupled_system(curr_R + h, Y + h * k3, R_grid, k, damp_arr, mode_arr, mass_sq_arr, src_phi, src_dr, phi_arr)
            
            Y = Y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            curr_R += h
        
        curr_R = target_R
        sol[:, i] = Y
        
    return sol

def get_mass(Temp, T_l, b, ma0):
    temp_arr = np.asarray(Temp, dtype=float)
    if b == 0:
        return np.full_like(temp_arr, ma0, dtype=float)
    return np.where(temp_arr <= T_l, ma0, ma0 * (T_l / temp_arr) ** b)

def get_bb_profile(R, R_limit, b):
    R_arr = np.asarray(R, dtype=float)
    if b == 0:
        return np.zeros_like(R_arr, dtype=float)
    return np.where(R_arr <= R_limit, b, 0.0)

def find_critical_scales(R_grid, Hub, Temp, T_l, ma):
    ma_interp = interp1d(R_grid, ma, kind='cubic')
    hub_interp = interp1d(R_grid, Hub, kind='cubic')
    temp_interp = interp1d(R_grid, Temp, kind='cubic')
    
    def root_func_L(R_val):
        return temp_interp(R_val) - T_l
        
    fl_max = root_func_L(R_grid[-1])
    fl_min = root_func_L(R_grid[0])
    
    if fl_max * fl_min < 0:
        R_limit = root_scalar(root_func_L, bracket=[R_grid[0], R_grid[-1]], method='brentq').root
    else:
        R_limit = np.nan
        
    def root_func_osc(R_val):
        return ma_interp(R_val) - 2.0 * hub_interp(R_val)
        
    f_max = root_func_osc(R_grid[-1])
    f_min = root_func_osc(R_grid[0])

    if f_max * f_min < 0:
        R_osc = root_scalar(root_func_osc, bracket=[R_grid[0], R_grid[-1]], method='brentq').root
    else:
        R_osc = np.nan
        
    return R_limit, R_osc

def compute_abundance(rho_theta, ma, R, Temp, R_osc, R_RH):
    filename = "Data/heffcbest.txt"
    data = np.loadtxt(filename, skiprows=0)
    T_tab = data[:, 0]  
    gstars_tab = data[:, 1]  
    gstars_interp = interp1d(T_tab, gstars_tab, fill_value='extrapolate')

    h_hub = 0.67810
    rho_crit = 3.445e-47 
    T_today = 2.33e-13
    omega_cdm = 0.1201075

    R_arr = np.asarray(R, dtype=float)
    Temp_arr = np.asarray(Temp, dtype=float)
    rho_theta_arr = np.asarray(rho_theta, dtype=float)
    ma_arr = np.asarray(ma, dtype=float)
    R_osc_val = float(np.asarray(R_osc))
    R_RH_val = float(np.asarray(R_RH))
    
    T_interp = interp1d(R_arr, Temp_arr, kind='linear', fill_value='extrapolate')
    T_osc = float(T_interp(R_osc_val))

    if R_osc_val >= R_RH_val:
        R_osc_origin = R_osc_val
        T_dilution = (T_today / T_osc) ** 3
        gstar_dilution = float(gstars_interp(T_today) / gstars_interp(T_osc))
        dilution_factor = T_dilution * gstar_dilution
    else:
        R_osc_origin = R_osc_val
        R_post = 1.2 * R_RH_val
        R_ratio = R_osc_origin / R_post
        T_post = float(T_interp(R_post))
        T_dilution = (T_today / T_post) ** 3
        gstar_dilution = float(gstars_interp(T_today) / gstars_interp(T_post))
        dilution_factor = T_dilution * gstar_dilution * (R_ratio ** 3)

    ma_final = float(ma_arr[-1])
    idx_osc = int(np.argmin(np.abs(R_osc_origin - R_arr)))
    ma_osc = float(ma_arr[idx_osc])

    if ma_osc <= 0.0:
        return np.nan

    ma_ratio = ma_final / ma_osc
    rho_ratio = float(rho_theta_arr[idx_osc]) / rho_crit
    rho_today = (h_hub ** 2) * ma_ratio * rho_ratio * dilution_factor

    if rho_today <= 0.0 or not np.isfinite(rho_today):
        return np.nan

    fa_value = np.sqrt(omega_cdm / rho_today)
    return fa_value

def axion_pert_coupled_solver(k, rho_phi, rho_r, Hub, R, delta_r, Phi, dPhi_dR, ma, ax0, dax0, bb_array, points_per_cycle):
    if len(R) < 5: 
        return np.zeros_like(R), np.zeros_like(R), np.zeros_like(R), np.zeros_like(R)
    
    R_arr = np.ascontiguousarray(R, dtype=np.float64)
    Hub_arr = np.ascontiguousarray(Hub, dtype=np.float64)
    ma_arr = np.ascontiguousarray(ma, dtype=np.float64)
    
    num_hub = 5.0 * rho_phi + 4.0 * rho_r
    den_hub = 2.0 * (rho_phi + rho_r)
    with np.errstate(divide='ignore', invalid='ignore'):
        Hub4 = np.where(den_hub != 0, num_hub / den_hub, 0.0)
    
    damp_arr = np.zeros_like(R_arr)
    mask_r = R_arr != 0
    damp_arr[mask_r] = Hub4[mask_r] / R_arr[mask_r]
    damp_arr = np.ascontiguousarray(damp_arr, dtype=np.float64)
    
    denom_mode = Hub_arr * (R_arr**2)
    mode_arr = np.zeros_like(R_arr)
    mask_mode = denom_mode > 0
    mode_arr[mask_mode] = 1.0 / (denom_mode[mask_mode])**2
    mode_arr = np.ascontiguousarray(mode_arr, dtype=np.float64)
    
    denom_H = Hub_arr * R_arr
    mass_sq_pure = np.zeros_like(R_arr)
    mask_ma = denom_H > 0
    mass_sq_pure[mask_ma] = (ma_arr[mask_ma] / denom_H[mask_ma])**2
    mass_sq_pure = np.ascontiguousarray(mass_sq_pure, dtype=np.float64)
    
    src_phi_arr = np.ascontiguousarray(dPhi_dR, dtype=np.float64)
    phi_arr_c = np.ascontiguousarray(Phi, dtype=np.float64)
    src_dr_arr = np.ascontiguousarray(bb_array * delta_r / 2.0, dtype=np.float64)
    
    y0_scalar = float(ax0[0])
    dy0_scalar = float(dax0[0])
    Y0 = np.array([y0_scalar, dy0_scalar, 0.0, 0.0], dtype=np.float64)
    
    sol = _integrate_smart(
        R_arr, float(k), damp_arr, mode_arr, mass_sq_pure, src_phi_arr, src_dr_arr, phi_arr_c,
        R_arr, Y0, 
        int(points_per_cycle) 
    )

    return sol[0, :], sol[1, :], sol[2, :], sol[3, :]

def NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, R_RH, ma0, T_l, fa_input, b_val):
    ma_arr = get_mass(Temp, T_l, b_val, ma0)
    R_l, R_osc = find_critical_scales(R, Hub, Temp, T_l, ma_arr)
    bb_arr = get_bb_profile(R, R_l, b_val)
    
    ax0_ini_arr = np.full_like(R, theta_ini)
    dax0_ini_arr = np.full_like(R, dtheta_ini)
    points_per_cycle =150
    
    ax0_new, dax0_new, ax1, dax1 = axion_pert_coupled_solver(
        k, rho_phi, rho_r, Hub, R, delta_r, Phi, dPhi_dR, ma_arr, ax0_ini_arr, dax0_ini_arr, bb_arr, 
        points_per_cycle=points_per_cycle
    )
    
    rho_theta = ((Hub * R * dax0_new)**2) / 2.0 + ma_arr**2 * (2.0 * np.sin(ax0_new / 2.0)**2)
    
    if fa_input is None:
        fa_val = compute_abundance(rho_theta, ma_arr, R, Temp, R_osc, R_RH)
    else:
        fa_val = fa_input
        
    rho_a_raw = fa_val**2 * rho_theta
    term_source = (bb_arr / 2.0) * ma_arr**2 * delta_r * (2.0 * np.sin(ax0_new / 2.0)**2)
    kinetic_coupling = (Hub * R)**2 * (dax0_new * dax1 - Phi * dax0_new**2)
    potential_coupling = ma_arr**2 * np.sin(ax0_new) * ax1
    
    delta_rho_r = fa_val**2 * (kinetic_coupling + potential_coupling - term_source)
    delta_p_r = fa_val**2 * (kinetic_coupling - potential_coupling + term_source)

    with np.errstate(divide='ignore', invalid='ignore'):
        delta_a_r = delta_rho_r / rho_a_raw
        cs2_r = delta_p_r / delta_rho_r

    log_R = np.log(R)
    log_R_uniform = np.linspace(log_R[0], log_R[-1], len(R))
    
    f_delta = interp1d(log_R, np.abs(delta_rho_r), kind='linear', fill_value="extrapolate")
    f_rho = interp1d(log_R, rho_a_raw, kind='linear', fill_value="extrapolate")
    f_press = interp1d(log_R, delta_p_r, kind='linear', fill_value="extrapolate")
    
    d_rho_unif = f_delta(log_R_uniform)
    rho_a_unif = f_rho(log_R_uniform)
    d_p_unif = f_press(log_R_uniform)
    
    window_len = 2001 
    poly_order = 3
    
    if len(d_rho_unif) < window_len:
        window_len = len(d_rho_unif) // 2 * 2 + 1 
    
    delta_rho_smooth = savgol_filter(d_rho_unif, window_len, poly_order)
    rho_a_smooth = savgol_filter(rho_a_unif, window_len, poly_order)
    delta_p_smooth = savgol_filter(d_p_unif, window_len, poly_order)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_a_uniform = delta_rho_smooth / rho_a_smooth
        cs2_uniform = delta_p_smooth / delta_rho_smooth
    
    f_final_delta = interp1d(log_R_uniform, delta_a_uniform, kind='cubic', fill_value="extrapolate")
    f_final_cs2 = interp1d(log_R_uniform, cs2_uniform, kind='cubic', fill_value="extrapolate")
    f_final_rho = interp1d(log_R_uniform, delta_rho_smooth, kind='cubic', fill_value="extrapolate")
    
    delta_a_s = f_final_delta(log_R)
    cs2_s = f_final_cs2(log_R)
    delta_rho_s = f_final_rho(log_R)
    background=[ax0_new, dax0_new,ma_arr,rho_theta,bb_arr,R_osc,R_l,fa_val]
    return delta_a_s, delta_a_r, ax1, dax1, background, delta_rho_s, delta_rho_r, cs2_s, cs2_r