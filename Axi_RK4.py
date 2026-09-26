import numpy as np
from numba import njit
#rho_i is a general energy density who behaves as dust and decay in SM radiation
def R_terms(rho_i, rho_r, P_r, Hub, R, ma_arr, perturb=False):
    R = np.ascontiguousarray(R, dtype=np.float64)
    Hub = np.ascontiguousarray(Hub, dtype=np.float64)
    ma_arr = np.ascontiguousarray(ma_arr, dtype=np.float64)

    den = 2.0 * (rho_i + rho_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hub4 = np.where(den != 0, (5.0 * rho_i + 5.0 * rho_r - 3.0 * P_r) / den, 0.0)

    damp = np.zeros_like(R)
    mask = R != 0
    damp[mask] = Hub4[mask] / R[mask]
    damp = np.ascontiguousarray(damp, dtype=np.float64)

    HR = Hub * R
    mass2 = np.zeros_like(R)
    mask = HR > 0
    mass2[mask] = (ma_arr[mask] / HR[mask])**2
    mass2 = np.ascontiguousarray(mass2, dtype=np.float64)

    if not perturb:
        return R, damp, mass2

    HR2 = Hub * R**2
    mode = np.zeros_like(R)
    mask = HR2 > 0
    mode[mask] = 1.0 / HR2[mask]**2
    return R, damp, np.ascontiguousarray(mode, dtype=np.float64), mass2

@njit
def _interp(x, y, xi):
    n = x.shape[0]
    if xi <= x[0]:
        return y[0]
    if xi >= x[n - 1]:
        return y[n - 1]

    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xi:
            lo = mid
        else:
            hi = mid

    return y[lo] + (y[hi] - y[lo]) * (xi - x[lo]) / (x[hi] - x[lo])

@njit
def _rhs_bg(R_now, y, R, damp, mass2):
    theta, dtheta = y
    out = np.empty(2, dtype=np.float64)
    out[0] = dtheta
    out[1] = -_interp(R, damp, R_now) * dtheta - _interp(R, mass2, R_now) * np.sin(theta)
    return out

@njit
def _sources(R_now, y, R, mass2, src_phi, src_dr, Phi):
    theta, dtheta = y[0], y[1]
    m2 = _interp(R, mass2, R_now)
    m2_sin = m2 * np.sin(theta)

    out = np.empty(3, dtype=np.float64)
    out[0] = 4.0 * _interp(R, src_phi, R_now) * dtheta
    out[1] = -2.0 * _interp(R, Phi, R_now) * m2_sin
    out[2] = _interp(R, src_dr, R_now) * m2_sin
    return out

@njit
def _rhs_pert(R_now, y, R, k, damp, mode, mass2, src_phi, src_dr, Phi):
    theta, dtheta, theta1, dtheta1 = y
    m2 = _interp(R, mass2, R_now)
    source = _sources(R_now, y, R, mass2, src_phi, src_dr, Phi)

    out = np.empty(4, dtype=np.float64)
    out[0] = dtheta
    out[1] = -_interp(R, damp, R_now) * dtheta - m2 * np.sin(theta)
    out[2] = dtheta1
    out[3] = -_interp(R, damp, R_now) * dtheta1 - (k * k * _interp(R, mode, R_now) + m2 * np.cos(theta)) * theta1 + source[0] + source[1] + source[2]
    return out

@njit
def _nsteps(delta_R, omega, points_per_cycle, min_steps, max_substeps, extra_steps):
    if omega <= 1e-10:
        n_sub = min_steps
    else:
        n_float = delta_R * omega * points_per_cycle / (2.0 * np.pi)
        n_sub = int(n_float) + extra_steps if extra_steps > 0 else int(np.ceil(n_float))
        if n_sub < min_steps:
            n_sub = min_steps

    if n_sub > max_substeps:
        n_sub = max_substeps
    return n_sub

@njit
def integrate_bg(R, damp, mass2, y0, points_per_cycle, min_steps=20, max_substeps=1000000, extra_steps=0, check_right=True):
    sol = np.zeros((2, R.size), dtype=np.float64)
    y = y0.astype(np.float64)
    R_now = R[0]
    sol[:, 0] = y

    for i in range(1, R.size):
        R_next = R[i]
        delta_R = R_next - R[i - 1]

        if delta_R <= 0:
            sol[:, i] = y
            continue

        m2_left = _interp(R, mass2, R_now)
        m2_right = _interp(R, mass2, R_next) if check_right else m2_left
        omega = np.sqrt(max(m2_left, m2_right))
        n_sub = _nsteps(delta_R, omega, points_per_cycle, min_steps, max_substeps, extra_steps)
        h = delta_R / n_sub

        for _ in range(n_sub):
            k1 = _rhs_bg(R_now, y, R, damp, mass2)
            k2 = _rhs_bg(R_now + 0.5 * h, y + 0.5 * h * k1, R, damp, mass2)
            k3 = _rhs_bg(R_now + 0.5 * h, y + 0.5 * h * k2, R, damp, mass2)
            k4 = _rhs_bg(R_now + h, y + h * k3, R, damp, mass2)
            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            R_now += h

        R_now = R_next
        sol[:, i] = y

    return sol

@njit
def integrate_pert(R, k, damp, mode, mass2, src_phi, src_dr, Phi, y0, points_per_cycle, min_steps=20, max_substeps=1000000, extra_steps=0, check_right=True):
    sol = np.zeros((4, R.size), dtype=np.float64)
    sources = np.zeros((3, R.size), dtype=np.float64)
    y = y0.astype(np.float64)
    R_now = R[0]
    sol[:, 0] = y
    sources[:, 0] = _sources(R_now, y, R, mass2, src_phi, src_dr, Phi)

    for i in range(1, R.size):
        R_next = R[i]
        delta_R = R_next - R[i - 1]

        if delta_R <= 0:
            sol[:, i] = y
            sources[:, i] = _sources(R_now, y, R, mass2, src_phi, src_dr, Phi)
            continue

        m2_left = _interp(R, mass2, R_now)
        mode_left = _interp(R, mode, R_now)

        if check_right:
            m2_right = _interp(R, mass2, R_next)
            mode_right = _interp(R, mode, R_next)
            omega = max(np.sqrt(k * k * mode_left + m2_left), np.sqrt(k * k * mode_right + m2_right))
        else:
            omega = np.sqrt(k * k * mode_left + m2_left)

        n_sub = _nsteps(delta_R, omega, points_per_cycle, min_steps, max_substeps, extra_steps)
        h = delta_R / n_sub

        for _ in range(n_sub):
            k1 = _rhs_pert(R_now, y, R, k, damp, mode, mass2, src_phi, src_dr, Phi)
            k2 = _rhs_pert(R_now + 0.5 * h, y + 0.5 * h * k1, R, k, damp, mode, mass2, src_phi, src_dr, Phi)
            k3 = _rhs_pert(R_now + 0.5 * h, y + 0.5 * h * k2, R, k, damp, mode, mass2, src_phi, src_dr, Phi)
            k4 = _rhs_pert(R_now + h, y + h * k3, R, k, damp, mode, mass2, src_phi, src_dr, Phi)
            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            R_now += h

        R_now = R_next
        sol[:, i] = y
        sources[:, i] = _sources(R_now, y, R, mass2, src_phi, src_dr, Phi)

    return sol, sources
