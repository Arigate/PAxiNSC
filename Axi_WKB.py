import numpy as np
from scipy.integrate import cumulative_trapezoid


def integrate_wkb(x, kappas, ax1_ic, dax1_ic, Phi, dPhi_dR, ax0_ic, dax0_ic, ma_ic, Hub, R_RH, x_ic, x_end, x_match=90.0, n_points=200_000):
    idx_ic = int(np.argmin(np.abs(x - x_ic)))
    H_ic = Hub[idx_ic]

    mu = ma_ic / (H_ic * x_ic**2)
    M_ic = mu * x_ic
    omega_ic = 0.5 * mu * x_ic**2

    dax0_dx = dax0_ic * R_RH
    psi0_ic = 0.5 * x_ic * np.sqrt(2.0 * M_ic) * (ax0_ic + 1j * dax0_dx / M_ic) * np.exp(1j * omega_ic)

    idx_match = int(np.argmin(np.abs(x - x_match)))
    x_match = float(x[idx_match])
    x_ext = np.logspace(np.log10(x_ic), np.log10(x_end), n_points)
    num_mask = x_ext < x_match
    ana_mask = ~num_mask

    delta_end = np.zeros(len(kappas), dtype=float)

    for i, kappa in enumerate(kappas):
        dax1_dx = dax1_ic[i] * R_RH
        psik_ic = 0.5 * x_ic * np.sqrt(2.0 * M_ic) * (ax1_ic[i] + 1j * dax1_dx / M_ic) * np.exp(1j * omega_ic)

        phi = Phi[i]
        dphi_dx = dPhi_dR[i] * R_RH
        phi_match = phi[idx_match]
        dphi_match = dphi_dx[idx_match]

        F_match = (kappa * x_match)**2 * phi_match
        dF_match = 2.0 * kappa**2 * x_match * phi_match + (kappa * x_match)**2 * dphi_match
        y_match = kappa * x_match / np.sqrt(3.0)
        deriv_term = np.sqrt(3.0) * dF_match / kappa

        C1 = F_match * np.cos(y_match) - deriv_term * np.sin(y_match)
        C2 = F_match * np.sin(y_match) + deriv_term * np.cos(y_match)

        phi_ext = np.zeros_like(x_ext)
        dphi_ext = np.zeros_like(x_ext)
        phi_ext[num_mask] = np.interp(x_ext[num_mask], x, phi, left=0.0, right=0.0)
        dphi_ext[num_mask] = np.interp(x_ext[num_mask], x, dphi_dx, left=0.0, right=0.0)

        y = kappa * x_ext[ana_mask] / np.sqrt(3.0)
        F = C1 * np.cos(y) + C2 * np.sin(y)
        dF = (kappa / np.sqrt(3.0)) * (-C1 * np.sin(y) + C2 * np.cos(y))

        phi_ext[ana_mask] = F / (kappa * x_ext[ana_mask])**2
        dphi_ext[ana_mask] = (dF - 2.0 * F / x_ext[ana_mask]) / (kappa * x_ext[ana_mask])**2

        phase = (kappa**2 / (2.0 * mu)) * np.log(x_ext / x_ic)
        integrand = np.exp(1j * phase) * (mu * x_ext * phi_ext + 2.0j * dphi_ext)
        integral = cumulative_trapezoid(integrand, x_ext, initial=0.0)[-1]

        psik_end = np.exp(-1j * phase[-1]) * (psik_ic - 1j * psi0_ic * integral)
        delta_end[i] = 2.0 * np.real(np.conj(psi0_ic) * psik_end) / np.abs(psi0_ic)**2

    return np.abs(delta_end)
