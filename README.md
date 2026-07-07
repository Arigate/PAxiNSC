# PAxiNSC

**PAxiNSC** (Perturbed Axions in Non-Standard Cosmologies) is a Python code for studying axion and axion-like-particle (ALP) dark matter in non-standard cosmologies. It solves the background evolution of an early matter-dominated universe sourced by a massive scalar field decaying into radiation, computes the corresponding linear perturbations, and evolves axion/ALP background and perturbation equations in the field formalism.

The code was developed and used for the analysis presented in:

> **From Rags to Jeans: Axion Miniclusters from Early Matter Domination**  
> Ariel Angulo, Paola Arias, Nicolás Bernal, Javier Redondo  
> arXiv:2606.19439 [hep-ph]

In this setup, radiation density and temperature perturbations grow efficiently during an early matter-dominated era. When the axion mass depends on temperature, these inhomogeneities source spatial fluctuations of the axion mass and provide an additional contribution to axion density perturbations. The code is designed to explore this mechanism for phenomenological ALPs and for the QCD axion.

---

## Repository layout

```text
PAxiNSC/
├── Data/
│   ├── geffcbest.txt                         # Energy effective degrees of freedom
│   ├── heffcbest.txt                         # Entropy effective degrees of freedom
│   ├── gstarcbest.txt                        # Auxiliary effective-degree table used by the background solver
│   └── chi_data.dat                          # QCD topological susceptibility input
├── Plots/                                    # Example output figures
├── Sim_Data/                                 # Local simulation outputs; intentionally not distributed
├── NSC_cosmo.py                              # Background and no-axion perturbation solvers
├── NSC_axions.py                             # ALP background and perturbation solver
├── QCD_axions.py                             # QCD axion background and perturbation solver
├── NSC_Pert_and_ax_Background_Plots_1_4.ipynb # Background and perturbation examples
├── Sources_Plot_5.ipynb                      # QCD axion source-term example plots
├── Varying_Tend_MA0_Plots_6_7.ipynb          # ALP parameter-scan plotting notebook
├── Varying_Cosmo_Plot_8.ipynb                # Standard vs non-standard cosmology comparison plots
├── Jeans_Grad_Plot_9.ipynb                   # Jeans-scale gradient maps
├── MC_plots.ipynb                            # QCD axion/minicluster plotting notebook
└── README.md
```

The Python modules expect the tabulated input files to be available under a local `Data/` directory. If the files are stored elsewhere, update the paths in the corresponding modules before running the code. The `Sim_Data/` directory is used for locally generated simulation outputs and is not distributed with the repository.

---

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy scipy numba matplotlib tqdm joblib jupyter
```

---

## Units and conventions

Unless stated otherwise, the code uses natural units with GeV-based inputs.

| Quantity | Convention |
| --- | --- |
| `Tend` | Reheating temperature |
| `Tini` | Initial integration temperature |
| `ma0` / `MA0` | Zero-temperature axion mass |
| `T_l` / `Tl` | ALP transition temperature, `T_\Lambda` |
| `fa` | Axion decay constant |
| `R` | Dimensionless scale factor used by the solver |
| `R_RH` | Scale factor at reheating, defined by `T(R_RH) ≈ Tend` |
| `k_RH` | Comoving reheating scale, `k_RH = H(R_RH) R_RH` |
| `kappa` | Dimensionless mode ratio, `kappa = k/k_RH` |

---

## Basic usage

### 1. Background evolution

```python
from NSC_cosmo import background

Tend = 0.020      # reheating temperature
Tini = 30.0       # initial temperature
maxi = 10         # integrate until R = maxi * R_RH
EP = 1e5          # initial scalar-to-radiation hierarchy

R, rho_phi, rho_r, Temp, Hub, Gamma, R_RH, k_RH = background(
    Tend=Tend,
    Tini=Tini,
    maxi=maxi,
    EP=EP,
)
```

`background` returns the scale-factor grid, scalar and radiation densities, temperature, Hubble rate, decay rate, reheating scale factor, and reheating comoving scale.

### 2. No-axion perturbations

```python
from NSC_cosmo import PertEMD_noAX

kappa = 10.0
k = kappa * k_RH

delta_r, delta_phi, Phi, theta_phi, theta_r, dPhi_dR = PertEMD_noAX(
    rho_phi=rho_phi,
    rho_r=rho_r,
    GG=Gamma,
    R=R,
    k=k,
)
```

This step computes the perturbations of the decaying scalar field, radiation, and the Newtonian potential. These arrays are used as external sources for the axion/ALP perturbation solvers.

### 3. ALP evolution

```python
from NSC_axions import NSC_PERT_axi as solve_alp

ma0 = 1e-19        # zero-temperature ALP mass
T_l = 0.025        # ALP transition temperature
b = 4
fa = None          # if None, the code estimates fa from the relic abundance

delta_a_s, delta_a_raw, ax1, dax1, axion_bg, delta_rho_s, delta_rho_raw, cs2_s, cs2_raw = solve_alp(
    rho_phi=rho_phi,
    rho_r=rho_r,
    Hub=Hub,
    Temp=Temp,
    R=R,
    k=k,
    theta_ini=1.0,
    dtheta_ini=0.0,
    delta_r=delta_r,
    Phi=Phi,
    dPhi_dR=dPhi_dR,
    R_RH=R_RH,
    ma0=ma0,
    T_l=T_l,
    fa_input=fa,
    b_val=b,
)
```

The ALP background information is returned in `axion_bg`:

```text
[theta, dtheta_dR, ma, rho_theta, b_profile, R_osc, R_Lambda, fa]
```

### 4. QCD axion evolution

```python
from QCD_axions import NSC_PERT_axi as solve_qcd_axion

ma0 = 1e-17          # zero-temperature QCD axion mass
fa = 5.69e-3 / ma0   # QCD axion relation used by the scan script

delta_a_s, delta_a_raw, ax1, dax1, axion_bg, delta_rho_s, delta_rho_raw, cs2_s, cs2_raw = solve_qcd_axion(
    rho_phi=rho_phi,
    rho_r=rho_r,
    Hub=Hub,
    Temp=Temp,
    R=R,
    k=k,
    theta_ini=0.8,
    dtheta_ini=0.0,
    delta_r=delta_r,
    Phi=Phi,
    dPhi_dR=dPhi_dR,
    ma0=ma0,
    fa_input=fa,
)
```

The QCD axion background information is returned in `axion_bg`:

```text
[theta, dtheta_dR, ma, rho_theta, dlnchi_dlnT, R_osc, R_QCD, fa]
```

The QCD module reads `Data/chi_data.dat` at import time, so this file must exist before importing `QCD_axions.py`.

---

## Parameter scans

### ALP scans

`Varying_Parameters.py` runs multiprocessing scans over reheating temperature, ALP transition temperature, and ALP mass. Edit the input arrays at the end of the file:

```python
Tends = [0.005, 0.020, 0.050]
Tlambdas = [0.025]
MA0_eVs = [1e-10]
```

Then run:

```bash
python Varying_Parameters.py
```

Depending on which array contains more than one value, the script writes compressed outputs to one of:

```text
Varying_Tl/
Varying_Tend/
Varying_MA0/
Single_Run/
```

### QCD axion scans

`QCD_MC.py` runs QCD axion simulations over paired arrays of zero-temperature masses and reheating temperatures. It solves the QCD axion dynamics numerically down to the low-temperature regime of the susceptibility table and then applies a WKB extension to matter-radiation equality.

Edit the arrays at the end of the file:

```python
MA0_array = np.array([...])
Tend_array = np.array([...])
```

Then run:

```bash
python QCD_MC.py
```

The current script writes outputs under:

```text
WKB_QCD_data/
```

including per-run data and a master file such as `qcd_mc.npz`.

---

## Plotting notebooks

The repository includes notebooks that reproduce the main plots and post-processing steps used in the paper. Two notebooks can generate the required numerical data directly from the notebook, while the remaining notebooks are intended to load precomputed data generated with the solver modules and stored locally under `Sim_Data/`.

### Self-contained plotting examples

- `NSC_Pert_and_ax_Background_Plots_1_4.ipynb`: computes and plots the non-standard-cosmology background and the no-axion perturbations for several Fourier modes. The final part of the notebook also computes and plots ALP background quantities such as the axion mass, field evolution, and energy density.
- `Sources_Plot_5.ipynb`: computes a QCD axion example and plots the source terms `S1`, `S2`, and `S3` for different modes. The final part also plots QCD axion background quantities such as the axion mass, field evolution, and energy density.

### Post-processing notebooks requiring precomputed data

- `Varying_Tend_MA0_Plots_6_7.ipynb`: loads ALP scan outputs and produces the reheating-temperature and axion-mass comparison plots.
- `Varying_Cosmo_Plot_8.ipynb`: loads previously generated data to compare standard and non-standard cosmological histories for both constant-mass and temperature-dependent-mass scenarios.
- `Jeans_Grad_Plot_9.ipynb`: loads scans over axion masses and reheating temperatures and produces gradient maps of the axion overdensity around the Jeans mode and around a lower mode.
- `MC_plots.ipynb`: loads QCD axion scan outputs to plot the nonlinear spectrum at equality, Jeans-related quantities, minicluster masses, and related phenomenological estimates.

The post-processing notebooks assume that the relevant `.npz` files have already been generated locally and that the folder names match those used in the notebook cells. These generated files can be very large and are therefore not included in the repository.

---

## Numerical notes

- The background solver uses `scipy.integrate.solve_ivp` with `LSODA`.
- The no-axion perturbation solver uses `Radau` for stiff evolution.
- Axion and ALP field equations are integrated with custom Numba-accelerated RK4 routines with sub-stepping based on the local oscillation frequency.
- The returned smoothed axion overdensity uses Savitzky-Golay filtering on a uniform logarithmic scale-factor grid.
- The cosine potential can become numerically sensitive for initial angles close to `pi`, where anharmonic effects and resonance-like features may be relevant.
- For precision runs, check convergence with respect to the scale-factor grid, mode grid, `points_per_cycle`, solver tolerances, and smoothing window.

---

## Citation

If you use this code in academic work, please cite the associated paper:

```bibtex
@article{Angulo:2026rags2jeans,
  title         = {From Rags to Jeans: Axion Miniclusters from Early Matter Domination},
  author        = {Angulo, Ariel and Arias, Paola and Bernal, Nicol{\'a}s and Redondo, Javier},
  year          = {2026},
  eprint        = {2606.19439},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph}
}
```

You may also cite this repository directly:

```bibtex
@software{paxinsc,
  title  = {PAxiNSC: Axion Perturbations in Non-Standard Cosmologies},
  author = {Angulo, Ariel},
  year   = {2026},
  url    = {https://github.com/Arigate/PAxiNSC}
}
```

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.
