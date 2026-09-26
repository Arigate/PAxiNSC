# PAxiNSC

**PAxiNSC** (Perturbed Axions in Non-Standard Cosmologies) is a Python code for studying axion and axion-like-particle (ALP) perturbations during an early matter-dominated era (EMD) sourced by a massive scalar field that decays into Standard-Model radiation.

The code was developed for:

> **From Rags to Jeans: Axion Miniclusters from Early Matter Domination**  
> Ariel Angulo, Paola Arias, Nicolás Bernal, Javier Redondo  
> arXiv:2606.19439 [hep-ph]

In this scenario, radiation density and temperature perturbations grow during EMD. If the axion mass depends on temperature, these inhomogeneities also perturb the axion mass and provide an additional source for axion density perturbations. PAxiNSC follows this mechanism for both ALPs and the QCD axion.

---

## Main modules

- `NSC_cosmo.py`: non-standard cosmological background, Standard-Model thermodynamics, primordial potential, and no-axion perturbations.
- `Axi_RK4.py`: shared Numba-accelerated RK4 integrator for the axion background field and its perturbations.
- `ALP_NSC.py`: phenomenological ALP with a smooth temperature-dependent mass.
- `QCD_axions.py`: QCD axion with a mass determined from the topological susceptibility.
- `QCD_MC.py`: QCD-axion scan and evolution to matter-radiation equality.

The `low_T_RH/` directory contains the idealized low-reheating-temperature comparison codes, while `notebooks/` contains the post-processing notebooks used to generate the final figures.

---

## Installation

The core Python dependencies are

```bash
pip install numpy scipy numba matplotlib tqdm jupyter
```

Some plotting notebooks also use

```bash
pip install cblind
```

Several figures use Matplotlib with `text.usetex=True`, so a working LaTeX installation must also be available in `PATH`.

---

## Numerical workflow

The main logic of PAxiNSC is to first compute the non-standard cosmological background and its perturbations **without axions**. These quantities are then used as external sources for the axion system.

### 1. Non-standard cosmology

The background is computed with

```python
R, rho_phi, rho_r, P_r, Temp, Hub, GG, dT_dR, drho_r_dR, dP_r_dR, R_RH, k_RH = background(T_RH, Tini, maxi, ep)
```

where `R_RH` is the reheating scale factor and `k_RH = H_RH R_RH` is the associated comoving scale.

The primordial Newtonian potential is scale dependent and is generated with `primordial_phi_ini`, using the scalar amplitude and tilt defined in `NSC_cosmo.py`.

### 2. No-axion perturbations

The radiation, scalar-field, and Newtonian-potential perturbations are evolved before introducing the axion:

```python
delta_r, delta_phi, Phi, theta_phi, theta_r, dPhi_dR = PertEMD_noAX(rho_phi, rho_r, c_r2, omega_r, GG, R, k, Phi_ini)
```

### 3. Axion and ALP evolution

The axion background field and its linear perturbation are evolved together with the shared RK4 implementation in `Axi_RK4.py`. The integration is performed directly in the scale factor `R`, with sub-stepping determined by the local oscillation frequency.

For ALPs, `ALP_NSC.py` uses a smooth temperature-dependent mass characterized by `ma0`, `T_l`, and `b`. Setting `b = 0` gives the constant-mass case.

For the QCD axion, `QCD_axions.py` constructs the mass from the topological susceptibility stored in `Data/chi_data.dat`.

A typical ALP call is

```python
out = NSC_PERT_axi(rho_phi, rho_r, Hub, Temp, R, k, theta_ini, dtheta_ini, delta_r, Phi, dPhi_dR, R_RH, ma0, T_l, fa_input=None, b_val=b, smooth=smooth, points_per_cycle=points_per_cycle, max_substeps=max_substeps)
```

The solver returns both smoothed and unsmoothed axion overdensities, field perturbations, effective sound-speed information, the axion background, and the three source terms.

### 4. QCD evolution to equality

Direct field integration becomes unnecessarily expensive once the QCD axion is deep in its oscillatory regime. `QCD_MC.py` therefore solves the field equations numerically through the temperature-dependent-mass epoch and then uses `Axi_WKB.py` to continue the perturbations to matter-radiation equality.

---

## Inputs

All energy scales, masses, and temperatures used internally by the code are expressed in **GeV**. Some scan scripts use variables such as `ma0_ev` to specify that the variable is given in eV.

The main cosmological inputs are:

| Input | Meaning |
| --- | --- |
| `T_RH` | Reheating temperature |
| `Tini` | Initial integration temperature |
| `ep` | Initial scalar-to-radiation energy-density ratio |
| `maxi` | Final numerical integration point in units of `R_RH` |
| `theta_ini` | Initial axion misalignment angle |
| `dtheta_ini` | Initial derivative of the axion field |
| `kappa` | Dimensionless Fourier mode, `k/k_RH` |

For ALPs:

| Input | Meaning |
| --- | --- |
| `ma0` | Zero-temperature ALP mass |
| `T_l` / `Tl` | Temperature scale where the ALP mass approaches its low-temperature value |
| `b` | Temperature-dependent mass exponent, typically `b = 4` |
| `smooth` | Smoothness of the mass transition |
| `points_per_cycle` | RK4 resolution of the oscillatory evolution |
| `max_substeps` | Maximum RK4 substeps per background interval |

For the QCD axion, the mass-temperature relation is fixed by `Data/chi_data.dat`. In `QCD_MC.py`, the zero-temperature QCD axion mass associated with each reheating temperature is calculated internally.

---

## Input data

The cosmological background uses

```text
Data/geffdhs.dat
Data/heffdhs.dat
Data/geffderivdhs.dat
Data/heffderivdhs.dat
```

for the Standard-Model energy and entropy effective degrees of freedom and their temperature derivatives.

The equation-of-state treatment associated with these tables follows

> M. Drees, F. Hajkarim and E. R. Schmitz,  
> *The Effects of QCD Equation of State on the Relic Density of WIMP Dark Matter*,  
> JCAP 06 (2015) 025, arXiv:1503.03513 [hep-ph].

The QCD axion module additionally uses

```text
Data/chi_data.dat
```

for the topological susceptibility. This table was taken from the `qcd-axion-cosmology` repository:

```text
https://github.com/mathieukaltschmidt/qcd-axion-cosmology
```

corresponding to `qcd_axion_cosmology/data/chi_data.dat`.

The scripts should normally be run from the repository root because these paths are relative.

---

## Production scripts and generated data

`Varying_Parameters.py` stores the **full ALP evolution**, including the no-axion perturbations, axion perturbations, source terms, background quantities, and characteristic scales. Depending on the selected scan, the data are written to

```text
Sim_Data/Varying_TRH/
Sim_Data/Varying_Tl/
Sim_Data/Varying_MA0/
Sim_Data/Single_Run/
Sim_Data/Varying_Custom/
```

The more specialized scans store reduced files tailored to their corresponding notebooks:

```text
Sim_Data/Jeans_grad/Jeans_summary.npz
Sim_Data/Cosmos/V_Cosmo.npz
Sim_Data/QCD_MC/qcd_mc.npz
```

The generated `Sim_Data/` directory is local and is not tracked by Git.

---

## Repository layout

```text
PAxiNSC/
├── ALP_NSC.py
├── ALP_jeans_run.py
├── Axi_RK4.py
├── Axi_WKB.py
├── NSC_cosmo.py
├── QCD_axions.py
├── QCD_MC.py
├── Sources.py
├── Var_COSMO_NSC_SC.py
├── Varying_Parameters.py
│
├── Data/
│   ├── chi_data.dat
│   ├── geffdhs.dat
│   ├── geffderivdhs.dat
│   ├── heffdhs.dat
│   └── heffderivdhs.dat
│
├── low_T_RH/
│   ├── NSC_cosmo_ideal.py
│   ├── QCD_axions_ideal.py
│   └── QCD_MC_ideal.py
│
├── notebooks/
│   ├── Jeans.ipynb
│   ├── MC_plots.ipynb
│   ├── Varying_Cosmo.ipynb
│   └── Varying_Parameters.ipynb
│
├── Sim_Data/                  # generated locally, not tracked
├── README.md
└── LICENSE
```

---

## Citation

If you use this code, please cite the associated paper:

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

You may also cite the repository:

```bibtex
@software{paxinsc,
  title  = {PAxiNSC: Perturbed Axions in Non-Standard Cosmologies},
  author = {Angulo, Ariel},
  year   = {2026},
  url    = {https://github.com/Arigate/PAxiNSC}
}
```

For the Standard-Model equation-of-state tables, see:

```bibtex
@article{Drees:2015exa,
  author        = {Drees, Manuel and Hajkarim, Fazlollah and Schmitz, Ernany Rossi},
  title         = {The Effects of QCD Equation of State on the Relic Density of WIMP Dark Matter},
  journal       = {JCAP},
  volume        = {06},
  pages         = {025},
  year          = {2015},
  eprint        = {1503.03513},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph}
}
```

---

## License

PAxiNSC is distributed under the MIT License.
