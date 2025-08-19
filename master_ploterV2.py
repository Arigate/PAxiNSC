def plot_delta_grid(nn, kk, Tend, pert_name, folder="GRIDS/deltadata"):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import LogLocator, NullFormatter
    import os

    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 34
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    fig, ax = plt.subplots(figsize=(10, 8))  
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    plt.yscale('log')

    # Load data
    R_pert = np.loadtxt(f"{folder}/R_pert.txt")
    k_values = np.loadtxt(f"{folder}/k_values.txt")
    data = np.loadtxt(f"{folder}/{pert_name}.txt")

    # Load variables from dictionary dat
    R_C = np.loadtxt(f"{folder}/R.txt")
    Hub_C = np.loadtxt(f"{folder}/Hub.txt")
    H_RH_C = np.loadtxt(f"{folder}/H_RH.txt")
    ma_C = np.loadtxt(f"{folder}/ma.txt")
    R_osc_C = np.loadtxt(f"{folder}/R_osc.txt")
    R_RH_C = np.loadtxt(f"{folder}/R_RH.txt")
    Rc = np.loadtxt(f"{folder}/Rc.txt")
    k_RH_C = Hub_C[np.argmin(np.abs(R_C - R_RH_C))] * R_RH_C
    k_osc_C = ma_C[0] * R_osc_C / 2
    k_c = Hub_C[np.argmin(np.abs(R_C - Rc))] * Rc
    HOR_C = Hub_C * R_C

    Rmin = R_C[(R_C >= R_RH_C/10000)][0] / R_RH_C
    Rmax = R_pert[-1] / R_RH_C

    kmax = k_values[0] / k_RH_C
    kmin = k_values[-1] / k_RH_C

    # Normalized axes
    R_plot = R_pert / R_RH_C
    k_plot = k_values / k_RH_C

    # Crop data to desired range
    mask_R = (R_plot >= Rmin) & (R_plot <= Rmax)
    R_plot = R_plot[mask_R]
    data_plot = data[:, mask_R]

    # Initial value at R_RH
    k_J0 = ma_C[0]*R_RH_C

    # Original expression for R > R_RH
    mask_valid = R_C > R_RH_C
    R_int = R_C[mask_valid]
    expr = np.sqrt((2*k_RH_C*ma_C[0]*R_RH_C)/np.log(R_int/R_RH_C))

    # Normalize so that at R=R_RH it equals k_J0
    k_J_full = np.concatenate(([k_J0], expr * k_J0 / expr[0]))

    R_kJ = np.concatenate(([R_RH_C], R_int))


    # Adjust stride if there are few data points
    stride_R = 10
    stride_k = 5

    pert_latex = {
    "delta_a_data": r"\delta_a",
    "delta_r_data": r"\delta_r",
    "delta_phi_data": r"\delta_\phi",
    "Phi_data": r"\Phi"
    }

    R_plot_fast = R_plot[::stride_R]
    k_plot_fast = k_plot[::stride_k]
    #data_plot_fast = (data_plot / data_norm[:, np.newaxis])[::stride_k, ::stride_R]
    data_plot_fast = (data_plot/(10**(-4.5)))[::stride_k, ::stride_R] #here you can normalize with the quantity that you want
    data_plot_fast_log = np.abs(data_plot_fast)



    vmax = 2.5*10**(3)
    vmin = 0.9
    pcm = ax.pcolormesh(
R_plot_fast, k_plot_fast, data_plot_fast_log,
shading='nearest', cmap='PuBu',
norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)
)
   



    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.loglog(R_C/R_RH_C, HOR_C/k_RH_C, label=r'$\mathcal{H}/k_{\mathrm{RH}}$', color='blue')

    if pert_name == "delta_a_data":
        ax.loglog(R_C/R_RH_C, (R_C/1)*ma_C[0]/k_RH_C/2, label=r'$m_a R / k_{\mathrm{RH}}$', color='red', linestyle='--')
        
    #if nn != 0 or pert_name == "Phi_data" or pert_name == "delta_r_data":
        #ax.axhline(k_c/k_RH_C, color='orange', linestyle='--', alpha=0.6, label=r'$k_{c}/k_{\mathrm{RH}}$')
        #ax.axvline(Rc/R_RH_C, color='orange', linestyle='--', alpha=0.6, label=r'$R_{c}/R_{\mathrm{RH}}$')
    if nn==0 and pert_name == "delta_a_data":
        ax.loglog(R_kJ/R_RH_C, k_J_full/k_RH_C, color='black', linestyle='--', label=r'$k_J/k_{\mathrm{RH}}$')
    #ax.axvline(1, color='green', linestyle='--', alpha=0.85, label=r'$R_{\mathrm{RH}}$')

    # Labels and titles
    colorbar_labels = {
        'delta_a_data': r'$|\delta_a(R,k)/\Phi(R_{\mathrm{ini}})|$',
        'delta_r_data': r'$|\delta_r(R,k)/\Phi(R_{\mathrm{ini}})|$',
        'delta_phi_data': r'$|\delta_\phi(R,k)/\Phi(R_{\mathrm{ini}})|$',
        'Phi_data': r'$|\Phi(R,k)/\Phi(R_{\mathrm{ini}})|$',
    }
    cbar = fig.colorbar(pcm)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(colorbar_labels.get(pert_name, pert_name), fontsize=28)
    ax.set_xlabel(r'$R/R_{\mathrm{RH}}$', fontsize=28)
    ax.set_ylabel(r'$k/k_{\mathrm{RH}}$', fontsize=28)
    if nn==0 and pert_name == "delta_a_data":
        ax.set_ylim(kmin*1.2, 250)
    else:
        ax.set_ylim(kmin*1.2, kmax)
    ax.set_xlim(Rmin, Rmax)
    plots_folder = f"master_ploter"
    os.makedirs(plots_folder, exist_ok=True)
    numer = 3 * nn - 8 * kk
    denom = 2 * (4 - nn)
    X_frac = rf"\frac{{{numer}}}{{{denom}}}"
    # Title and filename
    if nn == 0 and kk == 0:
        ax.set_title(
            rf'$x={0}$, $T_{{\mathrm{{RH}}}}={Tend}$ $GeV$',
            fontsize=28
        )
        fname = f"{plots_folder}/{pert_name}_nn0_kk0_Tend{Tend}.pdf"
    else:
        ax.set_title(
            rf'$x={{{X_frac}}}$, $T_{{\mathrm{{RH}}}}={Tend}$ $GeV$',
            fontsize=28
        )
        fname = f"{plots_folder}/{pert_name}_nn{nn}_kk{kk}_Tend{Tend}.pdf"
    ax.legend(loc='lower left', fontsize=24)
    plt.tight_layout()
    fig.savefig(fname)
    
    plt.close()
    print('plot saved as', fname)

