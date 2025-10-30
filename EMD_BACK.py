

from math import e


def axion_backsolver_ma(Tend, nn, kk, prop, theta, Approx, fa,s,ps,CDM):
        ##########################################################################################
        # IMPORTS
        import os
        import numpy as np
        from scipy.interpolate import interp1d
        from scipy.integrate import solve_ivp
        import matplotlib.pyplot as plt
        from scipy.optimize import root_scalar

        ##################################################################################################
        # Import data
        Mp = 2.4e18  # GeV  reduced Planck mass
        filename = "Data/geffcbest.txt"
        data = np.loadtxt(filename, skiprows=0)
        T = data[:, 0]  # [GeV]
        xgstar = data[:, 1]  # [-]
        gstar = interp1d(T, xgstar, fill_value='extrapolate')

        filename = "Data/heffcbest.txt"
        data = np.loadtxt(filename, skiprows=0)
        T = data[:, 0]  # [GeV]
        xgstars = data[:, 1]  # [-]
        gstars = interp1d(T, xgstars, fill_value='extrapolate')

        filename = "Data/gstarcbest.txt"
        data = np.loadtxt(filename, skiprows=0)
        T = data[:, 0]  # [GeV]
        xfgstars = data[:, 1]  # [-]
        fgstars = interp1d(T, xfgstars, fill_value='extrapolate')

        data = np.genfromtxt('Data/Chipaper.csv', delimiter=',')  
        Tchi = data[:, 0]  # Temperature in GeV
        Chi = 10**data[:, 1]
        Chiint = interp1d(Tchi, Chi)

        ##################################################################################################

        # 1 NON-STANDARD BACKGROUND:
        # WHO IS DOMINATING INITIALLY? Kappa small implies RD

        kappa = 1e-3
        H_RH = (Tend**2) * np.pi * np.sqrt((gstar(Tend) / 10)) / (3 * Mp)
        # Temperature:
        Tini = 1e7  # GeV
        # Scale Factor
        Rmax = 1e15
        Rini = 1
        ws = 0
        # 1.1 Radiation energy density:
        def rhoSM(T):
                T = np.where(T < 0, 0, T)
                return np.pi**2 / 30 * gstar(T) * T**4

        # 1.2 Entropy:
        def ss(T):
                T = np.where(T < 0, 0, T)
                return 2 * np.pi**2 / 45 * gstars(T) * T**3

  
        def Background(R, Y, w, bb, H_RH, R_RH, kk, nn):
                T, rho_phi = Y
                x = (3 * nn - 8 * kk) / (2 * (4 - nn))
                if T < 0:
                        T = 1e-15
                if rho_phi < 0:
                        rho_phi = 0

                HH = np.sqrt((rhoSM(T) + rho_phi) / (3 * Mp**2))
                # DEFINITION OF GAMMA DECAY RATE
                #GG = bb * (H_RH) * ((T/Tend)**nn) * ((R/R_RH)**kk)
                GG= bb * H_RH * (R_RH/R)**x
                dTdR = (-T / R + GG * rho_phi / (3 * HH * ss(T) * R)) * gstars(T) / np.sqrt(gstar(T)) / fgstars(T)
                dPdR = -3 * (1 + w) * rho_phi / R - GG * rho_phi / (HH * R)

                return [dTdR, dPdR]

        # INITIAL CONDITIONS:
        Phini = rhoSM(Tini) * kappa
        Y0 = [Tini, Phini]

        # 2 Scale factors
        R_range = [Rini, Rmax]

        # 2.2  Aproximated Reheating moment
        H_RH = (Tend**2) * np.pi * np.sqrt((gstar(Tend) / 10)) / (3 * Mp)
        H_ini = (Tini**2) * np.pi * np.sqrt((gstar(Tini) / 10)) / (3 * Mp)
        R_RH = ((2 * kappa / (kappa + 1)) * (H_ini / H_RH)**2)**(1 / 3)
        r_rh=R_RH
        k_rh=H_RH*R_RH
        # 3. GAMMA CONFIGURATION
        x = (3 * nn - 8 * kk) / (2 * (4 - nn))
        bb = np.abs(1 - x)
        
        # Numerical solution of RHO_R AND RHO_PHI
        R_eval = np.logspace(np.log10(R_range[0]), np.log10(R_range[1]), 15000)


        sol = solve_ivp(
        lambda R, Y: Background(R, Y, ws, bb, H_RH, R_RH, kk, nn),
        R_range,
        Y0,
        method='RK45',
        atol=1e-10,
        rtol=1e-10,
        t_eval=R_eval
        )
        # SOLUTIONS 
        R = sol.t 
        Temp = np.where(sol.y[0] < 0, 0, sol.y[0])  # TEMPERATURE
        Rho_phi = np.where(sol.y[1] < 0, 0, sol.y[1])  # SCALAR
        Rho_r = interp1d(R,rhoSM(Temp) ) # RADIATION
        Hub = np.sqrt((Rho_phi + rhoSM(Temp)) / (3 * Mp**2))  # HUBBLE PARAMETER

        # INTERPOLATIONS
        rho_r_d = interp1d(R, Rho_r(R))
        rho_phi_d = interp1d(R, Rho_phi)
        
        # FIND THE NUMERICAL REHEATING
        def equation_to_solve(R_val):
                return rho_r_d(R_val) - rho_phi_d(R_val)

        R_solutions = []
        for i in range(len(R) - 1):
                try:
                        sol = root_scalar(equation_to_solve, bracket=[R[i], R[i+1]], method='brentq')
                        if sol.converged:
                                R_solutions.append(sol.root)
                except ValueError:
                        continue

        R_eq = R_solutions[0]  # FIRST EQUALITY EPOCH
        R_RH=R_solutions[-1]  # REHEATING EPOCH
 


        ############################################
        # OUTPUT Hub, Rho_phi, Rho_r, R, R_RH, Rcn, Temp
        # 0      1      2    3    4    5    6

        if nn != 0:
                print(r'$\Gamma=\Gamma(T)$  - $m(T)=m_a$')
        if nn == 0:
                print(r'$\Gamma=constant$ - $m(T)=m_a$')

        # INTERPOLATIONS
        rangs=R[(R>=R_eq) & (R<=R_RH)]
        mask = (R >= R_eq) & (R <= R_RH)
        R_search = R[mask]
        rho_vals = rho_r_d(R_search)
        drho_dR = np.gradient(rho_vals, R_search)

      
        sign_change = np.where((drho_dR[:-1] < 0) & (drho_dR[1:] > 0))[0]
        if len(sign_change) > 0:
                Rc = R_search[sign_change[0]+1]
        else:
        
                rho_mod = rho_vals * R_search**3
                dmod_dR = np.gradient(rho_mod, R_search)
                sign_change_mod = np.where((dmod_dR[:-1] < 0) & (dmod_dR[1:] > 0))[0]
                if len(sign_change_mod) > 0:
                        Rc = R_search[sign_change_mod[0]+1]
                else:
                        
                        Rc = R_search[np.argmin(rho_mod)]


        Hub_d = interp1d(R, Hub)
        T = interp1d(R, Temp)
        umbral = 1e-30
        mask = rho_phi_d(R) < umbral
        if np.any(mask):
            R_fin_phi = R[mask][0]
        else:
            R_fin_phi = R[-1]

        
        Rho_phi_mod = np.where(R < R_fin_phi, Rho_phi, 0)
        
        rho_phi_d = interp1d(R, Rho_phi_mod)

        # GAMMA CONFIGURATION
        def Gamma(R, nn, kk):
                x = (3 * nn - 8 * kk) / (2 * (4 - nn))
                bb = np.abs(1 - x)
                if x == 0:
                        gamma_val = np.full(len(R), bb * Hub_d(R_RH))
                else:
                        #gamma_val = bb * Hub_d(R_RH) * ((T(R)/Tend)**nn) * ((R/R_RH)**kk)
                        gamma_val = bb * Hub_d(R_RH) * (R_RH/R)**x
                # Apaga Gamma para R >= R_fin_phi
                #gamma_val = np.where(R >= R_fin_phi, 0, gamma_val)
                return gamma_val

        # AXION COSMOLOGY (DARK MATTER)

        #######################
        # AXION MASSES

        q=2
        ma = q * Hub[np.argmin(np.abs(R-R_RH))]* prop**3
        print(f"ma={ma:.2e} GeV")
        ma = interp1d(R, np.full(len(R), ma))
        
        R_osc = R[np.argmin(np.abs(q * Hub - ma(R)))]

        

        def Hdot(R):
                rhophi = rho_phi_d(R)
                rhor = rho_r_d(R)
                hudot = 5 * rhophi + 4 * rhor
                hubsq = rhor + rhophi
                h_prime = hudot / (2 * hubsq)
                return h_prime

        def f1(R, Y):
                Y1, Y2 = Y
                h_term = Hdot(R)
                Mass = ma(R)
                dY1 = Y2
                dY2 = -Y2 * (h_term / R) - (Mass / (Hub_d(R) * R))**2 *Y1
                return [dY1, dY2]

        R_range = [R[0], R_RH * Approx]
        R_eval = R[(R >= R_range[0]) & (R <= R_range[1])]
        Y0 = [theta, 0]

        sol = solve_ivp(
        lambda R, Y: f1(R, Y),
        R_range,
        Y0,
        method='LSODA',
        atol=1e-10,
        rtol=1e-10,
        t_eval=R_eval
        )
        Rs = sol.t
        Rf = R[(Rs[-1] < R) & (R <= R[-1])]  # Posterior a la oscilación
    

        axo = sol.y[0]
        daxo = sol.y[1]

        
        ax_post= np.full(len(Rf), axo[-1])

        dax_post= np.full(len(Rf), 0)

        # Combine ax and ax_post into a single array
        ax = interp1d(R, np.concatenate((axo, ax_post)), fill_value='extrapolate', bounds_error=False)
        dax = interp1d(R, np.concatenate((daxo, dax_post)))




        Rprox = R_RH*Approx
        rho_pre = interp1d(R, fa**2 * ((Hub_d(R) * R * dax(R))**2 + 0.5*ma(R)**2 * ax(R)**2))
        rho_a = np.where(R >= Rprox, rho_pre(Rprox) * (Rprox / R)**3, rho_pre(R))

        P_a = np.where(R <= Rprox, fa**2 * ((Hub_d(R) * R * dax(R))**2 -  0.5*ma(R)**2 * ax(R)**2), 0)

        mask_post_osc = (R > Rprox)
        fluidaprox = True
        if fluidaprox:
                with np.errstate(divide='ignore', invalid='ignore'):


                 
                        wa = P_a / rho_a
                        wa[~np.isfinite(wa)] = 0
                        wass=wa.copy()
               
                        sign_change = np.where(np.diff(np.sign(wa)) != 0)[0]

                        if sign_change.size > 0:
                                idx_zero = sign_change[0] + 1  
                                wa[idx_zero:] = 0
                        mask_post_osc = (R > R_osc)
                        wa_prime = np.gradient(wa, R)
                        
                        cad2 = np.abs(wa - R * wa_prime / (3 * wa + 3))
                        cad2[~np.isfinite(cad2)] = 0
                        #wa[mask_post_osc] = 0
                        
        else:
                wa = P_a / rho_a
                aux = wa.copy()
                wass = wa.copy()
                sign_change = np.where(np.diff(np.sign(wass)) != 0)[0]

                if sign_change.size > 0:
                        idx_zero = sign_change[0] + 1  
                        wass[idx_zero:] = 0
                wa = aux
                wa[R>Rprox]=0
                wa_prime = np.gradient(wa, R)
                
                cad2 = np.abs(wa - R * wa_prime / (3 * wa + 3))
                cad2[~np.isfinite(cad2)] = 0
        def limit_ca2(ca2, R1, R_osc):
                mask = (R1 >= R1[0]) & (R1 <= R_osc)
                avg = np.mean(np.abs(ca2[mask]))
                ca2_limited = np.clip(ca2, -0.9*avg, 0.9*avg)
                ca2_limited = np.where(np.abs(ca2_limited) > avg, np.sign(ca2_limited)*avg, ca2_limited)
                ca2_limited[R1 > R_osc] = 0
                return ca2_limited

        #cad2 = limit_ca2(cad2, R, Rprox)
        #cad2[mask_post_osc] = 0
        wa[R>=R_osc]=0
        cad2=wa
        if CDM:
                wa=np.zeros_like(wa)
                cad2=np.zeros_like(cad2)

        Temp = Temp
        GG = Gamma(R, nn, kk)
        ma=ma(R)
        rho_r=rhoSM(Temp)
        rho_phi=Rho_phi
        Hub=np.sqrt((rho_a+rho_phi+rho_r) / (3 * Mp**2)) 


        def xi(R, Rc, R_RH, x, width=0.05):
                R = np.asarray(R)
                x2 = 8 * x / (2 * x + 3)
      
                mask1= R < Rc
                mask2 = (R >= Rc) & (R <= R_RH)
                mask3 = R > R_RH

                r1=np.full(R[mask1].shape, x)
                r2=np.full(R[mask2].shape, x2)
                r3=np.full(R[mask3].shape, x)

                res = np.concatenate((r1, r2, r3))/4
                return res
        x_i=interp1d(R, xi(R, Rc, R_RH, x))
        Xi= xi(R, Rc, R_RH, x)


        Raux=R
        R = R / R_osc  
        ma_val = float(ma[0]) 
        ma_eV = ma_val * 1e9
        exp = int(np.floor(np.log10(ma_eV)))
        mant = ma_eV / 10**exp
        theta_ini = theta / np.pi






        PLOT=ps
        if PLOT!=1:
                print("You chose not to plot background quantities.")
        else:
                def ploting_background_quantities( show):
                        # If you dont have LaTeX installed, comment the following lines:
                        plt.rcParams['text.usetex'] = True
                        plt.rcParams['font.family'] = 'serif'
                        plt.rcParams.update({
                                "text.usetex": True,
                                "axes.labelsize": 22,
                                "xtick.labelsize": 18,
                                "ytick.labelsize": 18,
                                "legend.fontsize": 18,
                                "axes.titlesize": 20,
                                "font.size": 18
                        })

                        # --- Plot especial para nn != 0 ---
                        if nn != 0:
                                output_dirs = "PLOTS_TESIS"
                                os.makedirs(output_dirs, exist_ok=True)
                                plt.figure(figsize=(8, 6))

                                blus  = '#0033A0'
                                orang = '#FF9900'

                                plt.plot(Raux / R_RH, (-x * np.gradient(Raux, Temp)) * (Temp / Raux) / 4,
                                        label='Numerical', color=blus, lw=2)
                                plt.plot(Raux / R_RH, x_i(Raux),
                                        label='Analytical', color=orang, lw=2, linestyle='--')

                                plt.axvline(Rc / R_RH, linestyle='--', color='red',   label=r'$R_c$')
                                plt.axvline(1,          linestyle='--', color='green', label=r'$R_{\mathrm{RH}}$')

                                plt.ylabel(r'$-x\frac{T}{R}\frac{dR}{dT}$', fontsize=20)
                                plt.xlabel(r'$R/R_{\mathrm{RH}}$',          fontsize=16)
                                plt.xlim(0.009, 1.75)
                                plt.ylim(-5, ((-x * np.gradient(Raux, Temp)) * (Temp / Raux) / 4)
                                        [np.argmin(np.abs(Raux - R_RH / 1.09))])

                                plt.legend(fontsize=16, loc='upper right')
                                plt.tight_layout()
                                filename = "xi_gamma.pdf"
                                plt.savefig(os.path.join(output_dirs, filename), format="pdf")
                                plt.close()

                        # --- Límites de ejes para los plots de background ---
                        plot_limits = {
                                'theta':   {'x': (1 / 2, (R_RH / R_osc) * Approx)},
                                'density': {'x': None, 'y': None},
                                'eos':     {'x': (1 / 2, (R_RH / R_osc) * Approx)},
                                'sound':   {'x': (1 / 1.1, (R_RH / R_osc) * 3), 'y': (0, 6)},
                                'rates':   {'x': (None), 'y': None}
                        }

                        def make_plot( show, filename='', xlim=None, ylim=None, title='', ylabel='', data_fn=None):
                                plt.figure(figsize=(7, 5))
                                if data_fn is not None:
                                        data_fn()
                                plt.axvline(1, linestyle='--', color='red', label=r'$R_{\mathrm{osc}}$')
                                plt.xlabel(r'$R/R_{\mathrm{osc}}$')
                                plt.ylabel(ylabel)
                                plt.xscale('log')
                                if xlim:
                                        plt.xlim(*xlim)
                                if ylim:
                                        plt.ylim(*ylim)
                                plt.legend(fontsize=16)
                                plt.title(title)
                                plt.tight_layout()
                                

                                if show:
                                        outdir = "GammaConstant" if nn == 0 else "GammaT"
                                        os.makedirs(outdir, exist_ok=True)
                                        plt.savefig(os.path.join(outdir, f"{filename}.pdf"))
                                        plt.show()
                                else:
                                        plt.close()

                        def plot_all( show, limits=plot_limits):
                                # theta(R)
                                make_plot(
                                show=show, filename='theta',
                                xlim=limits['theta']['x'],
                                title=r'', ylabel=r'$\theta(R)$',
                                data_fn=lambda: plt.plot(Rs / R_osc, axo, label=r'$\theta$', lw=2)
                                )

                                # Gamma, Hubble, ma/2
                                make_plot(
                                show=show, filename='background_rates',
                                xlim=limits['rates']['x'], ylim=limits['rates']['y'],
                                title='Decay, Hubble and Axion mass', ylabel='Rates',
                                data_fn=lambda: (
                                        plt.loglog(R, GG, label=r'$\Gamma$', lw=2),
                                        plt.loglog(R, Hub, label=r'$H$',     lw=2),
                                        plt.loglog(R, ma / 2, label=r'$m_a/2$', lw=2)
                                )
                                )

                                # Densities
                                make_plot(
                                show=show, filename='densities',
                                xlim=limits['density']['x'], ylim=limits['density']['y'],
                                title='Density evolution', ylabel=r'$\rho_i$',
                                data_fn=lambda: (
                                        plt.loglog(R, rho_r,  label=r'$\rho_r$',  lw=2),
                                        plt.loglog(R, rho_phi,label=r'$\rho_\phi$', lw=2),
                                        plt.loglog(R, rho_a,  label=r'$\rho_a$',  lw=2),
                                        plt.axvline(R_eq / R_osc, linestyle='--', color='blue',  label=r'$R_{\mathrm{eq}}$'),
                                        plt.axvline(R_RH / R_osc, linestyle='--', color='green', label=r'$R_{\mathrm{RH}}$'),
                                        plt.axvline(Rc   / R_osc, linestyle='--', color='orange',label=r'$R_{\mathrm{c}}$')
                                )
                                )

                                # Equation of state
                                make_plot(
                                show=show, filename='equation_of_state',
                                xlim=limits['eos']['x'],
                                title='', ylabel=r'$\omega_a$',
                                data_fn=lambda: (
                                        plt.axhline(0, linestyle='--', color='black', alpha=0.4, zorder=1),
                                        plt.plot(R, wass, label=r'$\omega_a$'),
                                        plt.plot(R, wa,  linestyle="--", label=r'$\langle \omega_a \rangle$')
                                )
                                )

                                # Sound speed
                                make_plot(
                                show=show, filename='sound_speed',
                                xlim=limits['sound']['x'], ylim=limits['sound']['y'],
                                title=r'Adiabatic sound speed $c_a^2$', ylabel=r'$c_a^2$',
                                data_fn=lambda: plt.plot(R, cad2, linestyle='--', color='red', label=r'$c_a^2$')
                                )

                        plot_all( show, limits=plot_limits)
        if PLOT==1:
                ploting_background_quantities( show=True)




        R=Raux

        
        # axionfield
        Field = {
        "ax": [axo,Rs],
        "dax": daxo
        }

        # Cosmological background quantities
        Cosmo = {
        "R": R,
        "Temp": Temp,
        "Hub": Hub,
        "H_RH": H_RH,
        "decay": GG,  # Decay rate
        "ma": ma,
        "rho_r": rho_r,
        "rho_phi": rho_phi,
        "k_rh": k_rh,
        "r_rh": r_rh,
        "Xi": Xi
        }

        # Axion fluid properties
        AxionFluid = {
        "rho_a": rho_a,
        "P_a": P_a,
        "wa": wa,
        "cad2": cad2
        }

        # Characteristic scale factors
        ScaleFactors = {
        "R_osc": R_osc,
        "R_RH": R_RH,
        "R_eq": R_eq,
        "Rc":Rc
        }
        
        dat={
        "Field": Field,
        "Cosmo": Cosmo,
        "AxionFluid": AxionFluid,
        "ScaleFactors": ScaleFactors
        }

        print(R[-1]/R_RH)
        return dat

