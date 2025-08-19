def PertEMD(rho_phi, rho_r, rho_a, Hub, GammaPhi, nn, kk, R, Rc, R_osc, R_RH,Rk, k, wa, cad2,ma,maxs,Xi):

    #Packages
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.integrate import solve_ivp
    from scipy.optimize import root_scalar
    import warnings
    import matplotlib.pyplot as plt

    #CALL ARRAYS

    Mp = 2.4e18 




 
    propors=R_RH*100

    ma_i=interp1d(R,ma)
    rho_r_i=interp1d(R,rho_r)
    rho_a_i=interp1d(R,rho_a)
    rho_phi_i=interp1d(R,rho_phi)
    E_i=interp1d(R,np.sqrt((rho_phi + rho_r + rho_a) / (3 * Mp**2)))
    Gamma_phi_i=interp1d(R,GammaPhi)

    import matplotlib.pyplot as plt
    epsilon = 1e-50
    wa_i=interp1d(R,wa)
    c_ad2=interp1d(R,cad2)
    R_k=Rk




    xi=interp1d(R, Xi)



    wak = wa_i(R_k)




    # PERTS EQUATIONS


    

    def perturbations(R, k, Y):
       
            delta_phi, theta_phi, delta_r, theta_r, delta_a, theta_a, Phi = Y




            def axions(R, k1, E, delta_a, Phi, theta_a,cad2,wa):
                

                cad2=wa
                denom = 1 + wa
                if denom == 0 or np.isnan(denom) or np.isinf(denom):
                    was = 0
                else:
                    was = 1 / denom
    

                cs2=k1**2 / (k1**2 + 4 * ma**2 * R**2 + epsilon)


                ddelta_a_dR = (3 / R) * (wa - cs2) * delta_a - (((9 * E / k1**2 + epsilon) * (cs2 - cad2) + (1 / (E * R**2 + epsilon)))* theta_a*(1+wa)) +3* dPhi_dR*(1+wa)

                dtheta_a_dR = ((3 * cs2 - 1) / R) * theta_a + ((k1**2) / (E * R**2 + epsilon)) * (cs2 * delta_a+Phi*(1+wa))*was
                return dtheta_a_dR, ddelta_a_dR

            
            ma=ma_i(R)
            wa=wa_i(R)
            cad2=c_ad2(R)
            rho_phi_val = rho_phi_i(R)
            rho_r_val = rho_r_i(R)
            rho_a_val = rho_a_i(R)
            DeltaGam=delta_r*xi(R)


         
            rho_phi = rho_phi_val
            rho_r = rho_r_val
            rho_a = rho_a_val

            E = E_i(R)
            Gamma_phi = Gamma_phi_i(R)
            
            
            if np.abs(rho_phi)<=1e-29:
                Gam_ratio = 0
                rho_del_phi =0
            else:
                Gam_ratio = (rho_phi / rho_r) * (Gamma_phi / (R * E))
                rho_del_phi = rho_phi * delta_phi   

            inv_R2E=1/(R**2*E)
            k2_R2E=k**2*inv_R2E

            


            dPhi_dR = -((1 / ((6*Mp**2)*R* E**2)) * (rho_del_phi + rho_r * delta_r+rho_a*delta_a ) + ((k**2 )/ (3 * R**3 * E**2) + 1/R ) *Phi)

            ddelta_phi_dR = -(Gamma_phi / (R * E)) * (Phi + DeltaGam) - inv_R2E * theta_phi + 3 * dPhi_dR
            dtheta_phi_dR = k2_R2E * Phi - (1 / R) * theta_phi

            dtheta_r_dR = Gam_ratio * (3 / 4 * theta_phi - theta_r) + k2_R2E* (delta_r / 4 + Phi)
            ddelta_r_dR = Gam_ratio * (delta_phi - delta_r + Phi+DeltaGam) - ((4/3)*inv_R2E) * theta_r + 4 * dPhi_dR


            dtheta_a_dR, ddelta_a_dR = axions(R,k, E,delta_a,Phi,theta_a,cad2,wa)
            






            return [ddelta_phi_dR, dtheta_phi_dR, ddelta_r_dR, dtheta_r_dR, ddelta_a_dR, dtheta_a_dR, dPhi_dR]


    
    def initial_conditions_BF(k):
        Rini=Rk
        Rmax = R_RH*maxs
        R_range = (Rini, Rmax)
        Phi_ini=  10**(-4.5)
        theta_ini_factor =2 / 3 * Phi_ini*np.sqrt(Rini)

        Delta_phi_ini = -2*Phi_ini
        if nn==0 and kk==0:
            Delta_r_ini=-Phi_ini
        else:
            Delta_r_ini = -2*Phi_ini 
        
        Delta_a_ini= (Delta_phi_ini*(1+wak))
        theta_a_ini= theta_ini_factor * k**2
        theta_r_ini = theta_ini_factor * k**2
        theta_phi_ini = theta_ini_factor * k**2

     
        
    
        Y2=[Delta_phi_ini, theta_phi_ini, Delta_r_ini, theta_r_ini, Delta_a_ini, theta_a_ini, Phi_ini]
        return Y2, R_range

    #SOLVING THE SYSTEM

    if R_k < R_osc:
        Y0, R_range = initial_conditions_BF(k)


    else:
        Y0, R_range = initial_conditions_BF(k)
 
    R_bf = R[(R < R_range[0]) & (R >= R[0])]         # Before horizon entry
    R_int = R[(R >= R_range[0]) & (R <= R_range[-1])]  # Within the integration range


    delta_a_bf = np.full(len(R_bf), np.abs(Y0[4]))
    delta_r_bf = np.full(len(R_bf), np.abs(Y0[2]))
    Phi_bf     = np.full(len(R_bf), np.abs(Y0[6]))
    delta_phi_bf = np.full(len(R_bf), np.abs(Y0[0]))

    sols = solve_ivp(
        lambda RR, Y: perturbations(RR, k, Y),
        [R_range[0], R_range[-1]],
        Y0,
        method='LSODA',
        atol=1e-8,
        rtol=1e-8,
        #t_eval=R_int,
    )


    from scipy.interpolate import interp1d

    delta_a_af = np.abs(interp1d(sols.t, sols.y[4], kind='cubic', fill_value="extrapolate")(R_int))
    delta_r_af = np.abs(interp1d(sols.t, sols.y[2], kind='cubic', fill_value="extrapolate")(R_int))
    Phi_af     = np.abs(interp1d(sols.t, sols.y[6], kind='cubic', fill_value="extrapolate")(R_int))
    delta_phi_af = np.abs(interp1d(sols.t, sols.y[0], kind='cubic', fill_value="extrapolate")(R_int))

    delta_a = np.concatenate((delta_a_bf, delta_a_af))
    delta_r = np.concatenate((delta_r_bf, delta_r_af))
    Phi     = np.concatenate((Phi_bf, Phi_af))
    delta_phi = np.concatenate((delta_phi_bf, delta_phi_af))
    Rfinal  = np.concatenate((R_bf, R_int))
    
    return delta_a, delta_r,delta_phi, Phi, Rfinal

def worker(args):
    import numpy as np
    (Rk_val, k,rho_phi, rho_r, rho_a, Hub, GG, nn, kk, R, Rc, R_osc, R_RH, wa, cad2, ma, maxs, k_osc, k_RH,Xi) = args
    delta_a, delta_r, delta_phi,Phi, Rp = PertEMD(rho_phi, rho_r, rho_a, Hub,GG, nn, kk,R, Rc, R_osc, R_RH,Rk_val, k, wa, cad2, ma, maxs,Xi)
    if np.isclose(k, k_osc, atol=1e-12):
        label = r"$k_{\mathrm{osc}}$"
    else:
        ratio = k / k_RH
        label = rf"$k = {ratio:.2f}\cdot k_{{\mathrm{{RH}}}}$"
    return delta_a, delta_r, delta_phi,Phi, Rp, label

def load_perts(Tend, nn, kk, prop, theta, Approx, fa, s, p, maxs, nums):
    import numpy as np
    from EMD_BACK import axion_backsolver_ma
    import matplotlib.pyplot as plt
    import os

    from scipy.interpolate import interp1d
    from scipy.optimize import root_scalar
    dat = axion_backsolver_ma(Tend, nn, kk, prop, theta, Approx, fa, s, p)
    print("Background solved successfully.")
  
    ax, Rs = dat["Field"]["ax"]
    dax = dat["Field"]["dax"]

    R = dat["Cosmo"]["R"]
    Temp = dat["Cosmo"]["Temp"]
    Hub = dat["Cosmo"]["Hub"]
    H_RH = dat["Cosmo"]["H_RH"]
    GG = dat["Cosmo"]["decay"]
    ma = dat["Cosmo"]["ma"]
    rho_r = dat["Cosmo"]["rho_r"]
    rho_phi = dat["Cosmo"]["rho_phi"]

    rho_a = dat["AxionFluid"]["rho_a"]
    P_a = dat["AxionFluid"]["P_a"]
    wa = dat["AxionFluid"]["wa"]
    cad2 = dat["AxionFluid"]["cad2"]

    R_osc = dat["ScaleFactors"]["R_osc"]
    R_eq = dat["ScaleFactors"]["R_eq"]
    R_RH = dat["ScaleFactors"]["R_RH"]
    Rc = dat["ScaleFactors"]["Rc"]



    k_osc = ma[0] * R_osc/2
    k_RH = Hub[np.argmin(np.abs(R - R_RH))] * R_RH
    r_rh=R_RH
    Xi = dat["Cosmo"]["Xi"]  
    delta_a_all = []
    delta_r_all = []
    Phi_all = []
    Rp_all = []
    labels_k = []
    multimax=nums[0]
    multimin=nums[1]
    numbers=nums[2]

    R_RH_times_bf=r_rh/multimax
    R_RH_times_af=r_rh*multimin
    R_RH_times = np.logspace(np.log10(R_RH_times_bf), np.log10(R_RH_times_af), numbers)


    k_values=[]
    Rk=[]
    for R_RH_times in R_RH_times:
        k_values.append(Hub[np.argmin(np.abs(R - R_RH_times))] * R_RH_times)
        Rk.append(R_RH_times)

    





    from tqdm import tqdm  



   
    print(f"Primer k: {k_values[0]:.3e} ({k_values[0]/k_RH:.2f} x k_RH, {k_values[0]/k_osc:.2f} x k_osc)")
    print(f"Último k: {k_values[-1]:.3e} ({k_values[-1]/k_RH:.2f} x k_RH, {k_values[-1]/k_osc:.2f} x k_osc)")
    import os
    import concurrent.futures
    import numpy as np
    import threading

    num_cores = os.cpu_count()
    n_workers = max(1, int(num_cores * 0.7)) # for now THIS MUST BE CONFIGURED MANNUALLY
    print(f"Usando {n_workers} núcleos.")


    delta_a_all = []
    delta_r_all = []
    Phi_all = []
    delta_phi_all = []
    Rp_all = []
    labels_k = []

    args_list = [
        (Rk_val, k,rho_phi, rho_r, rho_a, Hub, GG, nn, kk, R, Rc, R_osc, R_RH, wa, cad2, ma, maxs, k_osc, k_RH,Xi)
        for Rk_val, k in zip(Rk, k_values)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(worker, args_list),
            total=len(args_list),
            desc="Calculando perturbaciones"
        ))

    for delta_a, delta_r, delta_phi,Phi, Rp, label in results:
        delta_a_all.append(delta_a)
        delta_r_all.append(delta_r)
        delta_phi_all.append(delta_phi)
        Phi_all.append(Phi)
        Rp_all.append(Rp)
        labels_k.append(label)

        # Save data if s == True


    if s:
        folder = "GRIDS"
        base_folder = folder
        i = 1

        folder = base_folder
        data_folder = os.path.join(folder, "deltadata")
        os.makedirs(data_folder, exist_ok=True)


        np.savetxt(os.path.join(data_folder, "delta_a_data.txt"),
                   np.array(delta_a_all))
        np.savetxt(os.path.join(data_folder, "delta_r_data.txt"),
                   np.array(delta_r_all))
        np.savetxt(os.path.join(data_folder, "Phi_data.txt"),
                   np.array(Phi_all))
        np.savetxt(os.path.join(data_folder, "delta_phi_data.txt"),
                   np.array(delta_phi_all))
        np.savetxt(os.path.join(data_folder, "R_pert.txt"), Rp)
        np.savetxt(os.path.join(data_folder, "k_values.txt"), k_values)

        # Guardar variables del diccionario dat
        np.savetxt(os.path.join(data_folder, "ax.txt"), dat["Field"]["ax"][0])
        np.savetxt(os.path.join(data_folder, "Rs.txt"), dat["Field"]["ax"][1])
        np.savetxt(os.path.join(data_folder, "dax.txt"), dat["Field"]["dax"])
        np.savetxt(os.path.join(data_folder, "R.txt"), dat["Cosmo"]["R"])
        np.savetxt(os.path.join(data_folder, "Temp.txt"), dat["Cosmo"]["Temp"])
        np.savetxt(os.path.join(data_folder, "Hub.txt"), dat["Cosmo"]["Hub"])
        np.savetxt(os.path.join(data_folder, "H_RH.txt"), [dat["Cosmo"]["H_RH"]])
        np.savetxt(os.path.join(data_folder, "decay.txt"), dat["Cosmo"]["decay"])
        np.savetxt(os.path.join(data_folder, "ma.txt"), dat["Cosmo"]["ma"])
        np.savetxt(os.path.join(data_folder, "rho_r.txt"), dat["Cosmo"]["rho_r"])
        np.savetxt(os.path.join(data_folder, "rho_phi.txt"), dat["Cosmo"]["rho_phi"])
        np.savetxt(os.path.join(data_folder, "rho_a.txt"), dat["AxionFluid"]["rho_a"])
        np.savetxt(os.path.join(data_folder, "P_a.txt"), dat["AxionFluid"]["P_a"])
        np.savetxt(os.path.join(data_folder, "wa.txt"), dat["AxionFluid"]["wa"])
        np.savetxt(os.path.join(data_folder, "cad2.txt"), dat["AxionFluid"]["cad2"])
        np.savetxt(os.path.join(data_folder, "R_osc.txt"), [dat["ScaleFactors"]["R_osc"]])
        np.savetxt(os.path.join(data_folder, "R_eq.txt"), [dat["ScaleFactors"]["R_eq"]])
        np.savetxt(os.path.join(data_folder, "R_RH.txt"), [dat["ScaleFactors"]["R_RH"]])
        np.savetxt(os.path.join(data_folder, "Rc.txt"), [dat["ScaleFactors"]["Rc"]])

    

        import shutil


        dest_folder = f"{nn}_{kk}_{Tend}"
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder) 
        shutil.copytree(data_folder, dest_folder)





    # Plotting - for now just delta_a
    if p:
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels_k):
            plt.loglog(Rp/R_RH, delta_a_all[i], label=label)
        plt.ylabel(r"$\delta_a$")
        plt.title(r"Evolution of $\delta_a$ for different $k$ values")
        plt.axvline(R_osc/R_RH, linestyle='--', color='red', label=r'$R_{\mathrm{osc}}$')
        plt.axvline(1, linestyle='--', color='green', label=r'$R_{\mathrm{RH}}$')
        plt.legend()
        plt.xlim(1e-4, 1e2)
        plt.ylim(1e-6, 1)
        plt.tight_layout()
        if s:
            folder = "GammaConstant" if nn == 0 else "GammaT"
            os.makedirs(folder, exist_ok=True)
            filename = "Perts_GG_cte.pdf" if nn == 0 else "Perts_GG_T.pdf"
            plt.savefig(f"{folder}/{filename}", dpi=300)
            plt.close()

    return delta_a_all,Rp,k_values
