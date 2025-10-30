import numpy as np
from EMD_PERTS import load_perts
from master_ploterV2 import plot_delta_grid
import os

numers=300
maxi=int(1e3)
configs = [
    {
        "Tend": 0.005,
        "maxs": 150,
        "nums": [maxi, 10, numers],
        "theta": 0.1 * np.pi,
        "Approx": 1,
        "fa": 1e9,
        "prop": 10,
        "nnkk_list": [[2,4],[0,0],[3,2],[1,2]]
    }
]

pert_names = ["delta_a_data", "delta_r_data", "delta_phi_data", "Phi_data"]
s = True
q1=input("Its already calculated? (y/n): ")
if q1=='y':
    print("Plotting Perturbations . . .")
    for cfg in configs:
        Tend = cfg["Tend"]
        for nn, kk in cfg["nnkk_list"]:
            folder = os.path.join("Output", f"{nn}_{kk}_{Tend}", "deltadata")
            for pert_name in pert_names:
                plot_delta_grid(nn, kk, Tend, pert_name, folder=folder)
else:
    pconf=input("Plot Backgrounds? (y/n): ")
    if pconf=='y':
        p=1
    else:
        p = 0
    pp=input("Plot Perturbations? (y/n): ")
    if pp=='y':
        pp=True
    else:
        pp = False



    for cfg in configs:
        Tend = cfg["Tend"]
        maxs = cfg["maxs"]
        nums = cfg["nums"]
        theta = cfg["theta"]
        Approx = cfg["Approx"]
        fa = cfg["fa"]
        prop = cfg["prop"]
        for nn, kk in cfg["nnkk_list"]:
            x = (3 * nn - 8 * kk) / (2 * (4 - nn))
            print(f'Case x= {x} for nn= {nn} kk= {kk} Starting . . .')
            load_perts(Tend, nn, kk, prop, theta, Approx, fa, s, p, maxs, nums,CDM=False)
    print("All data saved!")

    if pp:
        print("Plotting Perturbations . . .")

        for cfg in configs:
            Tend = cfg["Tend"]
            for nn, kk in cfg["nnkk_list"]:
                folder = os.path.join("Output", f"{nn}_{kk}_{Tend}", "deltadata")
                for pert_name in pert_names:
                    plot_delta_grid(nn, kk, Tend, pert_name, folder=folder)






