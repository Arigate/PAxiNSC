import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, NullFormatter
import os

from EMD_BACK import axion_backsolver_ma
from EMD_PERTS import PertEMD, load_perts

numers=750
maxi=9*int(1e4)
configs = [
    {
        "Tend": 0.005,
        "maxs": 100,
        "nums": [maxi, 10, numers],
        "theta": 0.1 * np.pi,
        "Approx": 1,
        "fa": 1e14,
        "prop": 10,
        "nnkk_list": [[0, 0],[2,4]]
    }
]

s = True
p = False
from master_ploterV2 import plot_delta_grid

pert_names = ["delta_a_data"]
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
        load_perts(Tend, nn, kk, prop, theta, Approx, fa, s, p, maxs, nums)
        for pert_name in pert_names:
            plot_delta_grid(nn, kk, Tend, pert_name,folder='GRIDS/deltadata')







