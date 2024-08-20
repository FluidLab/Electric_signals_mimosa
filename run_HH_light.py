# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:54:12 2024

@author: melan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyHH_single_cell_light as HH
from tqdm import tqdm

# Plot single cell with different stimuli

# customize a stimulus waveform
N = 6
Nt = 45000
#g_Cl_stim = np.linspace(1, 4, N)
#I_stim = np.linspace(0, 0.25, N)
L_stim = np.linspace(250, 750, N)
k_light = np.linspace(0.0001, 0.0006, 8)

fig, ax = plt.subplots(2,4, sharex = True, sharey = True)
#fig.suptitle(r"Single cell with light ($g_{\mathbf{Cl}}$ = 1)")
fig.supxlabel('Time (s)', fontsize = 16)
fig.supylabel('Membrane potential (mV)', fontsize = 16)
cpt_row = 0
cpt_col = 0
for j in tqdm(k_light) :
    model = HH.HHModel()
    cmap = mpl.colormaps['viridis']
    index_color = 0
    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, N))
    for i in L_stim:
        stim = np.zeros(Nt)
        L = np.full((Nt), 250)
        L[15000:16000] = i
        #stim[10000:11000] = i
        g_Cl = np.full((Nt) ,1)
        #g_Cl[10000:11000] = i
        model.k_light = j
        sim = HH.Simulation(model)
        sim.Run(stim, g_Cl, L, 0.001)
        ax[cpt_row, cpt_col].plot(sim.times, sim.Vm*1000, color = colors[index_color], label = r'L = %.0f $W.m^{-2}$' %i)
        index_color = index_color + 1
    
    ax[cpt_row, cpt_col].set_ylim(-200, 30)
    #ax[cpt_row, cpt_col].title("Single cell")
    #ax[cpt_row, cpt_col].vlines(15, -180, -20,'r', linestyles='dashed', linewidths = 1)
    ax[cpt_row, cpt_col].vlines(16, -180, -20,'r', linestyles='dashed', linewidths = 1)
    ax[cpt_row, cpt_col].text(0,0, r"$\mathbf{k_{L}}$ = %.1e" %j)
    #ax[cpt_row, cpt_col].legend()
    cpt_col = cpt_col+1
    if (cpt_col%4 == 0):
        cpt_row = cpt_row + 1
        cpt_col = 0

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'center right', bbox_to_anchor=(1.3, 0.5), fontsize = 14)
plt.show()



# model = HH.HHModel()

# L = np.full((40000), 250)
# L[20000:21000] = 280
# stim = np.zeros(40000)
# stim[10000:11000] = 0
# g_Cl = np.full((40000) ,0.9)
# #g_Cl[10000:11000] = i
# sim = HH.Simulation(model)
# sim.Run(stim, g_Cl, L, 0.001)
# plt.plot(sim.times, sim.Vm*1000)