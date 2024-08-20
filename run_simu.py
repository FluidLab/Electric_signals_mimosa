# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:40:53 2024
"""

import pyHH_plants as HH
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Initialize simulation
nb_cells = 5
L = [HH.Simulation(HH.HHModel()) for i in range(nb_cells)]

# customize a stimulus waveform
stim = np.zeros(50000)
g_Cl = np.full((50000) , 0.9)
g_Cl[10000:11000] = 1.8
g_max = max(g_Cl)

# Choose a colormap
cmap = mpl.colormaps['viridis']
index_color = 0
# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, nb_cells))

# Run simulation
HH.Run_all_cells(L, stimulusWaveform = stim, Cl_conductance = g_Cl, stepSizeMs = 0.001)

# Plot Vm(t) for all cells
plt.figure()
for i in range(nb_cells):
    y = L[i].Vm*1000
    y_fit = y[40000:50000]
    t_fit = L[i].times[40000:50000]
    p = np.polyfit(t_fit, y_fit, 1)
    y = y - p[1]
    #if i != 0:
     #   y = y*5*i
    plt.plot(L[i].times, y, color = colors[index_color], label = 'cell N° %.0f' %i)
    index_color = index_color + 1

plt.vlines(11,-50,120,'r', linestyles='dashed', linewidths = 1)
plt.vlines(10,-50,120,'r', linestyles='dashed', linewidths = 1)
#plt.ylim((-200, 0))
#plt.xlim(10, 50)
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)')
plt.title('Multiple cells (g = %.2f)' %g_max)
plt.legend()
plt.show()