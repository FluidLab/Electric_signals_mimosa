# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:53:45 2024
Mélanie Labiausse


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyHH_plants_single_cell as HH
from tqdm import tqdm

# Plot single cell with different stimuli
model = HH.HHModel()
# customize a stimulus waveform
N = 9
g_Cl_stim = np.linspace(1, 3, N)
#I_stim = np.linspace(0.01, 0.19, N)
cmap = mpl.colormaps['viridis']
index_color = 0
# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, N))
plt.figure()
for i in tqdm(g_Cl_stim):
    stim = np.zeros(50000)
    #stim[10000:11000] = i
    g_Cl = np.full((50000) ,0.9)
    g_Cl[10000:11000] = i
    sim = HH.Simulation(model)
    sim.Run(stim, g_Cl, 0.001)
    plt.plot(sim.times, sim.Vm*1000, color = colors[index_color], label = r'$g_{Cl}$ = %.2f' %i)
    index_color = index_color + 1

plt.ylim((-190, 20))
plt.xlabel('Time (s)', fontsize = 16)
plt.ylabel('Membrane potential (mV)', fontsize = 15)
#plt.title("Single cell")
plt.vlines(11, -180, 15,'r', linestyles='dashed', linewidths = 1)
plt.vlines(10, -180, 15,'r', linestyles='dashed', linewidths = 1)
plt.legend(loc = "upper right", fontsize=12)
#plt.legend()
plt.show()

#%%

import pyHH_plants as HH
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Plot two cells for different stimuli
nb_cells = 1
model = HH.HHModel()
L = [HH.Simulation(model)]*nb_cells
# customize a stimulus waveform
N = 5
g_Cl_stim = np.linspace(1, 4.5, N)
#I_stim = np.linspace(0.01, 0.19, N)
cmap = mpl.colormaps['viridis']
index_color = 0
# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, N))
plt.figure()
for i in g_Cl_stim :
    stim = np.zeros(50000)
    #stim[10000:11000] = i
    g_Cl = np.full((50000) ,0.9)
    g_Cl[10000:11000] = i
    #sim = HH.Simulation(model)
    HH.Run_all_cells(L, stimulusWaveform = stim, Cl_conductance= g_Cl, stepSizeMs=0.001)
    y = L[0].Vm*1000
    plt.plot(L[0].times, y, color = colors[index_color], label = 'g_Cl = %.2f' %i)
    index_color = index_color + 1
    
plt.ylim((-200, 30))
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)')
plt.title('Multiple cells')
plt.legend()
plt.show()
#%%

## 
#AP Propagation
##
# Fix a stimuli and plot multiple cells
import pyHH_plants as HH
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Initialize simulation
nb_cells = 5
L = [HH.Simulation(HH.HHModel()) for i in range(nb_cells)]
# customize a stimulus waveform
stim = np.zeros(50000)
#stim[10000:11000] = 0.05
N = 5
g_Cl_stim = np.linspace(1, 4.5, N)

# Choose a colormap
cmap = mpl.colormaps['viridis']

# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, nb_cells))
for i in g_Cl_stim :
    index_color = 0
    g_Cl = np.full((50000) , 0.9)
    g_Cl[10000:11000] = 0.9
    HH.Run_all_cells(L, stimulusWaveform = stim, Cl_conductance = g_Cl, stepSizeMs = 0.001)
    # Plot all cells for given g_Cl
    plt.figure()
    for j in range(nb_cells):
        y = L[j].Vm*1000
        plt.plot(L[j].times, y, color = colors[index_color], label = 'cell N° %.0f' %j)
        index_color = index_color + 1
    plt.ylim((-180, -50))
    plt.vlines(11, -200, 30, 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane potential (mV)')
    plt.title('Multiple cells (g = %.2f)' %i)
    plt.legend()
    plt.show()


#%% Plot a calcium cell for different stimuli

import pyHH_single_cell_calcium as SCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from tqdm import tqdm
# Initialize simulation

g_Cl_stim = np.linspace(1, 2, 26)
#g_Ca_stim = np.linspace(1, 4, 25)
I_stim = np.linspace(0, 0.07, 29)

Result = np.full((26*29), False)
cpt = 0
for i,j in tqdm(itertools.product(I_stim, g_Cl_stim)):
    stim = np.zeros(40000)
    g_Cl = np.full((40000) ,0.9)
    g_Cl[20000:21000] = j
    g_Ca = np.full((40000) ,1)
    #g_Ca[20000:21000] = i
    stim[20000:21000] = i
    sim = SCA.Simulation(SCA.HHModel())
    sim.Run(stim, g_Cl, g_Ca, 0.001)
    # plt.figure()
    # plt.plot(sim.times, sim.Vm*1000)
    # plt.title("g_Cl = %.2f, I_stim = %.2f" %(i,j))
    #print(cpt)
    if (np.argmax(sim.Vm) > 20999):
        Result[cpt] = True
    cpt = cpt+1  
#%
plt.figure()
x,y = np.meshgrid(g_Cl_stim, I_stim*1000)  
#x1, y1 = [1, 1.7], [51, 1]
col = np.where(Result == True,'r','k').tolist()
plt.scatter(x,y, c = col, s = 15)
#plt.plot(x1, y1)
plt.xlabel(r"$g_{\mathbf{Cl}} (S.m^{-2})$", fontsize = 16)
#plt.ylabel(r"$g_{\mathbf{Ca}} (S.m^{-2})$", fontsize = 16)
plt.ylabel(r"$I_{\mathbf{stim}} (mA.m^{-2})$", fontsize = 16)
plt.title(r"Cell with Ca$^{2+}$")
plt.show()


#%% Plot a normal cell for different stimuli

import pyHH_plants_single_cell as SC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from tqdm import tqdm
# Initialize simulation

g_Cl_stim = np.linspace(1, 2, 26)
#g_Ca_stim = np.linspace(1, 4, 25)
I_stim = np.linspace(0, 0.07, 29)

Result = np.full((26*29), False)
cpt = 0
for i,j in tqdm(itertools.product(I_stim, g_Cl_stim)):
    stim = np.zeros(40000)
    g_Cl = np.full((40000) ,0.9)
    g_Cl[20000:21000] = j
    #g_Ca = np.full((40000) ,1)
    #g_Ca[20000:21000] = i
    stim[20000:21000] = i
    sim = SC.Simulation(SC.HHModel())
    sim.Run(stim, g_Cl, 0.001)
    # plt.figure()
    # plt.plot(sim.times, sim.Vm*1000)
    # plt.title("g_Cl = %.2f, I_stim = %.2f" %(i,j))
    #print(cpt)
    if (np.argmax(sim.Vm) > 20999):
        Result[cpt] = True
    cpt = cpt+1  
#%%
plt.figure()
x,y = np.meshgrid(g_Cl_stim, I_stim*1000)  
x1, y1 = [1, 1.7], [51, 1]
col = np.where(Result == True,'r','k').tolist()
plt.scatter(x,y, c = col, s = 15)
plt.plot(x1, y1, label = "y = -71 x + 122")
plt.xlabel(r"$g_{\mathbf{Cl}} (S.m^{-2})$", fontsize = 16)
plt.ylabel(r"$I_{\mathbf{stim}} (mA.m^{-2})$", fontsize = 16)
plt.title(r"Cell without Ca$^{2+}$")
plt.legend(shadow = True, framealpha = 1, fontsize = 12)
plt.show()