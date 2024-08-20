# -*- coding: utf-8 -*-
"""
Mélanie Labiausse

Simulation of action potential in plant cells : main code

"""
import Simulation_functions as fn
import matplotlib.pyplot as plt
# Initialization

# Number of cells along x and y direction
Nx = 15
Ny = 15
# Number and size of time steps
duration = 10 # Duration time of the simulations, in seconds
dt1 = 0.1 # Time-step for intercellular passive membrane potential change (in seconds)
dt2 = 2.5*10**-5 # Time-step for intracellular processes in seconds
dt3 = 0.01 # Time-step for ion diffusion between apoplasts (in seconds)
#Initial membrane potential in millivolts
Em_init = -180
# Initial ion concentrations in M (mol.L^-1). In = in the cytoplasm, out = in the apoplast
K_in_init = 0.165
K_out_init = 3.5*10**-3
K_atom_out_init = 0.082
P_in_init = 7*10**-8
P_out_init = 1*10**-6
H_in_init = 0.015
H_out_init = 2.1*10**-3
Cl_in_init = 2*10**-2
Cl_out_init = 2.7*10**-3
Ca_in_init = 1*10**-7
Ca_out_init = 5*10**-4
# Initial and final temperatures in Kelvins
T_init = 298
T_final = 290
T_droptime = 0 # Time at which the temperature begin to drop
#T_dropwidth = 0

# Computation

Nt = int(duration/dt1)


# result is a list of the lists of all membrane potential at each time
Em, K_in, Cl_in, Ca_in = fn.temporal_evolution(Nx, Ny, Nt, dt1, dt2, dt3, Em_init, K_in_init, K_out_init, H_in_init, H_out_init, P_in_init, P_out_init, Cl_in_init, Cl_out_init, Ca_in_init, Ca_out_init, T_init, T_final, T_droptime = T_droptime)


print("Reshape !")

# Transform the results into a list of matrices for visualization 
Potential = fn.lists_to_matrix(Em, Nx, Ny)
Potassium = fn.lists_to_matrix(K_in, Nx, Ny)
Chlorine = fn.lists_to_matrix(Cl_in, Nx, Ny)
Calcium = fn.lists_to_matrix(Ca_in, Nx, Ny)

# Uncomment to see the diference to initial values

# Potential = [(i - Em_init)*10**4 for i in Potential]
# Potassium = [(i-K_in_init)*10**15 for i in Potassium]
# Chlorine = [(i - Cl_in_init) for i in Chlorine]
# Calcium = [(i-Ca_in_init)*10**16 for i in Calcium]

#E_membrane = fn.list_mean_sub_matrix(Potential, 5)

print("Now, display !")
# Print membrane potential distribution over time
#plt.plot(E_membrane)
#plt.xlabel("Nt")
#plt.ylabel("Membrane potential (mV)")
#fn.animate_multiple(Nt,Potential, Potassium, Chlorine, Calcium)
fn.animate(Nt, Potential)
