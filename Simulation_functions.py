# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:22:58 2024

Mélanie Labiausse

Simulation of action potential in plant cells : functions and constants
"""
import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter #for saving
from tqdm import tqdm # for timer

# General constants
F = 96485 #C.mol^-1
R = 8.31 #J.K^-1
g = 1000 #conductance between neighbouring cells (S/m^-2). See S1
L = 10**-5 # Cell surface/Cell volume (m^-1)
V_spec = 0.41667 # V_intracell/V_extracell
#V_in = 3.5933391*10**-10 # Intracellular volume in Liters
#V_out = 8.764242*10**-10 # Extracellular volume in Liters

# Put a stimuli: i.e. impose a higher Em for a small time and see if it spikes into
# an AP or if it repolarizes quickly


def temporal_evolution(Nx, Ny, Nt, dt1, dt2, dt3, Em0, K_in, K_out, H_in, H_out, P_in, P_out, Cl_in, Cl_out, Ca_in, Ca_out, T_init, T_final, T_droptime = 0, T_dropwidth = 'inf'):
    """
    This function takes the initial values of membrane potential
    and ion concentration and iterates over time. It returns the list
    of the lists of parameters of interest distributions over time.
    
    Parameters
    ----------
    Nx : x numerical lengh of the space
    Ny : y numerical lengh of the space
    Nt : number of iterations over time
    dt1 : time interval for intracellular processes
    dt2 : time interval for passive membrane potential change
    dt3 : time interval for ion diffusion between apoplasts
    Em0 : initial membrane potential
    K_in : initial concentration in potassium ions in the cytoplasms
    K_out : initial concentration in potassium ions in the apoplasts
    H_in : initial concentration in protons in the cytoplasms
    H_out : initial concentration in protons in the apoplasts
    Cl_in : initial concentration in chlorine ions in the cytoplasms
    Cl_out : initial concentration in chlorine ions in the apoplasts
    Ca_in : initial concentration in calcium ions in the cytoplasms
    Ca_out : initial concentration in calcium ions in the apoplasts
    T0 : initial temperature
    T_final : final temperature
    T_droptime : 
    T_ dropwidth :
    Returns
    -------
    Em : list of the lists of membrane potential distributions over time
    
    """
    N = Nx*Ny
    # Initialize the membrane potential and the ions as lists of the initial values
    Em = [[Em0]*N]
    K_in, K_out = [[K_in]*N], [[K_out]*N]
    H_in, H_out = [[H_in]*N], [[H_out]*N]
    Cl_in, Cl_out = [[Cl_in]*N], [[Cl_out]*N]
    Ca_in, Ca_out = [[Ca_in]*N], [[Ca_out]*N]
    P_in, P_out = [[P_in]*N], [[P_out]*N]
    T_list = temperature(Nt, dt1, T_init, T_final, T_droptime = T_droptime, T_dropwidth = T_dropwidth)
    # We iterate over time
    for j in tqdm(range(Nt)):
        #Compute the new values of membrane potential and ions concentrations in all cells
        next_Em, next_K_in, next_K_out, next_H_in, next_H_out, next_P_in, next_P_out, next_Cl_in, next_Cl_out, next_Ca_in, next_Ca_out =  spatial_evolution(Nx, Ny, dt1, dt2, dt3, K_in[j], K_out[j], P_out[j], H_in[j], H_out[j], Cl_in[j], Cl_out[j], Ca_in[j], Ca_out[j], Em[j], T_init, T_list[j])
        # We add new values to the list of membrane potential and ion concentrations over time
        Em.append(next_Em)
        K_in.append(next_K_in)
        K_out.append(next_K_out)
        H_in.append(next_H_in)
        H_out.append(next_H_out)
        Cl_in.append(next_Cl_in)
        Cl_out.append(next_Cl_out)
        Ca_in.append(next_Ca_in)
        Ca_out.append(next_Ca_out)
        P_in.append(next_P_in)
        P_out.append(next_P_out)
    return Em, K_in, Cl_in, Ca_in



def spatial_evolution(Nx, Ny, dt1, dt2, dt3, K_in, K_out, P_out,  H_in, H_out, Cl_in, Cl_out, Ca_in, Ca_out, Em, T0,T):
    """
    This function takes the membrane potentials and ion concentrations and
    compute their new values after a time dt. It takes into account the interaction
    between neighboring cells (see S1) and ion diffusion between the apoplasts (see 64).
    ----------
    Nx : x numerical lengh of the space
    Ny : y numerical lengh of the space
    dt1 : time interval for intracellular processes
    dt2 : time interval for passive membrane potential change
    dt3 : time interval for ion diffusion between apoplasts
    K_in : list of concentration in potassium ions in the cytoplasms
    K_out : list of concentration in potassium ions in the apoplasts
    H_in : list of concentration in protons in the cytoplasms
    H_out : list of concentration in protons in the apoplasts
    Cl_in : list of concentration in chlorine ions in the cytoplasms
    Cl_out : list of concentration in chlorine ions in the apoplasts
    Ca_in : list of concentration in calcium ions in the cytoplasms
    Ca_out : list of concentration in calcium ions in the apoplasts
    Em : list of membrane potential of all cells
    T0 : initial temperature
    T_final : final temperature
    """
    N = Nx*Ny # Total number of cells
    a = 10**-4 # Typical length of a cell in m
    B0_in, B0_out = 0.2, 0.0833 # Total concentration of proton buffer in cytoplasm/apoplast (in M)
    Kd_BHin, Kd_BHout, Kd_BKout = 10**-6, 10**-6, 10**-4 # Dissociation constant of proton-buffer and potassium-buffer
    Dr_K, Dr_Cl, Dr_H = 1.96*10**-9, 2.03*10**-9, 7.8*10**-9 #m^2.s^-1, Diffusion coefficients for ions in apoplast
    K_in_next, K_out_next, H_in_next, H_out_next = [0]*N,[0]*N,[0]*N,[0]*N
    Cl_in_next, Cl_out_next, Ca_in_next, Ca_out_next  = [0]*N, [0]*N, [0]*N, [0]*N
    P_in, P_out, P_out_sqrt, K_atom_out = [0]*N, [0]*N, [0]*N, [0]*N
    K_atom_out_next, P_out_next, Em_next = [0]*N, [0]*N, [0]*N
    for i in range(N): # Compute new values for each cell
    # Define buffers of protons in cytoplasm and apoplast
        P_in[i] = (H_in[i] - B0_in - Kd_BHin)/2 + np.sqrt(((H_in[i] - B0_in - Kd_BHin)/2)**2 + Kd_BHin*H_in[i])
        P_out_sqrt[i] = np.sqrt(((H_out[i] + K_atom_out[i] - B0_out - Kd_BHout)*H_out[i]*Kd_BHout/(2*H_out[i]*Kd_BHout + 2*K_atom_out[i]*Kd_BKout))**2 + (H_out[i]*Kd_BHout)**2/(H_out[i]*Kd_BHout + K_atom_out[i]*Kd_BKout))
        P_out[i] = (H_out[i] + K_atom_out[i] - B0_out - Kd_BHout)*H_out[i]*Kd_BHout/(2*H_out[i]*Kd_BHout + 2*K_atom_out[i]*Kd_BKout) + P_out_sqrt[i]
        K_out[i] = P_out[i]*K_atom_out[i]*Kd_BKout/(H_out[i]*Kd_BHout)
    # Define characteristics of each ions channel
        g_in_K, j_in_K, E_K = K_in_channels(K_in[i], K_out[i], Em[i], T)
        g_out_K,j_out_K, E_K = K_out_channels(K_in[i], K_out[i], Em[i], T)
        g_C_H, j_C_H, E_H = P_channels(P_in[i], P_out[i], Em[i], T)
        g_P_Cl, j_P_Cl, E_P_Cl = P_Cl_antiporter(P_in[i], P_out[i], Cl_in[i], Cl_out[i], Em[i], T0, T)
        j_P_K = P_K_symporter(P_in[i], P_out[i], K_in[i], K_out[i], Em[i], T0, T)
        g_C_Ca, j_C_Ca, E_Ca = Ca_channels(Ca_in[i], Ca_out[i], Em[i], T)
        g_C_A, j_C_A, E_Cl = anion_channels(Cl_in[i], Cl_out[i], Ca_in[i], Em[i], T)
        g_P_H, j_P_H, E_P_H = H_ATP_ase(P_in[i], P_out[i], Ca_in[i], Em[i], T0, T)
        g_P_Ca, j_P_Ca, E_P_Ca = Ca_ATP_ase(Ca_in[i], Ca_out[i], P_in[i], P_out[i], Em[i], T0, T)
        # These are two parameters defined in S2 and S3
        gtot = g_C_A + g_C_Ca + g_P_Ca + g_in_K + g_out_K + g_C_H + g_P_Cl + g_P_H
        gEtot = (g_in_K+g_out_K)*E_K + g_C_H*E_H + g_P_Cl*E_P_Cl + g_C_A*E_Cl + g_C_Ca*E_Ca + g_P_Ca*E_P_Ca + g_P_H*E_P_H
    # Use index i to find the location of the cell (corner, border or bulk)
        #mod is a proxy of the borders of our space
        mod = i%Nx
        #We consider the top-left corner
        if ((i < Nx) & (mod==0)):
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i+1]+g*Em[i+Nx]+gEtot)/(gtot+2*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(2*K_out[i] - K_out[i+1] - K_out[i+Nx])/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(2*Cl_out[i] - Cl_out[i+1] - Cl_out[i+Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(2*P_out[i] - P_out[i+1] - P_out[i+Nx])/(a**2*(1+V_spec))
        # We consider the top-right corner
        elif ((i < Nx) & (mod==Nx-1)):
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i-1]+g*Em[i+Nx]+gEtot)/(gtot+2*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(2*K_out[i] - K_out[i-1] - K_out[i+Nx])/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(2*Cl_out[i] - Cl_out[i-1] - Cl_out[i+Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(2*P_out[i] - P_out[i-1] - P_out[i+Nx])/(a**2*(1+V_spec))
        # We consider the first Nx line
        elif i < Nx :
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i-1]+g*Em[i+1]+g*Em[i+Nx]+gEtot)/(gtot+3*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(3*K_out[i] - K_out[i-1] - K_out[i+1] - K_out[i+Nx])/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(3*Cl_out[i] - Cl_out[i-1] - Cl_out[i+1] - Cl_out[i+Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(3*P_out[i] - P_out[i-1] - P_out[i+1] - P_out[i+Nx])/(a**2*(1+V_spec))
        # We consider the bottom-left corner
        elif ((i >= N-Nx) & (mod==0)):
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i+1]+g*Em[i-Nx]+gEtot)/(gtot+2*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(2*K_out[i] - K_out[i+1] - K_out[i-Nx] )/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(2*Cl_out[i] - Cl_out[i+1] - Cl_out[i-Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(2*P_out[i] - P_out[i+1] - P_out[i-Nx])/(a**2*(1+V_spec))
        # We consider the bottom-right corner
        elif ((i >= N-Nx) & (mod==Nx-1)):
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i-1]+g*Em[i-Nx]+gEtot)/(gtot+2*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(2*K_out[i] - K_out[i-1] - K_out[i-Nx] )/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(2*Cl_out[i] - Cl_out[i-1] - Cl_out[i-Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(2*P_out[i] - P_out[i-1] - P_out[i-Nx])/(a**2*(1+V_spec))
        #We consider the last Nx line
        elif i >= N-Nx :
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i-1]+g*Em[i+1]+g*Em[i-Nx]+gEtot)/(gtot+3*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(3*K_out[i] - K_out[i-1] - K_out[i+1] - K_out[i-Nx] )/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(3*Cl_out[i] - Cl_out[i-1] - Cl_out[i+1] - Cl_out[i-Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(3*P_out[i] - P_out[i-1] - P_out[i+1] - P_out[i-Nx])/(a**2*(1+V_spec))
        #We consider the lines to calculate the left border values
        elif mod == 0 :
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i+1]+g*Em[i-Nx]+g*Em[i+Nx]+gEtot)/(gtot+3*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(3*K_out[i] - K_out[i+1] - K_out[i+Nx] - K_out[i-Nx] )/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(3*Cl_out[i] - Cl_out[i+1] - Cl_out[i+Nx] - Cl_out[i-Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(3*P_out[i] - P_out[i+1] - P_out[i+Nx] - P_out[i-Nx])/(a**2*(1+V_spec))
        #We consider the lines to calculate the right border values
        elif mod == Nx-1 :
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i-1]+g*Em[i-Nx]+g*Em[i+Nx]+gEtot)/(gtot+3*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(3*K_out[i] - K_out[i-1] - K_out[i+Nx] - K_out[i-Nx] )/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(3*Cl_out[i] - Cl_out[i-1] - Cl_out[i+Nx] - Cl_out[i-Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(3*P_out[i] - P_out[i-1] - P_out[i+Nx] - P_out[i-Nx])/(a**2*(1+V_spec))
        #We consider all the other cells (the bulk ones)
        else:
            # Update values for membrane potential and apoplast ions concentrations
            Em_next[i] = (g*Em[i-1]+g*Em[i+1]+g*Em[i-Nx]+g*Em[i+Nx]+gEtot)/(gtot+4*g)
            K_out_next[i] = K_out[i] + dt3*Dr_K*(4*K_out[i] - K_out[i-1] - K_out[i+1] - K_out[i+Nx] - K_out[i-Nx])/(a**2*(1+V_spec))
            Cl_out_next[i] = Cl_out[i] - dt1*L*V_spec*(j_C_A + j_P_Cl) + dt3*Dr_Cl*(4*Cl_out[i] - Cl_out[i-1] - Cl_out[i+1] - Cl_out[i+Nx] - Cl_out[i-Nx])/(a**2*(1+V_spec))
            P_out_next[i] = P_out[i] + dt3*Dr_H*(4*P_out[i] - P_out[i-1] - P_out[i+1] - P_out[i+Nx] - P_out[i-Nx])/(a**2*(1+V_spec))
        # Update ions concentrations inside cells
        K_atom_out_next[i] = K_atom_out[i] - dt1*L*V_spec*(j_in_K + j_out_K + j_P_K)
        K_in_next[i] = K_in[i] + dt1*L*(j_in_K + j_out_K + j_P_K)
        H_in_next[i] = H_in[i] + dt1*L*(j_C_H + j_P_H + 2*j_P_Cl - j_P_Ca - j_P_K)
        H_out_next[i] = H_out[i] - dt1*L*V_spec*(j_C_H + j_P_H + 2*j_P_Cl - j_P_Ca - j_P_K)
        Cl_in_next[i] = Cl_in[i] + dt1*L*(j_C_A + j_P_Cl)
        Ca_in_next[i] = Ca_in[i] - dt1*L*(j_C_Ca + j_P_Ca)
        Ca_out_next[i] = 5.10**-4
    return Em_next, K_in_next, K_out_next, H_in_next, H_out_next, P_in, P_out_next, Cl_in_next, Cl_out_next, Ca_in_next, Ca_out_next



# Calculation of ions fluxes 

# Inward potassium channels, See S16
def K_in_channels(K_in, K_out, Em, T): 
    """
    This function describes the effect of the inward-potassium channels on the
    conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    K_in : concentration in potassium ions in the cytoplasm
    K_out : concentration in potassium ions in the apoplast
    Em : membrane potential of the cell
    T : temperature
    
    Returns
    -------
    g_in_K : conductance of the inward-potassium channels
    j_in_K : ions flux of the inward-potassium channels
    E_K : Nernst potential of potassium ions
    
    """
    # Em is a scalar correponding to the membrane potential of the ij-cell
    # So the function returns g_in_K_ij and j_in_K_ij
    P_K_max = 2.9*10**-8 #m.s^-1
    C0 = 106.4843 # J.mV-1
    E0 = -190 #mV
    u = F*Em*0.001/(R*T)
    p0 = 1/(1+np.exp((C0*E0-Em*C0)/(R*T)))
    if (K_out<=0) :
        K_out = 10**-20
    if (K_in<=0):
        K_in = 10**-20
    E_K = (R*T/F)*np.log(K_out/K_in)
    j_in_K = p0*P_K_max*u*(K_in-K_out*np.exp(-u))/(1-np.exp(-u)) #See S16
    i_in_K = F*j_in_K
    g_in_K = i_in_K/(Em*0.001-E_K)
    return g_in_K, j_in_K, E_K

# Outward potassium channels, See S22
def K_out_channels(K_in, K_out, Em, T):
    """
    This function describes the effect of the outward-potassium channels on the
    conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    K_in : concentration in potassium ions in the cytoplasm
    K_out : concentration in potassium ions in the apoplast
    Em : membrane potential of the cell
    T : temperature
    
    Returns
    -------
    g_out_K : conductance of the outward-potassium channels
    j_out_K : ions flux of the outward-potassium channels
    E_K : Nernst potential of potassium ions
    
    """
    P_K_max = 2.9*10**-8 #m.s^-1
    C0 = 108.9607 # J.mV-1
    E0 = -65 #mV
    u = F*Em*0.001/(R*T)
    p0 = 1/(1+np.exp((C0*E0-Em*C0)/(R*T)))
    if (K_out<=0) :
        K_out = 10**-20
    if (K_in<=0):
        K_in = 10**-20
    E_K = (R*T/F)*np.log(K_out/K_in)
    j_out_K = p0*P_K_max*u*(K_in-K_out*np.exp(-u))/(1-np.exp(-u)) #See S16
    i_out_K = F*j_out_K
    g_out_K = i_out_K/(Em*0.001-E_K)
    return g_out_K, j_out_K, E_K

# H+ channels, See S34
def P_channels(P_in, P_out, Em, T):
    """
    This function describes the effect of the hydrogen channels (proton leakage) on the
    conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    P_in : concentration in protons in the cytoplasm
    P_out : concentration in protons in the apoplast
    Em : membrane potential of the cell
    T : temperature
    
    Returns
    -------
    g_in_K : conductance of the hydrogen channels
    j_in_K : ions flux of the hydrogen channels
    E_K : Nernst potential of protons
    
    """
    P = 10**-5
    u = F*Em*0.001/(R*T)
    if (P_out<=0) :
        P_out = 10**-20
    if (P_in<=0):
        P_in = 10**-20
    E_H = (R*T/F)*np.log(P_out/P_in)
    j_H = P*u*(P_in-P_out*np.exp(-u))/(1-np.exp(-u))
    i_H = F*j_H
    g_H = i_H/(Em*0.001-E_H)
    return g_H, j_H , E_H# g_H, j_H

# H+_K+ symporter, See S52
def P_K_symporter(P_in, P_out, K_in, K_out, Em, T0, T):
    """
    This function describes the effect of the hydrogen-potassium symporter on
    the conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    P_in : concentration in protons in the cytoplasm
    P_out : concentration in protons in the apoplast
    K_in : concentration in potassium ions in the cytoplasm
    K_out : concentration in potassium ions in the apoplast
    Em : membrane potential of the cell
    T0 : initial temperature
    T : temperature
    
    Returns
    -------
    j_P_K : ions flux of the hydrogen-potassium symporter
    
    """
    V_K = 0.015 #s^-1.M^-1
    I_T = 3**(0.1*(T0-T))
    j_P_K = I_T*V_K*(K_in*P_out - K_out*P_in)
    return j_P_K #J_P_K

# 2H+_Cl- antiporter, see S48
def P_Cl_antiporter(P_in, P_out, Cl_in, Cl_out, Em, T0, T):
    """
    This function describes the effect of the hydrogen-chlorine antiporter
    on the conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    P_in : concentration in protons in the cytoplasm
    P_out : concentration in protons in the apoplast
    Cl_in : concentration in chlorine ions in the cytoplasm
    Cl_out : concentration in chlorine ions in the apoplast
    Em : membrane potential of the cell
    T0 : initial temperature
    T : temperature
    
    Returns
    -------
    g_P_Cl : conductance of the hydrogen-chlorine antiporter
    j_P_Cl: ions flux of the hydrogen-chlorine antiporter
    E_P_Cl : Nernst potential of proton-chlorine
    
    """
    V_Cl = 20000 #s^-1.M^-2
    I_T = 3**(0.1*(T0-T))
    u = F*Em*0.001/(R*T)
    E_P_Cl = (R*T/F)*np.log((Cl_out*P_out**2)/(P_in**2*Cl_in))
    j_P_Cl = I_T*V_Cl*u*(Cl_in*P_in**2-Cl_out*P_out**2*np.exp(-u))/(1-np.exp(-u))
    i_P_Cl = F*j_P_Cl
    g_P_Cl = i_P_Cl/(Em*0.001-E_P_Cl)
    return g_P_Cl, j_P_Cl, E_P_Cl #g_P_Cl, j_P_Cl

# Ca2+ channels, See S12
def Ca_channels(Ca_in, Ca_out, Em, T):
    """
    This function describes the effect of the calcium channels on
    the conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    Ca_in : concentration in calcium ions in the cytoplasm
    Ca_out : concentration in calcium ions in the apoplast
    Em : membrane potential of the cell
    T : temperature
    
    Returns
    -------
    g_Ca : conductance of the calcium channels
    j_Ca: ions flux of the calcium channels
    E_Ca : Nernst potential of calcium ions
    
    """
    P_Ca_max = 1.5*10**-9
    u = 2*F*Em*0.001/(R*T)
    E_Ca = (R*T/F)*np.log(Ca_out/Ca_in)
    p0 = Proba(Em, T)
    j_Ca = p0*P_Ca_max*u*(Ca_in-Ca_out*np.exp(-u))/(1-np.exp(-u))
    i_Ca = 2*F*j_Ca
    g_Ca = i_Ca/(Em*0.001-E_Ca)
    return g_Ca, j_Ca, E_Ca

# Ca-dependent anion channels, See S27
def anion_channels(Cl_in, Cl_out, Ca_in, Em, T):
    """
    This function describes the effect of the calcium-dependent anion channel
    on the conductance and potential of the membrane of one cell. The only anion 
    considered here is chlorine.
    
    Parameters
    ----------
    Cl_in : concentration in chlorine ions in the cytoplasm
    Cl_out : concentration in chlorine ions in the apoplast
    Ca_in : concentration in calcium ions in the cytoplasm
    Em : membrane potential of the cell
    T : temperature
    
    Returns
    -------
    g_Cl : conductance of the anion channels
    j_Cl: ions flux of the anion channels
    E_Cl : Nernst potential of chlorine ions
    
    """
    P_Cl_max = 2.45*10**-8
    C0 = 247.638 # J.mV-1
    E0 = -120 #mV
    Kd = 8*10**-6 # M. Dissociation constant
    u = F*Em*0.001/(R*T)
    p0 = 1/(1+np.exp((C0*E0-Em*C0)/(R*T)))
    A_Ca = (Ca_in**2/(Ca_in**2+Kd**2))*((2.1*10**-6)**2/((2.1*10**-6)**2+Kd**2))**(-1)
    E_Cl = (R*T/F)*np.log(Cl_out/Cl_in)
    j_Cl = p0*A_Ca*P_Cl_max*u*(Cl_in-Cl_out*np.exp(-u))/(1-np.exp(-u))
    i_Cl = F*j_Cl
    g_Cl = i_Cl/(Em*0.001-E_Cl)
    return g_Cl, j_Cl, E_Cl

# H-ATP-ase and its regulation, See S37
def H_ATP_ase(P_in, P_out, Ca_in, Em, T0, T):
    """
    This function describes the effect of proton-ATP-ase channels on
    the conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    P_in : concentration in protons in the cytoplasm
    P_out : concentration in protons in the apoplast
    Ca_in : concentration in calcium ions in the cytoplasm
    Em : membrane potential of the cell
    T0 : initial temperature
    T : temperature
    
    Returns
    -------
    g_P_H : conductance of the H-ATP-ase channels
    j_P_H : ions flux of the H-ATP-ase channels
    E_P_H : Nernst potential of H-ATP-ase protons
    
    """
    G_ATP = -50000 # J.mol^-1, energy of ATP hydrolysis
    k_1 = 4.5*10**-2 # s^-1
    k_2 = 2.58*10**-5 #s^-1
    Kd = 4*10**-7
    zeta = 1
    u = F*Em*0.001/(R*T)
    if (P_out<=0) :
        P_out = 10*-20
    if (P_in<=0):
        P_in = 10**-20
    k_1_plus = k_1*P_in
    k_1_minus = k_1*np.exp(G_ATP/(R*T))
    k_2_plus = k_2*u/(1-np.exp(-u))
    k_2_minus = k_2*P_out*u*np.exp(-u)/(1-np.exp(-u))
    I_Ca = (Kd**2/(Ca_in**2+Kd**2))*((Kd)**2/((1*10**-7)**2+Kd**2))**(-1)
    I_T = 3**(0.1*(T0-T))
    j_P_H = zeta*I_Ca*I_T*(k_1_plus*k_2_plus-k_1_minus*k_2_minus/(k_1_plus + k_2_plus + k_1_minus + k_2_minus))
    i_P_H = F*j_P_H
    E_P_H = G_ATP/F + (R*T/F)*np.log(P_out/P_in)
    g_P_H = i_P_H/(Em*0.001-E_P_H)
    return g_P_H, j_P_H, E_P_H

# Ca-ATP-ase and its regulation, See S43

def Ca_ATP_ase(Ca_in, Ca_out, P_in, P_out, Em, T0, T):
    """
    This function describes the effect of calcium-ATP-ase channels on
    the conductance and potential of the membrane of one cell.
    
    Parameters
    ----------
    Ca_in : concentration in calcium ions in the cytoplasm
    Ca_out : concentration in calcium ions in the apoplast
    P_in : concentration in protons in the cytoplasm
    P_out : concentration in protons in the apoplast
    Em : membrane potential of the cell
    T0 : initial temperature
    T : temperature
    
    Returns
    -------
    g_P_Ca : conductance of the Ca-ATP-ase channels
    j_P_Ca : ions flux of the Ca-ATP-ase channels
    E_P_Ca : Nernst potential of Ca-ATP-ase calcium ions
    
    """
    G_ATP = -50000 # J.mol^-1, energy of ATP hydrolysis
    k_1 = 28.35 # s^-1
    k_2 = 1.62779*10**-5 #s^-1
    u = F*Em*0.001/(R*T)
    k_1_plus = k_1*Ca_in*P_out
    k_1_minus = k_1*np.exp(G_ATP/(R*T))
    k_2_plus = k_2*u/(1-np.exp(-u))
    k_2_minus = k_2*Ca_out*P_in*u*np.exp(-u)/(1-np.exp(-u))
    I_T = 3**(0.1*(T0-T))
    if (P_out<0) :
        P_out = 10**-20
    if (P_in<=0):
        P_in = 10**-20
    j_P_Ca = I_T*(k_1_plus*k_2_plus-k_1_minus*k_2_minus/(k_1_plus + k_2_plus + k_1_minus + k_2_minus))
    i_P_Ca = F*j_P_Ca
    E_P_Ca = G_ATP/F + (R*T/F)*np.log((Ca_out*P_in)/(Ca_in*P_out))
    g_P_Ca = i_P_Ca/(Em*0.001-E_P_Ca)
    return g_P_Ca, j_P_Ca, E_P_Ca

# Gradient of temperature 

def temperature(Nt, dt, T_init, T_final, T_droptime = 0, T_dropwidth = 'inf'):
    """
    This function creates a customizable gradient of temperature over time. 
    
    Parameters
    ----------
    Nt : number of time-steps
    dt : time interval
    T_init : initial temperature
    T_final = final temperature
    T_droptime : time at which the temperature begins to drop (length of 1st plateau)
    T_dropwidth : width over which the gradient is created
    Returns
    -------
    T_list : list of temperatures
    
    """
    T_drop = int(T_droptime/dt)
    T_plateau = [T_init]*T_drop # T = T0 for 0 < time < T_time
    if T_dropwidth == 'inf':
        T_width = Nt-T_drop
    else :
        T_width = int(T_dropwidth/dt)
    if (T_drop + T_width >= Nt):
        T_width = Nt-T_drop
    T_slope = np.linspace(T_init,T_final,T_width).tolist() # Then, T decreases towards T_final
    T_plateau_bis = [T_final]*(Nt - T_drop - T_width)
    T_list = T_plateau + T_slope + T_plateau_bis
    return T_list


# Probabilities for potential-dependent calcium channels

def Proba(Em, T):
    """
    This function computes the probability of having the 
    calcium ion channels in an open state.
    
    Parameters
    ----------
    Em : membrane potential of the cell
    T : temperature
    
    Returns
    -------
    p0 : probability of open state
    
    """
    # Em is a scalar correponding to the membrane potential of the ij-cell at a given time: EM_ij(t)
    Co = 111.437 #J/mV
    Ci = 123.819 #J/mV
    Ei = -180.6 #mV
    Eo = -90 #mV
    ko = 353 #Hz
    ki = 0.6 #Hz
    k_plus_o = ko*np.exp(Em*Co/(2*R*T))
    k_minus_o = ko*np.exp((2*Co*Eo - Co*Em)/(2*R*T))
    k_plus_i = ki*np.exp(Em*Ci/(2*R*T))
    k_minus_i = ki*np.exp((2*Ci*Ei - Ci*Em)/(2*R*T))
    #Defining constant add up
    constant = [0, k_plus_o]
    # Defining transition matrix
    M = [[-k_minus_i, k_plus_i], [k_minus_i - k_plus_o, -k_plus_o -k_minus_o -k_plus_i]]
    inv_M = np.linalg.inv(M)
    p_eq = -np.dot(inv_M,constant)
    p0 = p_eq[1]
    return p0


# Functions for display

#This function reshape the list of list of potentials into a list of matrices 
#of potential distribution with a shape of Ny*Nx

def lists_to_matrix(lists,Nx,Ny):
    """
    
    Parameters
    ----------
    lists : A list over time of the lists of potential distributions
    Nx : the x numerical lengh of the space
    Ny : the y numerical lengh of the space

    Returns
    -------
    P : list over time of the matrices of the potential distributions

    """
    #We initialise the list of matrices with P an empty list
    P = []
    #We iterate over the lists of temperatures
    for val in tqdm(lists) :
        #We reshape each list into a matrix of shape Ny*Nx
        matrix = np.reshape(val, (Ny, Nx))
        #We add this matrix to P
        P.append(matrix)
    #We return P as a list over time of the matrices of 
    #the temperature distributions
    return P


def list_mean_sub_matrix(P, margin):
    res = []
    for i in range(len(P)):
        matrix = P[0]
        sub_matrix = []
        margin = int(margin)
        nb_rows= np.shape(matrix)[0]
        for i in range(nb_rows):
            line = matrix[i]
            sub_line = line[margin:-margin]
            sub_matrix.append(sub_line)
        sub_matrix = sub_matrix[margin:-margin]
        mean = np.mean(sub_matrix)
        res.append(mean)
    return res

#Animate a list af matrices through imshow

def animate(Nt,P):
    """

    Parameters
    ----------
    Nt : number of iterations over time
    P : list over time of the matrices of the potential distributions

    Returns
    -------
    None.

    """
    #We iterate over time
    for i in range(Nt):
        #We show the i-th temperature distribution
        plt.imshow(P[i],aspect = "auto")
        #We show the colorbar
        plt.colorbar(label = "Membrane potential (a.u.)")
        plt.xlabel("x")
        plt.ylabel("y")
        #We add a small break to give enough time to observe
        plt.pause(0.5) 
    plt.show()
    

def animate_multiple(Nt,P1, P2, P3, P4):
    """

    Parameters
    ----------
    Nt : number of iterations over time
    P : list over time of the matrices of the potential distributions

    Returns
    -------
    None.

    """
    #We iterate over time
    for i in range(0, Nt, 10):
        #We show the i-th temperature distribution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('N = %s' % i)
        im1 = ax1.imshow(P1[i],aspect = "auto")#, vmin = 0, vmax = 2.75)
        cax1 = plt.axes((0.44, 0.57, 0.02, 0.31))
        plt.colorbar(im1, format="%.3f", cax = cax1)
        ax1.set_title('Membrane potential (mV)')
        im2 = ax2.imshow(P2[i]*10**3,aspect = "auto")#, vmin = 0, vmax =175)
        cax2 = plt.axes((0.91, 0.57, 0.02, 0.31))
        plt.colorbar(im2, format="%.3f", cax = cax2)
        ax2.set_title('K+ cytoplasms (mM)')
        im3 = ax3.imshow(P3[i]*10**3,aspect = "auto")#, vmin = 17, vmax = 25)
        cax3 = plt.axes((0.44, 0.12, 0.02, 0.31))
        plt.colorbar(im3, format="%.3f", cax = cax3)
        ax3.set_title('Cl- cytoplasms (mM)')
        im4 = ax4.imshow(P4[i]*10**6,aspect = "auto")#, vmin = 0, vmax = 7)
        cax4 = plt.axes((0.91, 0.12, 0.02, 0.31))
        plt.colorbar(im4, format="%.3f", cax = cax4)
        ax4.set_title('Ca2+ cytoplasms (µM)')
        #We show the colorbar
        plt.subplots_adjust(wspace = 0.6, hspace = 0.5)
        #We add a small break to give enough time to observe
        plt.pause(0.1)
    plt.show()
    
# Save the results as an mp4 file  
def mp4_anim(T, parameters, path_to_ffmpeg = "None", title = "Diffusion", fps = 15, dpi = 100, step = 1):
    """
    mp4_anim is a visualisation function that saves the result as an mp4 video.

    Parameters
    ----------
    T : A list over time of the lists of temperature distributions (and fluxes if so)
    parameters : a dictionary containing the parameters information
    path_to_ffmpeg : the path to an ffmpeg.exe file. The default is "None".
    title : Title of the saved file. The default is "Diffusion".
    fps : Number of frames per second. The default is 15.
    dpi : Resolution of the video in dpi. The default is 100.
    step : Step of the time iteration for the video. The default is 1.

    Returns
    -------
    None.

    """
    # try:
    #     '''To plot in an external window'''
    #     shell = IPython.get_ipython()
    #     shell.enable_matplotlib(gui='qt')#'qt' or 'inline'
    # except:
    #     pass
    # # We reshape T into a list over time of matrices of the temperatures
    # T = lists_to_matrix(T, parameters['nx'],parameters['ny'])
    # #We update the path to ffmpeg.exe if given
    # if not(path_to_ffmpeg == "None"):
    #     plt.rcParams["animation.ffmpeg_path"] = path_to_ffmpeg
    #We start a figure
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    #We calculate the x axis
    x = np.linspace(0, parameters["nx"],10)
    x_labels = x*parameters["dx"]
    x_labels = [f"{val:.1f}" for val in x_labels]
    #We calculate the y axis
    y = np.linspace(0, parameters["ny"],6)
    y_labels = y*parameters["dy"]
    y_labels = [f"{val:.1f}" for val in y_labels]
    #we set the x and y axis
    plt.xticks(x,x_labels)
    plt.yticks(y,y_labels)
    #we set the video title
    metadata = {"title" : title}
    #We define the writer type
    writer = FFMpegWriter(fps = fps, metadata = metadata)
    temp = 0
    with writer.saving(fig, "diffusion.mp4",dpi):
        #We iterate over time
        for i in range(0, parameters["nt"], step):
            #We print a progression percentage
            if int(i/parameters["nt"]*100) != temp :
                temp = int(i/parameters["nt"]*100)
                print(temp, "%")
            #im is the picture at time i*dt
            im = ax.imshow(T[i],cmap = "rainbow")
            #We set up the colorbar
            if i == 0:   
                cbar = fig.colorbar(im, label = "Temperature (°C)") #label = "Temperature (°C)"
            cbar.vmax= np.max(T[i])
            cbar.vmin= np.min(T[i])
            #We add the frame to the video
            writer.grab_frame()

