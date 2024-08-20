# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:50:51 2024
Mélanie Labiausse


"""
import numpy as np
import warnings

class Gate:
    alpha, beta, state = 0, 0, 0
    
    def update(self, deltaTms):
        alphaState = self.alpha * (1-self.state)
        betaState = self.beta * self.state
        self.state += deltaTms * (alphaState - betaState)
        
    def setInfiniteState(self):
        self.state = self.alpha / (self.alpha + self.beta)
       
class HHModel(Gate):
    """The HHModel tracks conductances of 5 channels to calculate Vm"""
    g_Ki, g_Ko, g_Cl, g_sy, g_pu, g_Ca = 1, 1, 0.9, 1, 1, 1 #S.m^2
    C = 2 #µF/m^2
    F = 96485 # C.mol^-1
    R = 8.31
    T = 293 # K
    u = F/(2*R*T)
    E_K, E_Cl, E_sy, E_pu, E_Ca = -0.1, 0.1, 0.02, -0.4, -0.3 #V
    
    def __init__(self, startingVoltage=-0.15):
        self.Vm = startingVoltage
        self.m_Ko, self.h_Ki, self.m_Cl, self.h_Cl,self.h_sy, self.h_pu, self.h_Ca, self.m_Ca = Gate(), Gate(), Gate(), Gate(), Gate(), Gate(), Gate(), Gate()
        self._UpdateGateTimeConstants(startingVoltage)
        self.m_Ko.setInfiniteState()
        self.h_Ca.setInfiniteState()
        self.h_Ki.setInfiniteState()
        self.m_Cl.setInfiniteState()
        self.h_Cl.setInfiniteState()
        self.h_sy.setInfiniteState()
        self.h_pu.setInfiniteState()
        self.m_Ca.setInfiniteState()
        pass

    def _UpdateGateTimeConstants(self, Vm):
        """Update time constants of all gates based on the given Vm"""
        self.m_Ko.alpha = 10*np.exp(self.u*Vm)
        self.m_Ko.beta = 1*np.exp(-self.u*Vm)
        self.h_Ki.alpha = 1*np.exp(-self.u*Vm)
        self.h_Ki.beta = 100*np.exp(self.u*Vm)
        self.m_Cl.alpha = 50*np.exp(self.u*Vm)
        self.m_Cl.beta = 0.5*np.exp(-self.u*Vm)
        self.h_Cl.alpha = 0.1*np.exp(-self.u*Vm)
        self.h_Cl.beta = 1*np.exp(self.u*Vm)
        self.h_sy.alpha = 0.015*np.exp(-self.u*Vm)
        self.h_sy.beta = 1*np.exp(self.u*Vm)
        self.h_pu.alpha = 1*np.exp(-self.u*Vm)
        self.h_pu.beta = 40*np.exp(self.u*Vm)
        self.m_Ca.alpha = 353*np.exp(Vm*111437/(2*self.R*self.T))
        self.m_Ca.beta = 353*np.exp((2*(-0.09)*111437 - 111437*Vm)/(2*self.R*self.T))
        self.h_Ca.alpha = 0.6*np.exp((2*(-0.1806)*123819 - 123819*Vm)/(2*self.R*self.T))
        self.h_Ca.beta = 0.6*np.exp(123819*Vm/(2*self.R*self.T))

    def _UpdateCellVoltage(self, stimulusCurrent, Cl_conductance, Ca_conductance, deltaTms):
        """calculate channel currents using the latest gate time constants"""
        self.I_Cl = self.m_Cl.state * Cl_conductance * self.h_Cl.state*(self.Vm-self.E_Cl)
        self.I_Ki = self.h_Ki.state * self.g_Ki * (self.Vm-self.E_K)
        self.I_Ko = self.m_Ko.state * self.g_Ko * (self.Vm-self.E_K)
        self.I_pu = self.g_pu * self.h_pu.state * (self.Vm-self.E_pu)
        self.I_sy = self.g_sy * self.h_sy.state * (self.Vm-self.E_sy)
        self.I_Ca = Ca_conductance * self.m_Ca.state * self.h_Ca.state * (self.Vm - self.E_Ca)
        I_sum = stimulusCurrent - self.I_Cl - self.I_Ki - self.I_Ko - self.I_pu - self.I_sy - self.I_Ca
        self.Vm += deltaTms * I_sum / self.C

    def _UpdateGateStates(self, deltaTms):
        """calculate new channel open states using latest Vm"""
        self.m_Ko.update(deltaTms)
        self.h_Ki.update(deltaTms)
        self.m_Cl.update(deltaTms)
        self.h_Cl.update(deltaTms)
        self.h_sy.update(deltaTms)
        self.h_pu.update(deltaTms)

    def iterate(self, stimulusCurrent, Cl_conductance, Ca_conductance, deltaTms):
        self._UpdateGateTimeConstants(self.Vm)
        self._UpdateCellVoltage(stimulusCurrent, Cl_conductance, Ca_conductance, deltaTms)
        self._UpdateGateStates(deltaTms)

        
class Simulation:

    def __init__(self, model):
        self.model = model
        self.CreateArrays(0, 0)
        pass

    def CreateArrays(self, pointCount, deltaTms):
        self.times = np.arange(pointCount) * deltaTms
        self.Vm = np.empty(pointCount)
        #for i in range(nb_cells):
         #   globals()[f"self.Vm{i}"] = np.empty(pointCount)
        self.I_Cl = np.empty(pointCount)
        self.I_Ki = np.empty(pointCount)
        self.I_Ko = np.empty(pointCount)
        self.I_pu = np.empty(pointCount)
        self.I_sy = np.empty(pointCount)
        self.StateMKO = np.empty(pointCount)
        self.StateHKI = np.empty(pointCount)
        self.StateMCL = np.empty(pointCount)
        self.StateHCL = np.empty(pointCount)
        self.StateHSY = np.empty(pointCount)
        self.StateHPU = np.empty(pointCount)

    def Run(self, stimulusWaveform, Cl_conductance, Ca_conductance, stepSizeMs):
        if (stepSizeMs > 0.05):
            warnings.warn("step sizes < 0.05 ms are recommended")
        assert isinstance(stimulusWaveform, np.ndarray)
        self.CreateArrays(len(stimulusWaveform), stepSizeMs)
        #print(f"simulating {len(stimulusWaveform)} time points...")
        for i in range(len(stimulusWaveform)):
            self.model.iterate(stimulusWaveform[i], Cl_conductance[i], Ca_conductance[i], stepSizeMs)
            self.Vm[i] = self.model.Vm
            self.I_Cl[i] = self.model.I_Cl
            self.I_Ki[i] = self.model.I_Ki
            self.I_Ko[i] = self.model.I_Ko
            self.I_pu[i] = self.model.I_pu
            self.I_sy[i] = self.model.I_sy
            self.StateMKO[i] = self.model.m_Ko.state
            self.StateHKI[i] = self.model.h_Ki.state
            self.StateMCL[i] = self.model.m_Cl.state
            self.StateHCL[i] = self.model.h_Cl.state
            self.StateHSY[i] = self.model.h_sy.state
            self.StateHPU[i] = self.model.h_pu.state
        #print("simulation complete")