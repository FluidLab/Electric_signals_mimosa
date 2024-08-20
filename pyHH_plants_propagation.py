# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:50:51 2024
Mélanie Labiausse


"""
import numpy as np
import warnings

class HHModel:
    """The HHModel tracks conductances of 5 channels to calculate Vm"""

    class Gate:
        """The Gate object manages a channel's kinetics and open state"""
        alpha, beta, state = 0, 0, 0

        def update(self, deltaTms):
            alphaState = self.alpha * (1-self.state)
            betaState = self.beta * self.state
            self.state += deltaTms * (alphaState - betaState)

        def setInfiniteState(self):
            self.state = self.alpha / (self.alpha + self.beta)

    E_K, E_Cl, E_sy, E_pu = -0.1, 0.1, 0.02, -0.4 #V
    g_Ki, g_Ko, g_Cl, g_sy, g_pu , g_plasma = 1, 1, 0.9, 1, 1, 0.05 #S.cm^2
    m_Ko, h_Ki, m_Cl, h_Cl, h_sy, h_pu = Gate(), Gate(), Gate(), Gate(), Gate(), Gate()
    C = 2 #µF/cm^2
    F = 96485 # C.mol^-1
    R = 8.31
    T = 293 # K
    u = F/(2*R*T)

    def __init__(self,  nb_cells, startingVoltage=-0.15):
        for i in range(nb_cells):
            setattr(self, f'Vm{i}', startingVoltage)
        self._UpdateGateTimeConstants(startingVoltage)
        self.m_Ko.setInfiniteState()
        self.h_Ki.setInfiniteState()
        self.m_Cl.setInfiniteState()
        self.h_Cl.setInfiniteState()
        self.h_sy.setInfiniteState()
        self.h_pu.setInfiniteState()
        self.Vm = startingVoltage
        
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

    def _UpdateCellVoltage(self, stimulusCurrent, Cl_conductance, deltaTms):
        """calculate channel currents using the latest gate time constants"""
        self.I_Cl = self.m_Cl.state * Cl_conductance * self.h_Cl.state*(self.Vm-self.E_Cl)
        self.I_Ki = self.h_Ki.state * self.g_Ki * (self.Vm-self.E_K)
        self.I_Ko = self.m_Ko.state * self.g_Ko * (self.Vm-self.E_K)
        self.I_pu = self.g_pu * self.h_pu.state * (self.Vm-self.E_pu)
        self.I_sy = self.g_sy * self.h_sy.state * (self.Vm-self.E_sy)
        I_sum = stimulusCurrent - self.I_Cl - self.I_Ki - self.I_Ko - self.I_pu - self.I_sy
        self.Vm += deltaTms * I_sum / self.C

    def _UpdateGateStates(self, deltaTms):
        """calculate new channel open states using latest Vm"""
        self.m_Ko.update(deltaTms)
        self.h_Ki.update(deltaTms)
        self.m_Cl.update(deltaTms)
        self.h_Cl.update(deltaTms)
        self.h_sy.update(deltaTms)
        self.h_pu.update(deltaTms)

    def iterate(self, stimulusCurrent, Cl_conductance, deltaTms):
        self._UpdateGateTimeConstants(self.Vm)
        self._UpdateCellVoltage(stimulusCurrent, Cl_conductance, deltaTms)
        self._UpdateGateStates(deltaTms)

        
class Simulation:

    def __init__(self, model):
        self.model = model
        self.CreateArrays(0, 0,0)
        pass

    def CreateArrays(self, pointCount, deltaTms, nb_cells):
        self.times = np.arange(pointCount) * deltaTms
        self.V = np.empty(pointCount)
        #self.Vm1 = np.empty(pointCount)

    def Run(self, stimulusWaveform, Cl_conductance, stepSizeMs, nb_cells):
        if (stepSizeMs > 0.05):
            warnings.warn("step sizes < 0.05 ms are recommended")
        assert isinstance(stimulusWaveform, np.ndarray)
        self.CreateArrays(len(stimulusWaveform), stepSizeMs, nb_cells)
        print(f"simulating {len(stimulusWaveform)} time points...")
        for i in range(len(stimulusWaveform)):
        print("simulation complete")