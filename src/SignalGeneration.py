"""
SignalGeneration.py

Module containing classes:
    - SignalGeneration: Class for generating signals from a particle in a trap
    - Readout: Class describing some readout chain parameters 
"""

import numpy as np
from src.BaseTrap import BaseTrap
import src.utils as utils
from src.Particle import Particle
from src.CircularWaveguide import CircularWaveguide
import scipy.constants as sc
from scipy.optimize import fsolve


class Readout:
    def __init__(self, sampleRate: float, fL0: float, noiseTemp: float):
        """
        Constructor for Readout class

        Parameters:
        -----------
            sampleRate (float): Sample rate in Hz
            fL0 (float): Local oscillator frequency in Hz
            noiseTemp (float): Noise temperature in Kelvin
        """
        self.__sampleRate = sampleRate
        self.__fL0 = fL0
        self.__noiseTemp = noiseTemp

    def GetLOOutput(self, t):
        """
        Calculate local oscillator output at a given time

        Parameters:
        -----------
            t: Time in seconds
        """

        return np.cos(2 * np.pi * self.__fL0 * t)

    def GetSampleRate(self) -> float:
        """
        Getter for sample rate

        Returns:
        --------
            float: Sample rate in Hertz
        """
        return self.__sampleRate

    def GetNoiseTemp(self) -> float:
        """
        Getter for noise temperature

        Returns:
        --------
            float: Noise temperature in Kelvin
        """
        return self.__noiseTemp

    def GetLOFrequency(self) -> float:
        """
        Getter for local oscillator frequency

        Returns:
        --------
            float: Local oscillator frequency in Hz
        """
        return self.__fL0


class SignalGeneration:
    def __init__(self, electron: Particle, trap: BaseTrap, wg: CircularWaveguide,
                 tSignal: float, readout: Readout, receiverPos: np.ndarray):
        """
        Constructor for SignalGeneration class

        Parameters:
        -----------
            electron (Particle): Particle object
            trap (BaseTrap): Trap object
            wg (CircularWaveguide): Circular waveguide object
            tSignal (float): Signal time in seconds
            readout (Readout): Readout object
            receiverPosition (np.ndarray): Receiver position in metres
        """
        self.__electron = electron
        self.__trap = trap
        self.__wg = wg
        self.__t = tSignal
        self.__readout = readout
        self.__receiverPos = receiverPos

        # Begin by calculating a few important parameters
        normFactor = self.__wg.CalcNormalisationFactor()
        v0 = self.__electron.GetSpeed()     # m/s
        p0 = self.__electron.GetMomentum()  # kg m/s
        paInit = self.__electron.GetPitchAngle()  # Pitch angle in radians
        magMomentInit = utils.EquivalentMagneticMoment(
            p0, paInit, self.__trap.GetBzPosition(np.array([0.0, 0.0, 0.0])))

        # Given the motion of the particle we want to calculate a list of
        # retarded times
        # Initially we want to sample at 10 times the digitizer rate
        timeFine = np.arange(
            0, tSignal, 1 / (10 * self.__readout.GetSampleRate()))
        tRet = np.zeros_like(timeFine)
        for iT, T in enumerate(timeFine):
            def func(te): return T - te - np.linalg.norm(self.__receiverPos -
                                                         np.array([1e-5 * np.ones_like(te), np.zeros_like(te), self.__trap.GetZPosTime(te, v0, paInit)])) / sc.c
            tRet[iT] = fsolve(func, T)

        # We ultimately need the electron velocity and position at these times
        BzRet = self.__trap.GetBzTime(tRet, paInit, v0)
        paRet = utils.PitchAngleFromField(BzRet, p0, magMomentInit)
        phasesRet = self.__trap.GetCyclotronPhase(tRet, v0, paInit)
        eVelRet = utils.ElectronVelocity(v0, paRet, phasesRet)
        initialPos = electron.GetPosition()
        ePosRet = np.array([initialPos[0] * np.ones_like(tRet),
                            initialPos[1] * np.ones_like(tRet),
                            self.__trap.GetZPosTime(tRet, v0, paInit)])

        # Do waveguide calculations for the TE11 mode(s)
        # Calculate the impedance of the waveguide mode
        Z = self.__wg.CalcTE11Impedance(self.__trap.CalcOmega0(v0, paInit))
        wgField1 = self.__wg.EFieldTE11Pos_1(ePosRet, normFactor)
        wgField2 = self.__wg.EFieldTE11Pos_2(ePosRet, normFactor)
        self.amp1 = -sc.e * np.einsum('ij,ij->j', wgField1, eVelRet) * -Z / 2
        self.amp2 = -sc.e * np.einsum('ij,ij->j', wgField2, eVelRet) * -Z / 2

        # Now add some noise to the signals
        sigmaNoise = np.sqrt(sc.k * self.__readout.GetNoiseTemp()
                             * (self.__readout.GetSampleRate() * 10.0) / 2)
        self.amp1 += np.random.normal(0, sigmaNoise,
                                      len(timeFine)) * np.sqrt(Z)
        self.amp2 += np.random.normal(0, sigmaNoise,
                                      len(timeFine)) * np.sqrt(Z)

        # Downmix with local oscillator
        self.amp1 *= self.__readout.GetLOOutput(timeFine)
        self.amp2 *= self.__readout.GetLOOutput(timeFine)
        # Now apply a low pass filter to the data
        cutoffFreq = self.__readout.GetSampleRate() / 2
        filterOrder = 6
        self.amp1 = utils.ButterLowPassFilter(self.amp1, cutoffFreq,
                                              self.__readout.GetSampleRate() * 10, filterOrder)
        self.amp2 = utils.ButterLowPassFilter(self.amp2, cutoffFreq,
                                              self.__readout.GetSampleRate() * 10, filterOrder)
        # Keep every 10th element
        self.amp1 = self.amp1[::10]
        self.amp2 = self.amp2[::10]
        # Divide by the square root of the mode impedance to give units of sqrt(W)
        self.amp1 /= np.sqrt(Z)
        self.amp2 /= np.sqrt(Z)

    def GetSampleRate(self) -> float:
        """
        Getter for sample rate

        Returns:
        --------
            float: Sample rate in Hertz
        """
        return self.__sampleRate
