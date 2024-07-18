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
from scipy.optimize import fsolve, curve_fit
import src.RetardedTimes as ret


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
                 tSignal: float, readout: Readout, receiverPos: np.ndarray,
                 usefsolve: bool = False):
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
        EJoules = self.__electron.GetEnergy() * sc.e + \
            self.__electron.GetMass() * sc.c**2  # Total energy in Joules
        initialPos = electron.GetPosition()
        paInit = self.__electron.GetPitchAngle()  # Pitch angle in radians
        magMomentInit = utils.EquivalentMagneticMoment(
            p0, paInit, self.__trap.GetBzPosition(initialPos))

        # Calculate the Larmor power
        omega0 = self.__trap.CalcOmega0(v0, paInit)
        pLarmor = sc.e**2 * (v0 / sc.c)**2 * np.sin(paInit)**2 * omega0**2 / \
            (6 * np.pi * sc.epsilon_0 * sc.c)

        # Given the motion of the particle we want to calculate a list of
        # retarded times
        # Initially we want to sample at 10 times the digitizer rate
        timeFine = np.arange(
            0, self.__t, 1 / (10 * self.__readout.GetSampleRate()))

        # Calculate the electron speed as a function of time
        vTime = np.zeros_like(timeFine)
        deltaT = timeFine[1] - timeFine[0]
        for iT, T in enumerate(timeFine):
            # Calculate the speed from the energy in joules
            gammaTmp = EJoules / (self.__electron.GetMass() * sc.c**2)
            vTmp = np.sqrt(1 - 1 / gammaTmp**2) * sc.c
            vTime[iT] = vTmp
            EJoules -= pLarmor * deltaT

        tRet = np.zeros_like(timeFine)
        if usefsolve:
            for iT, T in enumerate(timeFine):
                def func(te): return T - te - np.linalg.norm(self.__receiverPos -
                                                             self.__trap.CalcPositionTime(self.__electron, te)[:, 0]) / sc.c
                tRet[iT] = fsolve(func, T)
        else:
            # Use the interpolation of advanced times
            solver = ret.ForwardSolver(timeFine, self.__trap.CalcPositionTime(self.__electron, timeFine),
                                       self.__receiverPos)
            tRet = solver.CalcTRet()

        # We ultimately need the electron velocity and position at these times
        BzRet = self.__trap.GetBzTime(tRet, paInit, vTime)
        paRet = utils.PitchAngleFromField(BzRet, p0, magMomentInit)
        phasesRet = self.__trap.GetCyclotronPhase(tRet, vTime, paInit)

        eVelRet = utils.ElectronVelocity(vTime, paRet, phasesRet)
        ePosRet = self.__trap.CalcPositionTime(self.__electron, tRet)

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
