"""
QTNM trap module

Module containing implementations of BaseTrap.

Implementations:
---------------

HarmonicTrap: Harmonic magnetic field providing trapping field 
BathtubTrap: Two-coil bathtub magnetic field providing trapping field
"""

import numpy as np
from src.BaseTrap import BaseTrap
import scipy.constants as sc


class HarmonicTrap(BaseTrap):
    """
    Trap class representing a harmonic magnetic field

    """

    __B0 = 0.0
    __L0 = 0.0

    def __init__(self, B0, L0):
        """
        Constructor for HarmonicTrap

        Parameters:
        ----------
        B0: float representing the magnetic field strength at the trap centre
        L0: float representing the characteristic length of the trap
        """
        self.__B0 = B0
        self.__L0 = L0

    def CalcZMax(self, pitchAngle):
        """
        Calc the maximum axial position

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        """
        return self.__L0 / np.tan(pitchAngle)

    def CalcOmegaAxial(self, pitchAngle, v):
        """
        Get the axial frequency of the electron's motion in radians/s

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """
        return v * np.sin(pitchAngle) / self.__L0

    def CalcOmega0(self, v, pitchAngle):
        """
        Get the average cyclotron frequency in radians/s

        Parameters:
        ----------
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """
        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta ** 2)
        return sc.e * self.__B0 / (sc.m_e * gamma) * (1 + self.CalcZMax(pitchAngle)**2 / (2 * self.__L0**2))

    def GetBzTime(self, t, pitchAngle, v):
        """
        Get the axial component of the magnetic field at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """
        zMax = self.CalcZMax(pitchAngle)
        omegaA = self.CalcOmegaAxial(pitchAngle, v)
        return self.__B0 * (1 + zMax**2 / (2 * self.__L0**2) - zMax**2 / (2 * self.__L0**2) * np.cos(2 * omegaA * t))

    def GetBzPosition(self, pos):
        """
        Get the axial component of the magnetic field at a given position

        Parameters:
        ----------
        pos: numpy array representing the position
        """
        return self.__B0 * (1 + (pos[2] / self.__L0)**2)

    def GetCyclotronPhase(self, t, v, pitchAngle):
        """
        Get the phase of the cyclotron motion

        Parameters:
        ----------
        t: float representing the time in seconds
        """
        omega0 = self.CalcOmega0(v, pitchAngle)
        omegaA = self.CalcOmegaAxial(pitchAngle, v)
        zMax = self.CalcZMax(pitchAngle)
        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta ** 2)
        q = -sc.e * self.__B0 / (gamma * sc.m_e) * \
            zMax**2 / (4 * self.__L0**2 * omegaA)
        return omega0 * t + q * np.sin(2 * omegaA * t)

    def GetZPosTime(self, t, v, pitchAngle):
        """
        Get the axial position at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """
        zMax = self.CalcZMax(pitchAngle)
        omegaA = self.CalcOmegaAxial(pitchAngle, v)
        return zMax * np.sin(omegaA * t)


class BathtubTrap(BaseTrap):
    """
    Trap class representing a bathtub magnetic field
    """

    __B0 = 0.0
    __L0 = 0.0
    __L1 = 0.0

    def __init__(self, B0, L0, L1):
        """
        Constructor for BathtubTrap
        """
        self.__B0 = B0
        self.__L0 = L0
        self.__L1 = L1

    def CalcZMax(self, pitchAngle):
        """
        Calc the maximum axial position

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        """
        return self.__L0 / np.tan(pitchAngle)

    def CalcOmegaAxial(self, pitchAngle, v):
        """
        Get the axial frequency of the electron's motion in radians/s

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """
        wa = v * np.sin(pitchAngle) / self.__L0
        return wa / (1 + self.__L1 * np.tan(pitchAngle) / (self.__L0 * np.pi))

    def CalcOmega0(self, v, pitchAngle):
        """
        Get the average cyclotron frequency in radians/s

        Parameters:
        ----------
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """
        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta**2)
        prefac = sc.e * self.__B0 / (sc.m_e * gamma)
        return prefac * (1 + (self.CalcZMax(pitchAngle)**2 / (2 * self.__L0**2)) / (1 + self.__L1 * np.tan(pitchAngle) / (self.__L0 * np.pi)))

    def GetBzTime(self, t, pitchAngle, v):
        """
        Get the axial component of the magnetic field at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """
        zMax = self.CalcZMax(pitchAngle)
        wa = v * np.sin(pitchAngle) / self.__L0
        t0 = 0.0
        t1 = self.__L1 / (v * np.cos(pitchAngle))
        t2 = t1 + np.pi / wa
        t3 = t1 + t2
        T = 2 * t2

        tPeriodic = t % T
        if tPeriodic < t1:
            return self.__B0
        elif tPeriodic > t1 and tPeriodic < t2:
            return self.__B0 * (1 + zMax**2 / (2 * self.__L0**2) - zMax**2 / (2 * self.__L0**2) * np.cos(2 * wa * (tPeriodic - t1)))
        elif tPeriodic > t2 and tPeriodic < t3:
            return self.__B0
        else:
            return self.__B0 * (1 + zMax**2 / (2 * self.__L0**2) - zMax**2 / (2 * self.__L0**2) * np.cos(2 * wa * (tPeriodic - t3)))

    def GetBzPosition(self, pos):
        """
        Get the axial component of the magnetic field at a given position

        Parameters:
        ----------
        pos: numpy array representing the position in metres
        """
        z = pos[2]
        if z < -self.__L1 / 2:
            return self.__B0 * (1 + (z + self.__L1 / 2)**2 / self.__L0**2)
        elif z > -self.__L1 / 2 and z < self.__L1 / 2:
            return self.__B0
        else:
            return self.__B0 * (1 + (z - self.__L1 / 2)**2 / self.__L0**2)

    def GetZPosTime(self, t, v, pitchAngle):
        zMax = self.CalcZMax(pitchAngle)
        wa = v * np.sin(pitchAngle) / self.__L0
        t0 = 0.0
        t1 = self.__L1 / (v * np.cos(pitchAngle))
        t2 = t1 + np.pi / wa
        t3 = t1 + t2
        T = 2 * t2
        vz0 = v * np.cos(pitchAngle)

        tPeriodic = t % T
        if tPeriodic < t1:
            return vz0 * tPeriodic - self.__L1 / 2
        elif tPeriodic > t1 and tPeriodic < t2:
            return zMax * np.sin(wa * (tPeriodic - t1)) + self.__L1 / 2
        elif tPeriodic > t2 and tPeriodic < t3:
            return -vz0 * (tPeriodic - t2) + self._L1 / 2
        else:
            return -zMax * np.sin(wa * (tPeriodic - t3)) - self.__L1 / 2
