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

    def __init__(self, B0, L0, gradB=0.0):
        """
        Constructor for HarmonicTrap

        Parameters:
        ----------
        B0: float representing the magnetic field strength at the trap centre
        L0: float representing the characteristic length of the trap
        gradB: float representing the gradient of the magnetic field
        """
        self.__B0 = B0
        self.__L0 = L0
        self.SetGradB(gradB)

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

    def __init__(self, B0, L0, L1, gradB=0.0):
        """
        Constructor for BathtubTrap

        Parameters:
        ----------
        B0: float representing the magnetic field strength at the trap centre
        L0: float representing the field gradient of the trap
        L1: float representing the length of the trap
        gradB: float representing the radial gradient of the magnetic field in T/m
        """
        self.__B0 = B0
        self.__L0 = L0
        self.__L1 = L1
        self.__gradB = gradB

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
        conditions = [tPeriodic < t1, (tPeriodic > t1) & (tPeriodic < t2),
                      (tPeriodic > t2) & (tPeriodic < t3), tPeriodic > t3]
        choices = [self.__B0,
                   self.__B0 * (1 + zMax**2 / (2 * self.__L0**2) - zMax**2 /
                                (2 * self.__L0**2) * np.cos(2 * wa * (tPeriodic - t1))),
                   self.__B0,
                   self.__B0 * (1 + zMax**2 / (2 * self.__L0**2) - zMax**2 / (2 * self.__L0**2) * np.cos(2 * wa * (tPeriodic - t3)))]
        return np.select(conditions, choices, default=0.0)

    def GetBzPosition(self, pos):
        """
        Get the axial component of the magnetic field at a given position

        Parameters:
        ----------
        pos: numpy array representing the position in metres
        """
        z = pos[2]
        conditions = [z < -self.__L1 / 2,
                      (z > -self.__L1 / 2) & (z < self.__L1 / 2), z > self.__L1 / 2]
        choices = [self.__B0 * (1 + (z + self.__L1 / 2)**2 / self.__L0**2),
                   self.__B0, self.__B0 * (1 + (z - self.__L1 / 2)**2 / self.__L0**2)]
        return np.select(conditions, choices, default=0.0)

    def GetCyclotronPhase(self, t, v, pitchAngle):
        """
        Get the phase of the cyclotron motion

        Parameters:
        ----------
        t: float representing the time in seconds
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """
        zMax = self.CalcZMax(pitchAngle)
        wa = v * np.sin(pitchAngle) / self.__L0
        t0 = 0.0
        t1 = self.__L1 / (v * np.cos(pitchAngle))
        t2 = t1 + np.pi / wa
        t3 = t1 + t2
        T = 2 * t2

        conditions = [t % T < t1, (t % T > t1) & (t % T < t2),
                      (t % T > t2) & (t % T < t3), t % T > t3]
        choices = [t % T, (t % T) + zMax**2 * ((t % T) - t1) / (2 * self.__L0**2) - zMax**2 * np.sin(2 * wa * ((t % T) - t1)) / (4 * wa * self.__L0**2),
                   (t2 - t1) * zMax**2 / (2 * self.__L0**2) + (t % T),
                   (t % T) + (t2 - t1 - t3 + (t % T)) * zMax**2 / (2 * self.__L0**2) - zMax**2 * np.sin(2 * wa * ((t % T) - t3)) / (4 * wa * self.__L0**2)]

        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta**2)
        prefactor = sc.e * self.__B0 / (sc.m_e * gamma)

        # Account for completed axial oscillations
        nAxialCycles = np.floor(t / T)
        additionalTerm = (T + zMax**2 * np.pi /
                          (wa * self.__L0**2)) * nAxialCycles

        return (np.select(conditions, choices, default=0.0) + additionalTerm) * prefactor

    def GetZPosTime(self, t, v, pitchAngle):
        """
        Get the axial position of the electron at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """
        zMax = self.CalcZMax(pitchAngle)
        wa = v * np.sin(pitchAngle) / self.__L0
        t0 = 0.0
        t1 = self.__L1 / (v * np.cos(pitchAngle))
        t2 = t1 + np.pi / wa
        t3 = t1 + t2
        T = 2 * t2
        vz0 = v * np.cos(pitchAngle)

        tPeriodic = t % T
        conditions = [tPeriodic < t1, (tPeriodic > t1) & (tPeriodic < t2),
                      (tPeriodic > t2) & (tPeriodic < t3), tPeriodic > t3]
        choices = [vz0 * tPeriodic - self.__L1 / 2,
                   zMax * np.sin(wa * (tPeriodic - t1)) + self.__L1 / 2,
                   -vz0 * (tPeriodic - t2) + self.__L1 / 2,
                   -zMax * np.sin(wa * (tPeriodic - t3)) - self.__L1 / 2]
        return np.select(conditions, choices, default=0.0)

    def GetOmegaTime(self, t, pitchAngle, v):
        """
        Get the axial frequency of the electron's motion at a given time

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
        conditions = [tPeriodic < t1, (tPeriodic > t1) & (tPeriodic < t2),
                      (tPeriodic > t2) & (tPeriodic < t3), tPeriodic > t3]
        choices = [1.0, 1.0 + zMax**2 / (2 * self.__L0**2) - zMax**2 * np.cos(2 * wa * (tPeriodic - t1)) / (
            2 * self.__L0**2), 1.0, 1.0 + zMax**2 / (2 * self.__L0**2) - zMax**2 * np.cos(2 * wa * (tPeriodic - t3)) / (2 * self.__L0**2)]
        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta**2)
        prefactor = sc.e * self.__B0 / (sc.m_e * gamma)
        return np.select(conditions, choices, default=0.0) * prefactor
