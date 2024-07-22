"""
RealTraps.py

Module containing classes used to represent slightly more realistic traps

BaseRealTrap: Class for representing a more realistic trap
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as sc
from scipy.special import ellipk, ellipe


class RealBaseTrap(ABC):
    """
    Base class for realistic traps

    Implementations of this class should give us
    """


class CoilField():
    """
    Class describing the magnetic field produced by a current loop
    """

    def __init__(self, coilRadius: float, coilCurrent: float, zOffset: float):
        """
        Constructor for CoilField

        Parameters:
        -----------
            coilRadius (float): Radius of the coil in meters
            coilCurrent (float): Current in the coil in Amperes
            zOffset (float): Z offset of the coil from the origin in meters
        """

        self.__rCoil = coilRadius
        self.__iCoil = coilCurrent
        self.__zOff = zOffset

    def __CentralField(self):
        """
        Calculate the central field of the coil
        """
        return self.__iCoil * sc.mu_0 / self.__rCoil / 2.0

    def __OnAxisField(self, z: float):
        """
        Calculate the field on the axis of the coil

        Parameters:
        -----------
            z (float): Distance along the coil axis in metres
        """
        return (sc.mu_0 * self.__iCoil * self.__rCoil**2 / 2.0 / (self.__rCoil**2 + (z - self.__zOff)**2)**(1.5))

    def BFieldAtPoint(self, pos):
        """
        Calculate magnetic field vector at a given position

        Parameters:
        -----------
            pos (np.array): Position vector in meters

        Returns:
        --------
            np.array: Magnetic field vector in Tesla
        """
        if rho / self.__rCoil < 1e-6:
            return np.array([0.0, 0.0, self.__CentralField(pos[2])])

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        zRel = pos[2] - self.__zOff

        BCentral = self.__CentralField()
        rhoNorm = rho / self.__rCoil
        zNorm = zRel / self.__rCoil
        alpha = (1.0 + rhoNorm)**2 + zNorm**2
        rootAlphaPi = np.sqrt(alpha) * np.pi
        beta = 4.0 * rhoNorm / alpha
        int_e = ellipe(beta)
        int_k = ellipk(beta)
        gamma = alpha - 4 * rhoNorm

        BRho = BCentral * (int_e * ((1.0 + rhoNorm**2 + zNorm**2) /
                           gamma) - int_k) / rootAlphaPi * (zRel / rho)
        BZ = BCentral * \
            (int_e * ((1.0 - rhoNorm**2 - zNorm**2) / gamma) + int_k) / rootAlphaPi

        return np.array([BRho * pos[0] / rho, BRho * pos[1] / rho, BZ])


class RealBathtubField():
    """
    Class representing a realistic bathtub field centred at z = 0.
    The field is produced by two current loops at z = -L/2 and z = L/2.
    """

    def __init__(self, coilRadius: float, coilCurrent: float, trapLength: float,
                 bkgField: float):
        """
        Constructor for RealBathtubField

        Parameters:
        -----------
            coilRadius (float): Radius of the coil in meters
            coilCurrent (float): Current in the coil in Amperes
            trapLength (float): Length of the trap in meters
            bkgField (float): Background field in Tesla
        """

        self.__bkg = bkgField
        self.__coil1 = CoilField(coilRadius, coilCurrent, -trapLength / 2.0)
        self.__coil2 = CoilField(coilRadius, coilCurrent, trapLength / 2.0)

    def BFieldAtPoint(self, pos):
        """
        Calculate magnetic field vector at a given position

        Parameters:
        -----------
            pos (np.array): Position vector in meters

        Returns:
        --------
            np.array: Magnetic field vector in Tesla
        """
        return self.__coil1.BFieldAtPoint(pos) + self.__coil2.BFieldAtPoint(pos) + np.array([0.0, 0.0, self.__bkg])


class RealHarmonicTrap():
    """
    Class representing a real harmonic trap

    Presenting this with an electron position should output the correct analytic
    harmonic trap.
    """

    def __init__(self, coilRadius: float, coilCurrent: float, bkgField: float):
        """
        Constructor for RealHarmonicTrap

        Parameters:
        -----------
            coilRadius (float): Radius of the coil in meters
            coilCurrent (float): Current in the coil in Amperes
        """

        self.__coil1 = CoilField(coilRadius, coilCurrent, 0.0)
        self.__bkg = bkgField

    def BFieldAtPoint(self, pos):
        """
        Calculate magnetic field vector at a given position.

        Parameters:
        -----------
            pos (np.array): Position vector in meters

        Returns:
        --------
            np.array: Magnetic field vector in Tesla
        """
        return np.array([0.0, 0.0, self.__bkg]) - self.__coil1.BFieldAtPoint(pos)
