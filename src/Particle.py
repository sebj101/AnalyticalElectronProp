"""
Particle.py

File containing particle class for easy calculation of particle kinematics 
"""

import numpy as np
import scipy.constants as sc


class Particle:
    """
    Class representing a particle with mass, charge, initial kinetic energy and
    pitch angle
    """

    def __init__(self, ke, startPos: np.ndarray, q=-sc.e, mass=sc.m_e,
                 pitchAngle=np.pi/2) -> None:
        """
        Constructor for Particle class

        Parameters:
        ----------
        ke: float representing the initial kinetic energy in eV
        startPos: 3-vector representing the initial position of the particle
        q: float representing the charge of the particle in C
        mass: float representing the mass of the particle in kg
        pitchAngle: float representing the pitch angle of the particle in radians
        """
        self.__ke = ke
        self.__pos = startPos
        self.__q = q
        self.__mass = mass
        self.__pitchAngle = pitchAngle

    def GetGamma(self):
        """
        Get the Lorentz factor of the particle
        """
        return 1.0 + self.__ke * sc.e / (self.__mass * sc.c**2)

    def GetBeta(self):
        """
        Get the beta factor of the particle
        """
        return np.sqrt(1 - 1 / self.GetGamma()**2)

    def GetSpeed(self):
        """
        Get the speed of the particle
        """
        return sc.c * self.GetBeta()

    def GetMomentum(self):
        """
        Get the momentum of the particle
        """
        return self.GetGamma() * self.__mass * self.GetSpeed()

    def GetPitchAngle(self):
        """
        Get the pitch angle of the particle in radians
        """
        return self.__pitchAngle

    def GetPosition(self):
        """
        Get the position of the particle
        """
        return self.__pos
