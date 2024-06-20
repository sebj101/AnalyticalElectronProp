"""
Particle.py

File containing particle class for easy calculation of particle kinematics 
"""

import numpy as np
import scipy.constants as sc


class Particle:
    """
    Class representing a particle with mass, charge and initial kinetic energy
    """

    def __init__(self, ke, q=sc.e, mass=sc.m_e) -> None:
        """
        Constructor for Particle class

        Parameters:
        ----------
        ke: float representing the initial kinetic energy in eV
        q: float representing the charge of the particle in C
        mass: float representing the mass of the particle in kg
        """
        self.__ke = ke
        self.__q = q
        self.__mass = mass

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
