"""
utils.py

This file contains utility functions
"""

import numpy as np
import scipy.constants as sc


def EquivalentMagneticMoment(p0, pitchAngle, B0):
    """
    Calculate the equivalent magnetic moment

    Parameters:
    ----------
    p0: float representing the momentum in kg m/s
    pitchAngle: float representing the pitch angle in radians
    B0: float representing the magnetic field strength at the trap centre in T
    """
    return 0.5 * p0**2 * np.sin(pitchAngle)**2 / (B0 * sc.m_e)


def ElectronVelocity(v, pitchAngle, phi):
    """
    Calculate the electron velocity

    Parameters:
    ----------
    v: float representing the speed of the electron in m/s
    pitchAngle: float representing the pitch angle in radians
    phi: float representing the azimuthal angle in radians
    """
    return v * np.array([np.sin(pitchAngle) * np.cos(phi),
                         np.sin(pitchAngle) * np.sin(phi), np.cos(pitchAngle)])


def PitchAngleFromField(B, p0, mu):
    '''Calculate the pitch angle of an electron from the magnetic field

    B: float representing the magnetic field strength
    p0: float representing the momentum of the electron
    mu: float representing the magnetic moment of the electron'''
    return np.arcsin(np.sqrt(2 * mu * sc.m_e * B / p0**2))


def LOOutput(t: float, f: float):
    '''Calculate the output of the LO at a given time

    t: float representing the time in seconds
    f: float representing the frequency of the LO'''
    return np.cos(2 * np.pi * f * t)


def CyclotronFrequency(B: float, gamma: float):
    """
    Calculate the cyclotron frequency of an electron in a magnetic field                                                                                                             

    Parameters:
    ----------
    B: float representing the magnetic field strength in Tesla
    gamma: float representing the Lorentz factor of the electron
    """
    return sc.e * B / (2 * np.pi * gamma * sc.m_e)
