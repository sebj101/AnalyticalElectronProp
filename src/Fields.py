"""
Fields.py

Code for calculating relativistic electric fields

Author: S. Jones 
Date: 18/07/24
"""

import numpy as np
import scipy.constants as sc


def CalcEField(r, re, ve, ae):
    """
    Calculate the electric field at a point r due to an electron at position re.
    This is calculated from the Lienard-Wiechert potentials.

    Parameters:
    -----------
        r: Field point
        re: Position of the electron
        ve: Velocity of the electron
        ae: Acceleration of the electron
    """
    beta = ve / sc.c
    betaDot = ae / sc.c
    prefac = -sc.e / (4 * np.pi * sc.epsilon_0)
    R = np.linalg.norm(r - re)
    RHat = (r - re) / R
    term1 = (RHat - beta) * (1 - np.dot(beta, beta)) / \
        ((1 - np.dot(beta, RHat))**3 * R**2)
    term2 = np.cross(RHat, np.cross(RHat - beta, betaDot)) / \
        (sc.c * R * (1 - np.dot(beta, RHat))**3)
    return prefac * (term1 + term2)
