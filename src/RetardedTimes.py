"""
RetardedTimes.py

Contains various different classes used for solving for retarded times in the
system.

S. Jones 18/07/24
"""

import numpy as np
import scipy.constants as sc
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline


class TaylorSolver:
    """
    Class using Taylor series expansion to solve for retarded times
    """

    def __init__(self, receiverTimes, electronPositions, receiverPosition,
                 order: int = 1):
        """
        Constructor for TaylorSolver class

        Parameters:
        -----------
            receiverTimes: Array of times at which the receiver is sampling
            electronPositions: Array of electron positions at each time
            receiverPosition: Position of the receiver (3D array)
            order (int): Order of the Taylor series expansion
        """
        self.__receiverTimes = receiverTimes
        self.__electronPositions = electronPositions
        self.__receiverPosition = receiverPosition
        self.__order = order

        if order < 0:
            raise ValueError("Order must be a positive integer")
        elif order > 2:
            raise NotImplementedError("Order > 2 not supported")

    def __Differentiate(self, y, x):
        """
        Differentiate array y with respect to array x

        Parameters:
        -----------
            y: Array to differentiate
            x: Array to differentiate with respect to
        """
        diff = np.zeros_like(y)
        diff[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / (x[2:] - x[:-2])
        return diff

    def __ZerothOrder(self):
        """
        Solve for the retarded times using the zeroth order Taylor series
        """
        d = np.zeros_like(self.__receiverTimes)
        for i in range(len(self.__receiverTimes)):
            d[i] = np.linalg.norm(
                self.__receiverPosition - self.__electronPositions[:, i])

        return self.__receiverTimes - d / sc.c

    def __FirstOrder(self):
        """
        Solve for the retarded times using the first order Taylor series
        """
        d = np.zeros_like(self.__receiverTimes)
        for i in range(len(self.__receiverTimes)):
            d[i] = np.linalg.norm(
                self.__receiverPosition - self.__electronPositions[:, i])

        v = self.__Differentiate(d, self.__receiverTimes)
        return self.__receiverTimes - d / (sc.c + v)

    def __SecondOrder(self):
        """
        Solve for the retarded times using the second order Taylor series
        """
        d = np.zeros_like(self.__receiverTimes)
        for i in range(len(self.__receiverTimes)):
            d[i] = np.linalg.norm(
                self.__receiverPosition - self.__electronPositions[:, i])

        v = self.__Differentiate(d, self.__receiverTimes)
        a = self.__Differentiate(v, self.__receiverTimes)
        return self.__receiverTimes - (sc.c + v) / a + np.sqrt(
            (sc.c + v)**2 - 2 * a * d) / a

    def CalcTRet(self):
        """
        Solve for the retarded times using the Taylor series expansion
        """
        if self.__order == 0:
            return self.__ZerothOrder()
        elif self.__order == 1:
            return self.__FirstOrder()
        elif self.__order == 2:
            return self.__SecondOrder()


class ForwardSolver:
    """
    Class utilising calculation of the advanced time to solve for the retarded
    time
    """

    def __init__(self, receiverTimes, electronPositions, receiverPosition):
        """
        Constructor for ForwardSolver class

        Parameters:
        -----------
            receiverTimes: Array of times at which the receiver is sampling
            electronPositions: Array of electron positions at each time
            receiverPosition: Position of the receiver (3D array)
        """
        self.__receiverTimes = receiverTimes
        self.__electronPositions = electronPositions
        self.__receiverPosition = receiverPosition

        # Initially calculate distances between receiver and electron
        d = np.zeros_like(self.__receiverTimes)
        for i in range(len(self.__receiverTimes)):
            d[i] = np.linalg.norm(
                self.__receiverPosition - self.__electronPositions[:, i])

        self.__tAdvanced = self.__receiverTimes + d / sc.c

    def CalcTRet(self):
        """
        Calculate the retarded times by interpolating between advanced times
        """

        spl = CubicSpline(self.__tAdvanced,
                          self.__receiverTimes, extrapolate=True)
        return spl(self.__receiverTimes)
