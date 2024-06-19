'''
BaseTrap.py

This file contains the BaseTrap class, which is an abstract class representing a
generic electron trap.

S. Jones 19-06-24
'''
from abc import ABC, abstractmethod


class BaseTrap(ABC):
    """
    Base class for electron traps
    """

    __zMax = 0.0
    __omega0 = 0.0

    def GetZMax(self):
        """
        Get the maximum axial position

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        """
        return self.__zMax

    @abstractmethod
    def CalcZMax(self, pitchAngle):
        """
        Calculate the maximum axial position

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        """

    @abstractmethod
    def CalcOmegaAxial(self, pitchAngle, v):
        """
        Get the axial frequency of the electron's motion

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """

    def GetOmega0(self, v):
        """
        Get the average cyclotron frequency

        Parameters:
        ----------
        v: float representing the speed of the electron in m/s
        """
        return self.__omega0

    @abstractmethod
    def CalcOmega0(self, v):
        """
        Get the average cyclotron frequency

        Parameters:
        ----------
        v: float representing the speed of the electron in m/s
        """

    @abstractmethod
    def GetBzTime(self, t):
        """
        Get the axial component of the magnetic field at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        """

    @abstractmethod
    def GetBzPosition(self, pos):
        """
        Get the axial component of the magnetic field at a given position

        Parameters:
        ----------
        pos: numpy array representing the position
        """

    @abstractmethod
    def GetCyclotronPhase(self, t):
        """
        Get the phase of the cyclotron motion

        Parameters:
        ----------
        t: float representing the time in seconds
        """

    @abstractmethod
    def GetZPosTime(self, t):
        """
        Get the axial position of the electron at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        """
