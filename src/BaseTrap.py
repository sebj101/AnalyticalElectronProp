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

    Attributes:
    ----------
    __gradB: float representing the gradient of the magnetic field
    """

    __gradB = 0.0

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

    @abstractmethod
    def CalcOmega0(self, v, pitchAngle):
        """
        Get the average cyclotron frequency

        Parameters:
        ----------
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """

    @abstractmethod
    def GetBzTime(self, t, pitchAngle, v):
        """
        Get the axial component of the magnetic field at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
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
    def GetCyclotronPhase(self, t, v, pitchAngle):
        """
        Get the phase of the cyclotron motion

        Parameters:
        ----------
        t: float representing the time in seconds
        """

    @abstractmethod
    def GetZPosTime(self, t, v, pitchAngle):
        """
        Get the axial position of the electron at a given time

        Parameters:
        ----------
        t: float representing the time in seconds
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """

    def GetGradB(self):
        """
        Getter for the gradient of the magnetic field

        Returns:
        --------
            float: Gradient of the magnetic field in Tesla per metre
        """
        return self.__gradB

    def SetGradB(self, gradB):
        """
        Setter for the gradient of the magnetic field

        Parameters:
        -----------
            gradB (float): Gradient of the magnetic field in Tesla per metre
        """
        self.__gradB = gradB
