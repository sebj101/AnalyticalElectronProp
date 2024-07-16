'''
BaseTrap.py

This file contains the BaseTrap class, which is an abstract class representing a
generic electron trap.

S. Jones 19-06-24
'''
from abc import ABC, abstractmethod
from src.Particle import Particle
import src.utils as utils
import numpy as np


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

    def CalcPositionTime(self, electron: Particle, t: float):
        """
        Calculate the position of the electron at a given time.
        This takes into account both the axial motion and the grad-B motion.

        Parameters:
        -----------
            electron (Particle): Particle object representing initial electron state
            t (float): Time in seconds
        """
        initialPos = electron.GetPosition()
        # Calculate the speed of the grad-B motion in m/s
        omega0 = self.CalcOmega0(electron.GetSpeed(), electron.GetPitchAngle())
        magMomentInit = utils.EquivalentMagneticMoment(electron.GetMomentum(),
                                                       electron.GetPitchAngle(),
                                                       self.GetBzPosition(initialPos))
        vGradB = magMomentInit * self.GetGradB() / (electron.GetMass() * omega0)

        # Calculate the frequency of the grad B motion
        rho = np.sqrt(initialPos[0]**2 + initialPos[1]**2)
        omegaGradB = vGradB / rho
        # Phase of the grad B motion
        phaseGradB = np.arctan2(initialPos[1], initialPos[0])
        xPos = rho * np.cos(omegaGradB * t + phaseGradB)
        yPos = rho * np.sin(omegaGradB * t + phaseGradB)

        # Now get the z position
        zPos = self.GetZPosTime(t, electron.GetSpeed(),
                                electron.GetPitchAngle())
        return np.array([xPos, yPos, zPos])
