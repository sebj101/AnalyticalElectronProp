from scipy.special import j1, jvp
import numpy as np


class CircularWaveguide:
    def __init__(self, radius):
        self.wgR = radius

    def __str__(self):
        return f"Waveguide with radius {self.wgR} metres"

    def EFieldTE11Rho_1(self, rho: float, phi: float, A: float):
        '''Calculate the radial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        '''
        kc = 1.841 / self.wgR
        if rho > self.wgR:
            return 0
        else:
            return (A / (kc * rho)) * j1(kc * rho) * np.cos(phi)

    def EFieldTE11Phi_1(self, rho: float, phi: float, A: float):
        '''Calculate the radial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        '''
        kc = 1.841 / self.wgR
        if rho > self.wgR:
            return 0
        else:
            return -A * jvp(1, kc * rho, 1) * np.sin(phi)

    def EFieldTE11_1(self, rho, phi, A):
        '''Calculate the electric field vector for mode 1 in Cartesian coordinates

        Parameters:
        ----------
        rho: float representing the radial position
        phi: float representing the azimuthal position
        alpha: float representing the alpha parameter of the waveguide
        '''

        return np.array([self.EFieldTE11Rho_1(rho, phi, A) * np.cos(phi) - self.EFieldTE11Phi_1(rho, phi, A) * np.sin(phi),
                         self.EFieldTE11Rho_1(rho, phi, A) * np.sin(phi) + self.EFieldTE11Phi_1(rho, phi, A) * np.cos(phi), 0])

    def EFieldTE11Pos_1(self, pos, A):
        '''Calculate the electric field vector for mode 1 in Cartesian coordinates

        Parameters:
        ----------
        pos: numpy array representing the position
        A: float representing the alpha parameter of the waveguide
        '''

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.EFieldTE11_1(rho, phi, A)

    def EFieldTE11Rho_2(self, rho: float, phi: float, A: float):
        '''Calculate the radial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        '''

        kc = 1.841 / self.wgR
        if rho > self.wgR:
            return 0
        else:
            return (-A / (kc * rho)) * j1(kc * rho) * np.sin(phi)

    def EFieldTE11Phi_2(self, rho: float, phi: float, A: float):
        '''Calculate the radial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        '''

        kc = 1.841 / self.wgR
        if rho > self.wgR:
            return 0
        else:
            return -A * jvp(1, kc * rho, 1) * np.cos(phi)

    def EFieldTE11_2(self, rho, phi, A):
        '''Calculate the electric field vector for mode 2 in Cartesian coordinates

        rho: float representing the radial position
        phi: float representing the azimuthal position
        '''

        return np.array([self.EFieldTE11Rho_2(rho, phi, A) * np.cos(phi) - self.EFieldTE11Phi_2(rho, phi, A) * np.sin(phi),
                        self.EFieldTE11Rho_2(rho, phi, A) * np.sin(phi) + self.EFieldTE11Phi_2(rho, phi, A) * np.cos(phi), 0])

    def EFieldTE11Pos_2(self, pos, A):
        '''Calculate the electric field vector for mode 2 in Cartesian coordinates

        pos: numpy array representing the position
        A: float representing normlisation constant
        '''

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.EFieldTE11_2(rho, phi, A)
