from scipy.special import j1, jvp
import numpy as np
import scipy.constants as sc


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
        conditions = [rho > self.wgR, rho == 0.0, rho <= self.wgR]
        choices = [0.0, A * np.cos(phi) / kc, (A / (kc * rho))
                   * j1(kc * rho) * np.cos(phi)]
        return np.select(conditions, choices)

    def HFieldTE11Rho_1(self, rho: float, phi: float, omega: float, A: float):
        '''Calculate the radial magnetic field for the TE11 mode

        Parameters:
        ----------
        rho: float
            Radial position in the waveguide
        phi: float
            Azimuthal angle in the waveguide
        omega: float
            Angular frequency of the mode in rad/s
        A: float
            Amplitude of the mode
        '''
        kc = 1.841 / self.wgR
        beta = np.sqrt(omega**2 / sc.c**2 - kc**2)
        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, A * beta *
                   jvp(1, kc * rho, 1) * np.sin(phi) / (omega * sc.mu_0)]
        return np.select(conditions, choices)

    def EFieldTE11Phi_1(self, rho: float, phi: float, A: float):
        '''Calculate the radial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        '''
        kc = 1.841 / self.wgR
        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, -A * jvp(1, kc * rho, 1) * np.sin(phi)]
        return np.select(conditions, choices)

    def HFieldTE11Phi_1(self, rho: float, phi: float, omega: float, A: float):
        '''
        Calculate the azimuthal magnetic field for the TE11 mode

        Parameters:
        -----------
        rho: float
            Radial position in the waveguide in metres
        phi: float
            Azimuthal angle in the waveguide in radians
        omega: float
            Angular frequency of the mode in rad/s
        A: float
            Amplitude of the mode

        Returns:
        --------
        float: The azimuthal magnetic field in Tesla
        '''
        kc = 1.841 / self.wgR
        beta = np.sqrt(omega**2 / sc.c**2 - kc**2)
        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, A * beta *
                   j1(kc * rho) * np.cos(phi) / (kc * rho * omega * sc.mu_0)]
        return np.select(conditions, choices)

    def EFieldTE11Z(self, rho, phi, A):
        """
        Calculate the axial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position
        phi: float representing the azimuthal position
        A: float representing the amplitude of the mode
        """
        return np.zeros_like(rho)

    def EFieldTE11_1(self, rho, phi, A):
        '''Calculate the electric field vector for mode 1 in Cartesian coordinates

        Parameters:
        ----------
        rho: float  
            Radial position in metres
        phi: float 
            Azimuthal position in radians
        A: float 
            Amplitude of the mode
        '''

        return np.array([self.EFieldTE11Rho_1(rho, phi, A) * np.cos(phi) - self.EFieldTE11Phi_1(rho, phi, A) * np.sin(phi),
                         self.EFieldTE11Rho_1(
                             rho, phi, A) * np.sin(phi) + self.EFieldTE11Phi_1(rho, phi, A) * np.cos(phi),
                         self.EFieldTE11Z(rho, phi, A)])

    def EFieldTE11Pos_1(self, pos, A):
        '''Calculate the electric field vector for mode 1 in Cartesian coordinates

        Parameters:
        ----------
        pos: np.ndarray
            Position three vector in metres
        A: float 
            Amplitude of the mode
        '''

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.EFieldTE11_1(rho, phi, A)

    def HFieldTE11_1(self, rho, phi, omega, A):
        '''
        Calculate the magnetic field vector for mode 1 in Cartesian coordinates

        Parameters:
        ----------
        rho: float 
            Radial position in metres
        phi: float 
            Azimuthal position in radians
        omega: float
            Angular frequency of the mode in rad/s
        A: float 
            Amplitude of the mode
        '''

        return np.array([self.HFieldTE11Rho_1(rho, phi, omega, A) * np.cos(phi) - self.HFieldTE11Phi_1(rho, phi, omega, A) * np.sin(phi),
                         self.HFieldTE11Rho_1(rho, phi, omega, A) * np.sin(
                             phi) + self.HFieldTE11Phi_1(rho, phi, omega, A) * np.cos(phi),
                         0.0])

    def HFieldTE11Pos_1(self, pos, omega, A):
        '''
        Calculate the magnetic field vector for mode 1 in Cartesian coordinates

        Parameters:
        ----------
        pos: np.ndarray
            Position three vector in metres
        omega: float
            Angular frequency of the mode in rad/s
        A: float 
            Amplitude of the mode
        '''

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.HFieldTE11_1(rho, phi, omega, A)

    def EFieldTE11Rho_2(self, rho: float, phi: float, A: float):
        '''Calculate the radial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        '''

        kc = 1.841 / self.wgR
        conditions = [rho > self.wgR, rho == 0.0, rho <= self.wgR]
        choices = [
            0.0, -A * np.sin(phi) / kc, (-A / (kc * rho)) * j1(kc * rho) * np.sin(phi)]
        return np.select(conditions, choices)

    def HFieldTE11Rho_2(self, rho: float, phi: float, omega: float, A: float):
        '''
        Calculate the radial magnetic field for the TE11 mode

        Parameters:
        -----------
        rho: float
            Radial position in the waveguide
        phi: float
            Azimuthal angle in the waveguide
        omega: float
            Angular frequency of the mode in rad/s
        A: float
            Amplitude of the mode

        Returns:
        --------
        float: The radial magnetic field in Tesla
        '''
        kc = 1.841 / self.wgR
        beta = np.sqrt(omega**2 / sc.c**2 - kc**2)
        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, A * beta *
                   np.cos(phi) * jvp(1, kc * rho, 1) / (omega * sc.mu_0)]
        return np.select(conditions, choices)

    def EFieldTE11Phi_2(self, rho: float, phi: float, A: float):
        '''Calculate the radial electric field for the TE11 mode

        Parameters:
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        '''

        kc = 1.841 / self.wgR
        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, -A * jvp(1, kc * rho, 1) * np.cos(phi)]
        return np.select(conditions, choices)

    def HFieldTE11Phi_2(self, rho: float, phi: float, omega: float, A: float):
        '''
        Calculate the azimuthal magnetic field for the TE11 mode

        Parameters:
        -----------
        rho: float
            Radial position in the waveguide in metres
        phi: float
            Azimuthal angle in the waveguide in radians
        omega: float
            Angular frequency of the mode in rad/s
        A: float
            Amplitude of the mode

        Returns:
        --------
        float: The azimuthal magnetic field in Tesla
        '''
        kc = 1.841 / self.wgR
        beta = np.sqrt(omega**2 / sc.c**2 - kc**2)
        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, -A * beta *
                   j1(kc * rho) * np.sin(phi) / (kc * rho * omega * sc.mu_0)]
        return np.select(conditions, choices)

    def EFieldTE11_2(self, rho, phi, A):
        '''Calculate the electric field vector for mode 2 in Cartesian coordinates

        rho: float representing the radial position
        phi: float representing the azimuthal position
        '''

        return np.array([self.EFieldTE11Rho_2(rho, phi, A) * np.cos(phi) - self.EFieldTE11Phi_2(rho, phi, A) * np.sin(phi),
                        self.EFieldTE11Rho_2(
                            rho, phi, A) * np.sin(phi) + self.EFieldTE11Phi_2(rho, phi, A) * np.cos(phi),
                        self.EFieldTE11Z(rho, phi, A)])

    def EFieldTE11Pos_2(self, pos, A):
        '''Calculate the electric field vector for mode 2 in Cartesian coordinates

        pos: numpy array representing the position
        A: float representing normlisation constant
        '''

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.EFieldTE11_2(rho, phi, A)

    def HFieldTE11_2(self, rho, phi, omega, A):
        '''
        Calculate the magnetic field vector for mode 2 in Cartesian coordinates

        Parameters:
        -----------
        rho: float 
            Radial position in metres
        phi: float 
            Azimuthal position in radians
        omega: float
            Angular frequency of the wave in rad/s
        A: float
            Amplitude of the mode

        Returns:
        --------
        numpy array: The magnetic field vector in Tesla
        '''

        return np.array([self.HFieldTE11Rho_2(rho, phi, omega, A) * np.cos(phi) - self.HFieldTE11Phi_2(rho, phi, omega, A) * np.sin(phi),
                        self.HFieldTE11Rho_2(
                            rho, phi, omega, A) * np.sin(phi) + self.HFieldTE11Phi_2(rho, phi, omega, A) * np.cos(phi),
                         0.0])

    def HFieldTE11Pos_2(self, pos, omega, A):
        '''
        Calculate the magnetic field vector for mode 2 in Cartesian coordinates

        Parameters:
        -----------
        pos: np.ndarray
            numpy array representing the position in metres
        omega: float
            Angular frequency of the wave in rad/s
        A: float 
            representing normlisation constant
        '''

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.HFieldTE11_2(rho, phi, omega, A)

    def CalcTE11Impedance(self, omega):
        """
        Calculate the impedance of the TE11 mode

        Parameters:
        ----------
        omega: float representing the angular frequency of the mode in rad/s

        Returns:
        -------
        float representing the impedance of the mode in Ohms
        """
        k = omega / sc.c
        kc = 1.841 / self.wgR
        betaMode = np.sqrt(k**2 - kc**2)
        return k * np.sqrt(sc.mu_0 / sc.epsilon_0) / betaMode

    def CalcNormalisationFactor(self):
        """
        Calculate the required normalisation factor for the waveguide

        Returns:
        -------
        float: The required normalisation factor
        """

        xArray = np.linspace(-self.wgR, self.wgR, 100)
        yArray = np.linspace(-self.wgR, self.wgR, 100)
        E1Integral = 0.0
        for i in range(len(xArray)):
            for j in range(len(yArray)):
                E1Integral += np.linalg.norm(self.EFieldTE11Pos_1(np.array(
                    [xArray[i], yArray[j], 0]), 1))**2 * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])

        return 1 / np.sqrt(E1Integral)
