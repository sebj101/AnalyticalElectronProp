#!/usr/bin/python3

import src.CircularWaveguide as cw
import src.utils as utils
import src.Particle as part
import src.QTNMTraps as traps
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.signal as sig
import src.CircularWaveguide as cw
from scipy.optimize import fsolve

# Use Latex in figure text
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# Make labels bigger
plt.rcParams.update({'font.size': 12})

wgRadius = 5e-3  # metres
wg = cw.CircularWaveguide(wgRadius)

# Integrate square of electric field over the cross-section of the waveguide
xArray = np.linspace(-wgRadius, wgRadius, 50)
yArray = np.linspace(-wgRadius, wgRadius, 50)
E1Integral = 0.0
E2Integral = 0.0
for i in range(len(xArray)):
    for j in range(len(yArray)):
        E1Integral += np.linalg.norm(wg.EFieldTE11Pos_1(np.array(
            [xArray[i], yArray[j], 0]), 1))**2 * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        E2Integral += np.linalg.norm(wg.EFieldTE11Pos_2(np.array(
            [xArray[i], yArray[j], 0]), 1))**2 * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])

normFactor1 = 1 / np.sqrt(E1Integral)
normFactor2 = 1 / np.sqrt(E2Integral)

eKE = 30e3  # eV
electron = part.Particle(eKE)
v0 = electron.GetSpeed()    # m/s
p0 = electron.GetMomentum()  # kg m/s

# Define trap parameters
L0 = 0.35    # m
L1 = 0.5e-2  # m
B0 = 1.0     # T
trap = traps.BathtubTrap(B0, L0, L1)

# Plot the signal structure for a range of pitch angles
pitchAngleArray = np.linspace(87.0, 90.0, 7) * np.pi / 180.0

sampleCoarsePeriod = 1e-9
sampleFinePeriod = sampleCoarsePeriod / 10.0
timeFine = np.arange(0, 1e-6, sampleFinePeriod)
# Define receiver position
zR = 0.2  # metres
receiverPosition = np.array([0.0, 0.0, zR])

# Loop over the pitch angles
for pa in pitchAngleArray:
    Omega_a = trap.CalcOmegaAxial(pa, v0)
    dopplerFactor = 1 / (1 - v0 * np.cos(pa) / sc.c)
    fieldFactor = 1 / (1 + (trap.CalcZMax(pa) / L0)**2)
    fractionalFieldChange = abs(1 - fieldFactor) + abs(1 - dopplerFactor)
    h = fractionalFieldChange * trap.GetOmegaTime(0, pa, v0) / Omega_a
    print(
        f"Calculating for pitch angle {pa * 180 / np.pi:.1f} degrees: f_a = {Omega_a / (2 * np.pi) / 1e6:.1f} MHz")
    print(
        f"Field factor = {fieldFactor:.2e}, Doppler factor = {dopplerFactor:.2e}, Total = {fractionalFieldChange:.2e}\th = {h:.2e}")

    mu = utils.EquivalentMagneticMoment(p0, pa, B0)

    teSolutions = np.zeros(len(timeFine))
    for iT, T in enumerate(timeFine):
        def func(te): return T - te - np.abs(zR -
                                             trap.GetZPosTime(te, v0, pa)) / sc.c
        teSolutions[iT] = fsolve(func, T)

    # Calculate the magnetic field as a function of time
    BzTime = trap.GetBzTime(teSolutions, pa, v0)
    # Pitch angle over time
    paTime = utils.PitchAngleFromField(BzTime, p0, mu)
    # Cyclotron phase over time
    phaseTime = trap.GetCyclotronPhase(teSolutions, v0, pa)
    eVels = utils.ElectronVelocity(v0, paTime, phaseTime)
    ePos = np.array([1e-5 * np.ones(len(timeFine)),
                    np.zeros(len(timeFine)), trap.GetZPosTime(teSolutions, v0, pa)])
    posFixed = np.array([1e-5, 0.0, 0.0])

    # Now do some actual waveguide calculations
    Z = wg.CalcTE11Impedance(trap.CalcOmega0(v0, pa))
    # Calculate waveguide mode amplitudes
    A1 = -sc.e * np.dot(wg.EFieldTE11Pos_1(posFixed,
                        normFactor1), eVels) * -Z / 2
    A2 = -sc.e * np.dot(wg.EFieldTE11Pos_2(posFixed,
                        normFactor2), eVels) * -Z / 2
    # Now multiply by local oscillator
    fLO = trap.CalcOmega0(v0, pa) / (2 * np.pi) - (1 / sampleCoarsePeriod) / 4
    A1 *= utils.LOOutput(timeFine, fLO)
    A2 *= utils.LOOutput(timeFine, fLO)

    # Apply a low pass filter to the data
    cutoffFreq = (1 / sampleCoarsePeriod) / 2
    filterOrder = 5
    A1 = utils.ButterLowPassFilter(A1, cutoffFreq, 1 / sampleFinePeriod,
                                   filterOrder)
    A2 = utils.ButterLowPassFilter(A2, cutoffFreq, 1 / sampleFinePeriod,
                                   filterOrder)

    # Keep overy 10th element
    A1 = A1[::10]
    A2 = A2[::10]

    # Make periodograms of signals
    fA1, PowerA1 = sig.periodogram(A1, fs=1 / sampleCoarsePeriod,
                                   window='boxcar', scaling='spectrum')
    fA2, PowerA2 = sig.periodogram(A2, fs=1 / sampleCoarsePeriod,
                                   window='boxcar', scaling='spectrum')

    # Plot the power spectrum
    plt.figure()
    plt.title(
        f"30 keV electron in a 1 T bathtub trap: ${pa * 180 / np.pi:.1f}^\circ$")
    plt.plot(fA1 / 1e6, PowerA1 * 1e15 / Z,
             label=f"$\\theta = {pa * 180 / np.pi:.1f}^\circ$")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power [fW]")

plt.show()
