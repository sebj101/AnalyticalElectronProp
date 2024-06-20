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
# Use Seaborn style
plt.style.use('seaborn')


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
B0 = 1.0  # T
L0 = 0.2  # m
# Define trap
trap = traps.HarmonicTrap(B0, L0)

# Plot pitch angle against zMax
pitchAngleArray = np.linspace(86.0, 90.0, 50) * np.pi / 180.0
zMaxArray = trap.CalcZMax(pitchAngleArray)
mainbandPower = np.zeros(len(pitchAngleArray))
sideband1Power = np.zeros(len(pitchAngleArray))
sideband2Power = np.zeros(len(pitchAngleArray))
for iP, pitchAngleInit in enumerate(pitchAngleArray):
    sampleCoarsePeriod = 1e-9
    sampleFinePeriod = 1e-10
    timeFine = np.arange(0, 1e-6, sampleFinePeriod)
    # For a given sample time we want to find the emission time of the signal
    zR = 0.05  # metres
    receiverPosition = np.array([0.0, 0.0, zR])
    mu = utils.EquivalentMagneticMoment(p0, pitchAngleInit, B0)

    teSolutions = np.zeros(len(timeFine))
    for iT, T in enumerate(timeFine):
        def func(te): return T - te - np.abs(zR -
                                             trap.GetZPosTime(te, v0, pitchAngleInit)) / sc.c
        teSolutions[iT] = fsolve(func, T)

    # Now calculate the magnetic field for these emission times
    BzTime = trap.GetBzTime(teSolutions, pitchAngleInit, v0)
    pAngleTime = utils.PitchAngleFromField(BzTime, p0, mu)
    phases = trap.GetCyclotronPhase(teSolutions, v0, pitchAngleInit)
    eVels = utils.ElectronVelocity(v0, pAngleTime, phases)
    ePos = np.array([1e-5 * np.ones(len(timeFine)), np.zeros(
        len(timeFine)), trap.GetZPosTime(teSolutions, v0, pitchAngleInit)])
    posFixed = np.array([1e-5, 0.0, 0.0])

    # Now do some actual waveguide calculations
    Z = wg.CalcTE11Impedance(trap.CalcOmega0(v0, pitchAngleInit))
    # Calculate an amplitude
    A1 = -sc.e * np.dot(wg.EFieldTE11Pos_1(posFixed,
                        normFactor1), eVels) * -Z / 2
    A2 = -sc.e * np.dot(wg.EFieldTE11Pos_2(posFixed,
                        normFactor2), eVels) * -Z / 2
    # Now multiply by local oscillator
    fLO = trap.CalcOmega0(v0, pitchAngleInit) / (2 * np.pi) - \
        (1 / sampleCoarsePeriod) / 4
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

    fA1, PowerA1 = sig.periodogram(A1, fs=1 / sampleCoarsePeriod, window='boxcar',
                                   scaling='spectrum')
    fA2, PowerA2 = sig.periodogram(A2, fs=1 / sampleCoarsePeriod, window='boxcar',
                                   scaling='spectrum')

    thisMainband = 0.0
    thisSideband1 = 0.0
    thisSideband2 = 0.0
    mainbandF = trap.CalcOmega0(v0, pitchAngleInit) / (2 * np.pi)
    sideband1F_1 = mainbandF - \
        trap.CalcOmegaAxial(pitchAngleInit, v0) / (2 * np.pi)
    sideband1F_2 = mainbandF + \
        trap.CalcOmegaAxial(pitchAngleInit, v0) / (2 * np.pi)
    sideband2F_1 = mainbandF - \
        2 * trap.CalcOmegaAxial(pitchAngleInit, v0) / (2 * np.pi)
    sideband2F_2 = mainbandF + \
        2 * trap.CalcOmegaAxial(pitchAngleInit, v0) / (2 * np.pi)
    mainbandFDownmixed = mainbandF - fLO
    sideband1F_1Downmixed = sideband1F_1 - fLO
    sideband1F_2Downmixed = sideband1F_2 - fLO
    sideband2F_1Downmixed = sideband2F_1 - fLO
    sideband2F_2Downmixed = sideband2F_2 - fLO
    for iF, f in enumerate(fA1):
        if f > mainbandFDownmixed - 3e6 and f < mainbandFDownmixed + 3e6:
            thisMainband += PowerA1[iF]
        elif (f > sideband1F_1Downmixed - 3e6 and f < sideband1F_1Downmixed + 3e6) or (f > sideband1F_2Downmixed - 3e6 and f < sideband1F_2Downmixed + 3e6):
            thisSideband1 += PowerA1[iF]
        elif (f > sideband2F_1Downmixed - 3e6 and f < sideband2F_1Downmixed + 3e6) or (f > sideband2F_2Downmixed - 3e6 and f < sideband2F_2Downmixed + 3e6):
            thisSideband2 += PowerA1[iF]

    mainbandPower[iP] = thisMainband
    sideband1Power[iP] = thisSideband1
    sideband2Power[iP] = thisSideband2

plt.figure()
plt.title("30 keV electron in a 1 T harmonic trap")
plt.plot(zMaxArray * 100, mainbandPower / np.max(mainbandPower),
         label='Mainband')
plt.plot(zMaxArray * 100, sideband1Power / np.max(mainbandPower) / 2,
         label='First-order sideband')
plt.plot(zMaxArray * 100, sideband2Power / np.max(mainbandPower) / 2,
         label='Second-order sideband')
plt.xlabel(r'$z_{\rm{max}}$ [cm]')
plt.ylabel('Relative power')
plt.ylim([0, 1.0])
plt.xlim([0, np.max(zMaxArray) * 100])
plt.legend()
plt.show()
