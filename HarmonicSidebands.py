#!/usr/bin/python3

import src.CircularWaveguide as cw
import src.utils as utils
import src.Particle as part
import src.QTNMTraps as traps
import src.CircularWaveguide as cw
import src.SignalGeneration as siggen
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.signal as sig
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

eKE = 30e3  # eV
startPos = np.array([1e-5, 0.0, 0.0])

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

zR = 0.05  # metres
receiverPosition = np.array([0.0, 0.0, zR])
for iP, pitchAngleInit in enumerate(pitchAngleArray):
    # Generate the electron
    electron = part.Particle(eKE, startPos, pitchAngle=pitchAngleInit)
    v0 = electron.GetSpeed()    # m/s
    p0 = electron.GetMomentum()  # kg m/s

    digitizerSampleRate = 1e9  # Hz
    fLO = trap.CalcOmega0(v0, pitchAngleInit) / \
        (2 * np.pi) - digitizerSampleRate / 4
    noiseTemp = 0.0  # K
    readout = siggen.Readout(digitizerSampleRate, fLO, noiseTemp)
    theSignal = siggen.SignalGeneration(electron, trap, wg, 1e-6, readout,
                                        receiverPosition)

    fA1, PowerA1 = sig.periodogram(theSignal.amp1, fs=digitizerSampleRate,
                                   window='boxcar', scaling='spectrum')
    fA2, PowerA2 = sig.periodogram(theSignal.amp2, fs=digitizerSampleRate,
                                   window='boxcar', scaling='spectrum')

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
