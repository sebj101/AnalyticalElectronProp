#!/usr/bin/python3

import src.CircularWaveguide as cw
import src.utils as utils
import src.Particle as part
import src.QTNMTraps as traps
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

wgRadius = 5e-3  # metres
wg = cw.CircularWaveguide(wgRadius)

# Define trap parameters
L0 = 0.35    # m
L1 = 0.5e-2  # m
B0 = 1.0     # T
trap = traps.BathtubTrap(B0, L0, L1)

# Plot the signal structure for a range of pitch angles
pitchAngleArray = np.linspace(87.0, 90.0, 7) * np.pi / 180.0

# Define receiver position
zR = 0.2  # metres
receiverPosition = np.array([0.0, 0.0, zR])
eStartPos = np.array([1e-5, 0.0, 0.0])

# Loop over the pitch angles
for pa in pitchAngleArray:
    eKE = 30e3  # eV
    electron = part.Particle(eKE, eStartPos, pitchAngle=pa)
    v0 = electron.GetSpeed()  # m/s

    # Define readout parameters
    digitizerSampleRate = 1e9  # Hz
    fLO = trap.CalcOmega0(v0, pa) / (2 * np.pi) - digitizerSampleRate / 4
    noiseTemp = 0.0  # K
    readout = siggen.Readout(digitizerSampleRate, fLO, noiseTemp)

    Omega_a = trap.CalcOmegaAxial(pa, v0)
    dopplerFactor = 1 / (1 - v0 * np.cos(pa) / sc.c)
    fieldFactor = 1 / (1 + (trap.CalcZMax(pa) / L0)**2)
    fractionalFieldChange = abs(1 - fieldFactor) + abs(1 - dopplerFactor)
    h = fractionalFieldChange * trap.GetOmegaTime(0, pa, v0) / Omega_a
    print(
        f"Calculating for pitch angle {pa * 180 / np.pi:.1f} degrees: f_a = {Omega_a / (2 * np.pi) / 1e6:.1f} MHz")
    print(
        f"Field factor = {fieldFactor:.2e}, Doppler factor = {dopplerFactor:.2e}, Total = {fractionalFieldChange:.2e}\th = {h:.2e}")

    theSignal = siggen.SignalGeneration(electron, trap, wg, 1e-6, readout,
                                        receiverPosition)

    # Make periodograms of signals
    fA1, PowerA1 = sig.periodogram(theSignal.amp1, fs=digitizerSampleRate,
                                   window='boxcar', scaling='spectrum')
    fA2, PowerA2 = sig.periodogram(theSignal.amp2, fs=digitizerSampleRate,
                                   window='boxcar', scaling='spectrum')

    # Plot the power spectrum
    plt.figure()
    plt.title(
        f"30 keV electron in a 1 T bathtub trap: ${pa * 180 / np.pi:.1f}^\circ$")
    plt.plot(fA1 / 1e6, PowerA1 * 1e15,
             label=f"$\\theta = {pa * 180 / np.pi:.1f}^\circ$")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power [fW]")

plt.show()
