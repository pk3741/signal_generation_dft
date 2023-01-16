import math
from tkinter import Tk
import numpy as np
from scipy.ndimage import shift
from matplotlib import pyplot as plt


def sin_gen(amplitude, frequency, phase, points):
    sin_points = np.zeros(points)
    for i in range(0, points):
        sin_points[i] = amplitude * math.sin(2 * math.pi * frequency * i * 1 / points + phase)
    return sin_points


def rect_gen(amplitude, frequency, points):
    rect_points = np.zeros(points)
    rect_points = amplitude * (sin_gen(amplitude, frequency, 0, points) > 0.0)
    return rect_points


def sawtooth_gen(amplitude, frequency, points):
    sawtooth_points = np.zeros(points)
    for i in range(0, points):
        sawtooth_points[i] = (i * 1 / points) % (1 / frequency) * frequency
    return sawtooth_points


def combine_signals(signal_a, signal_b):
    """
    TODO: REWRITE IN BETTER WAY
    """
    output = []
    signal_b = np.flip(signal_b)
    len_b = len(signal_b)
    diff = len(signal_a) - len(signal_b)

    for i in range(0, diff):
        signal_b = np.append(signal_b, np.nan)

    for i in range(0, 2 * len_b):
        signal_a = np.append(signal_a, np.nan)

    signal_a = np.roll(signal_a, len_b)

    for i in range(0, len(signal_a) - len_b):
        new = signal_a[i:i + len(signal_b)]
        value = np.nan
        for i in range(0, len(new)):
            if not np.isnan(new[i]) and not np.isnan(signal_b[i]):
                if np.isnan(value):
                    value = 0
                value += new[i] * signal_b[i]
        if not np.isnan(value):
            output.append(value)
    output = np.asarray(output)
    return output


def blackman_window(samples):
    n=samples
    windowed = np.zeros(n)
    for i in np.arange(0, n):
        windowed[i] = 0.42 - 0.5 * math.cos((2*math.pi*i)/n) + 0.08 * math.cos((4*math.pi*i)/n)
    return windowed


def hanning_window(samples):
    n = samples
    windowed = np.zeros(n)
    for i in np.arange(0, n):
        windowed[i] = 0.5 - 0.5 * math.cos((2 * math.pi * i) / n)
    return windowed


def hamming_window(samples):
    n = samples
    windowed = np.zeros(n)
    for i in np.arange(0, n):
        windowed[i] = 0.54 - 0.46 * math.cos((2 * math.pi * i) / n)
    return windowed


def bartlett_window(samples):
    n = samples
    windowed = np.zeros(n)
    for i in np.arange(0, n):
        windowed[i] = (2/(n-1)) * (((n-1)/2) - abs(i - ((n-1)/2)))
    return windowed



sinus_signal = sin_gen(1, 2, 0, 1000)
hanning = bartlett_window(51)
hanning_org = np.bartlett(51)

#
fig, axs = plt.subplots(2, sharex=True, sharey=True)
axs[0].plot(np.arange(0,51), hanning)
axs[1].plot(np.arange(0, 51), hanning_org)

plt.show()

# root = Tk()
# root.title('Signal generator')
# root.geometry("800x450")
# root.mainloop()
