import math
from tkinter import Tk
import numpy as np
from scipy.ndimage import shift
from matplotlib import pyplot as plt
import cmath


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
    n = samples
    windowed = np.zeros(n)
    for i in np.arange(0, n):
        windowed[i] = 0.42 - 0.5 * math.cos((2 * math.pi * i) / n) + 0.08 * math.cos((4 * math.pi * i) / n)
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
        windowed[i] = (2 / (n - 1)) * (((n - 1) / 2) - abs(i - ((n - 1) / 2)))
    return windowed


def dft(data):
    pts = len(data)
    out = np.ndarray(pts, dtype=np.complex128)
    for n in range(0, pts, 1):
        output = complex(0,0)
        for k in range(0, pts, 1):
            output += data[k] * cmath.exp(-1j*k*n*2*math.pi/pts)
        out[n]=output
    return out


sinus_signal = sin_gen(1,1 ,0,128)
sinus_dft = dft(sinus_signal)
test = np.fft.fft(sinus_signal, n=128, axis=0)

fig, axs = plt.subplots(2)
axs[0].plot(np.arange(0, 128), abs(test))
axs[1].plot(np.arange(0, 128), abs(sinus_dft))
plt.show()

# root = Tk()
# root.title('Signal generator')
# root.geometry("800x450")
# root.mainloop()
