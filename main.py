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
    signal_b = np.flip(signal_b)
    len_b = len(signal_b)
    diff = len(signal_a) - len(signal_b)

    for i in range(0, diff):
        signal_b = np.append(signal_b, np.nan)

    for i in range(0, 2 * len_b):
        signal_a = np.append(signal_a, np.nan)

    signal_a = np.roll(signal_a, 2)
    print("a", signal_a)
    print("b", signal_b)

    for i in range(0, len(signal_a) - len_b):
        new = signal_a[i:i + len(signal_b)]
        value = np.nan

        value = np.nan
        for i in range(0, len(new)):
            if not np.isnan(new[i]) and not np.isnan(signal_b[i]):
                if np.isnan(value):
                    value = 0
                value += new[i] * signal_b[i]
        if not np.isnan(value):
            print(value)


sinus_signal = sin_gen(1, 2, 0, 1000)
square_signal = rect_gen(4, 4, 1000)

a = np.array([1, 2, 3])
b = np.array([1, 2])

print(np.convolve(a, b))
combine_signals(a, b)

#
# plt.plot(np.arange(0, 1000), actual_signal)
# plt.show()

# root = Tk()
# root.title('Signal generator')
# root.geometry("800x450")
# root.mainloop()
