import math
from tkinter import Tk
import numpy as np
from matplotlib import pyplot as plt


def sin_gen(amplitude, frequency, phase, points):
    sin_points = np.zeros([points, 1])
    for i in range(0, points):
        sin_points[i] = amplitude * math.sin(2 * math.pi * frequency * i * 1 / points + phase)
    return sin_points


def rect_gen(amplitude, frequency, points):
    rect_points = np.zeros([points, 1])
    rect_points = amplitude * (sin_gen(amplitude, frequency, 0, points) > 0.0)
    return rect_points


def sawtooth_gen(amplitude, frequency, points):
    sawtooth_points = np.zeros([points, 1])
    for i in range(0, points):
        sawtooth_points[i] = (i * 1 / points) % (1 / frequency) * frequency
    return sawtooth_points


def combine_signals(signal_a, signal_b, points):
    combined_points = np.zeros([points, 1])
    for i in range(0, points):
        combined_points[i]= signal_a[i] * signal_b[i]
    return combined_points



actual_signal = np.zeros([1000, 1])
sawtooth_signal = np.ones([1000, 1])
sinus_signal = sin_gen(1, 2, 0, 1000)

for i in range(0, 1000):
    actual_signal[i] = sawtooth_signal[i] * sinus_signal[i]

plt.plot(np.arange(0, 1000), actual_signal)
plt.show()

# root = Tk()
# root.title('Signal generator')
# root.geometry("800x450")
# root.mainloop()
