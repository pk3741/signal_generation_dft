import cmath
import math
import tkinter
from tkinter import *

import numpy as np
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


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
        output = complex(0, 0)
        for k in range(0, pts, 1):
            output += data[k] * cmath.exp(-1j * k * n * 2 * math.pi / pts)
        out[n] = output
    return out


class Application(tkinter.Frame):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        root.title("Wave generation tool")
        root.geometry("800x900")
        self.data = []
        self.createWidgets()
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.plot1 = 0
        self.canvas = FigureCanvasTkAgg(self.fig,
                                        master=root)
        self.canvas.get_tk_widget().grid(column=0, columnspan=9, row=9, sticky=tkinter.S)

    def createWidgets(self):
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=1)
        root.columnconfigure(4, weight=1)
        root.columnconfigure(5, weight=1)
        root.columnconfigure(6, weight=1)
        root.columnconfigure(7, weight=1)
        root.columnconfigure(8, weight=1)

        label1 = Label(root, text='Amplitude:')
        label1.grid(column=0, row=0, sticky=tkinter.W)

        amplitude = Text(root, height=1, width=10)
        amplitude.insert("1.0", "1")
        amplitude.grid(column=1, row=0, sticky=tkinter.W)

        label2 = Label(root, text='Frequency:')
        label2.grid(column=2, row=0, sticky=tkinter.W)

        freq = Text(root, height=1, width=10)
        freq.insert("1.0", "1")
        freq.grid(column=3, row=0, sticky=tkinter.W)

        label3 = Label(root, text='Phase:')
        label3.grid(column=4, row=0, sticky=tkinter.W)

        phase = Text(root, height=1, width=10)
        phase.insert("1.0", "0")
        phase.grid(column=5, row=0, sticky=tkinter.W)

        label4 = Label(root, text='Steps:')
        label4.grid(column=6, row=0, sticky=tkinter.W)

        steps = Text(root, height=1, width=11)
        steps.insert("1.0", "256")
        steps.grid(column=7, row=0, sticky=tkinter.W)

        # self.plot_button = Button(master=root,
        #                           command=lambda: self.plot(),
        #                           height=2,
        #                           width=10,
        #                           text="Plot", )
        # self.plot_button.grid(column=0, row=4, sticky=tkinter.W)

        self.plot_button2 = Button(master=root,
                                   command=lambda: self.add_data(
                                       sin_gen(float(amplitude.get('1.0', 'end')), float(freq.get('1.0', 'end')),
                                               float(phase.get('1.0', 'end')), int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Add sinus")
        self.plot_button2.grid(column=0, row=1, sticky=tkinter.NW)

        self.plot_button3 = Button(master=root,
                                   command=lambda: self.add_data(
                                       rect_gen(float(amplitude.get('1.0', 'end')), float(freq.get('1.0', 'end')),
                                                int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Add rect")
        self.plot_button3.grid(column=1, row=1, sticky=tkinter.NW)

        self.plot_button4 = Button(master=root,
                                   command=lambda: self.add_data(
                                       sawtooth_gen(float(amplitude.get('1.0', 'end')), float(freq.get('1.0', 'end')),
                                               int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Add sawtooth")
        self.plot_button4.grid(column=2, row=1, sticky=tkinter.NW)

        self.plot_button5 = Button(master=root,
                                   command=lambda: self.add_window(blackman_window(int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Blackman")
        self.plot_button5.grid(column=0, row=2, sticky=tkinter.NW)

        self.plot_button6 = Button(master=root,
                                   command=lambda: self.add_window(hanning_window(int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Hanning")
        self.plot_button6.grid(column=1, row=2, sticky=tkinter.NW)

        self.plot_button7 = Button(master=root,
                                   command=lambda: self.add_window(hamming_window(int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Hamming")
        self.plot_button7.grid(column=2, row=2, sticky=tkinter.NW)

        self.plot_button8 = Button(master=root,
                                   command=lambda: self.add_window(bartlett_window(int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Bartlett")
        self.plot_button8.grid(column=3, row=2, sticky=tkinter.NW)


        self.plot_button9 = Button(master=root,
                                   command=lambda: self.make_dft(),
                                   height=2,
                                   width=10,
                                   text="DFT")
        self.plot_button9.grid(column=0, row=7, sticky=tkinter.NW)

        self.plot_button9 = Button(master=root,
                                   command=lambda: self.clearplot(),
                                   height=2,
                                   width=10,
                                   text="Clear")
        self.plot_button9.grid(column=0, row=9, sticky=tkinter.NW)

        self.plot_button10 = Button(master=root,
                                   command=lambda: self.add_conv(
                                       sin_gen(float(amplitude.get('1.0', 'end')), float(freq.get('1.0', 'end')),
                                               float(phase.get('1.0', 'end')), int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Conv sin")
        self.plot_button10.grid(column=5, row=1, sticky=tkinter.NW)

        self.plot_button11 = Button(master=root,
                                   command=lambda: self.add_conv(
                                       rect_gen(float(amplitude.get('1.0', 'end')), float(freq.get('1.0', 'end')),
                                                int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=10,
                                   text="Conv rect")
        self.plot_button11.grid(column=6, row=1, sticky=tkinter.NW)

        self.plot_button12 = Button(master=root,
                                   command=lambda: self.add_conv(
                                       sawtooth_gen(float(amplitude.get('1.0', 'end')), float(freq.get('1.0', 'end')),
                                                    int(steps.get('1.0', 'end')))),
                                   height=2,
                                   width=12,
                                   text="Conv sawtooth")
        self.plot_button12.grid(column=7, row=1, sticky=tkinter.NW)

        self.plot_button13 = Button(master=root,
                                    command=lambda: self.spectrum(freq, steps),
                                    height=2,
                                    width=10,
                                    text="Spectrum")
        self.plot_button13.grid(column=0, row=8, sticky=tkinter.NW)

    # def plot(self):
    #     self.fig.clear()
    #     self.plot1 = self.fig.add_subplot(111).plot(self.data)
    #     self.canvas.draw()

    def clearplot(self):
        self.data = []
        self.fig.clear()
        self.plot1 = self.fig.add_subplot(111).plot(self.data)
        self.canvas.draw()

    def replot(self):
        self.fig.clear()
        self.plot1 = self.fig.add_subplot(111).plot(self.data)
        self.canvas.draw()

    def replot_spectrum(self, x, y):
        self.fig.clear()
        self.plot1 = self.fig.add_subplot(111).plot(x, y)
        self.canvas.draw()

    def add_data(self, newdata):
        if len(self.data)==0:
            for i in range(0, len(newdata)):
                self.data.append(newdata[i])
        else:
            for i in range(0, len(self.data)):
                self.data[i]=self.data[i]+newdata[i]
        self.replot()

    def add_window(self, window):
        for i in range(0, len(self.data)):
            self.data[i]*=window[i]
        self.replot()

    def add_conv(self, conv):
        self.data = combine_signals(self.data, conv)
        self.replot()

    def make_dft(self):
        self.data = abs(dft(self.data))
        self.replot()

    def spectrum(self, freq, steps):
        xf = np.fft.fftfreq(int(steps.get('1.0','end')), 1 / float(steps.get('1.0', 'end')))
        new_data = []
        self.replot_spectrum(xf, self.data)

root = tkinter.Tk()
app = Application(master=root)
app.mainloop()
