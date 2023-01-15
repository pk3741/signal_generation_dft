import math
from tkinter import Tk
import numpy as np
from matplotlib import pyplot as plt


def sin_gen(amplitude, frequency, phase, points):
    sin_points = np.zeros([points,1])
    for i in range(0, points):
        sin_points[i] = amplitude * math.sin(2 * math.pi * frequency * i * 1/points + phase)
    return sin_points

x = sin_gen(1,2, 0, 10)

plt.plot(np.arange(0, 10), x)
plt.show()

# root = Tk()
# root.title('Signal generator')
# root.geometry("800x450")
# root.mainloop()
