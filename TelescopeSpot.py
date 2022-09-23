# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:41:51 2022

@author: mbinv
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit


def gaussian(x, amp, cen, wid, c):
    return amp * np.exp(-2 * (x - cen) ** 2 / wid ** 2) + c


ext = "pharos_prelens_2.jpg"
image = Image.open('C:/Users/mbinv/Documents/Physics/Laser Lab/9_14/' + ext).convert('L')

arr = np.asarray(image)

line = arr.sum(axis=0)

plt.plot(line.T)

line = line[0:1200]

init_vals = [32000, 350, 50, 16000]  # for [amp, cen, wid]
best_vals, covar = curve_fit(gaussian, np.arange(len(line)), line, p0=init_vals)

stdevs = np.sqrt(np.diag(covar))
amp = best_vals[0]
cen = best_vals[1]
width = best_vals[2]

fig = plt.figure()
plt.plot(np.arange(len(line)), line)
plt.title(ext)
plt.plot(np.arange(len(line)), gaussian(np.arange(len(line)), *best_vals))
plt.title(ext)
plt.show()
print(ext + ":")
print('Amp:', amp, '+-', stdevs[0])
print('Cen:', cen, '+-', stdevs[1])
print('Width:', width, '+-', stdevs[2])
print('Or', width * 5.2 / 1000, 'mm +-', stdevs[2] * 5.2 / 1000)
