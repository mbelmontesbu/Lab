# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:04:09 2022

@author: mbinv
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from scipy.optimize import curve_fit
from scipy import ndimage
import glob
import pandas as pd
from matplotlib.offsetbox import AnchoredText

#fname = "5uJ25kHz.jpg"
direc = 'C:/Users/mbinv/Documents/Physics/9_21/'

Pos1DF = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])
Pos1DFerr = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])
Pos2DF = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])
Pos2DFerr = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])
Pos1DF2 = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])
Pos1DFerr2 = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])
Pos2DF2 = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])
Pos2DFerr2 = pd.DataFrame(index=['25kHz','50kHz','100kHz','200kHz'],columns=['20uJ','40uJ','60uJ','80uJ','100uJ'])

def gaussian(x, amp, cen, wid, c):
    return amp * np.exp(-2 * (x - cen) ** 2 / wid ** 2) + c

print("opening images in: " + direc)
for file in glob.iglob(f"{direc}/*.jpg"):
    fname = file[0:-4]
    image = Image.open(file).convert("L")
    arr = np.asarray(image)
    Nx = len(arr[:,0])
    Ny = len(arr[0,:])
    x = np.arange(Nx)
    y = np.arange(Ny)

    Threshold = 0.75

    center = ndimage.center_of_mass((arr/255)>Threshold)
    yc, xc = center
    xc, yc = int(xc), int(yc)

    ycarr = arr[xc,:]
    xcarr = arr[:,yc]

    '''
    figure, axes = plt.subplots(nrows=3, ncols=1)
    figure.tight_layout()
    plt.subplot(3, 1, 1)
    plt.title(fname)
    plt.imshow(arr,cmap=cm.gray)
    plt.plot(xc,yc,'g.')

    #plots the gaussian along the x and y lines cenetred about the beam
    plt.subplot(3, 1, 2)
    plt.plot(x,xcarr)
    plt.title('Center of beam along x')

    plt.subplot(3, 1, 3)
    plt.plot(y,ycarr)
    plt.title('Center of beam along y')
    plt.show()

    #%%

    #3d surface plot of beam vs brightness
    x, y = np.meshgrid(y, x)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, arr, cmap=cm.magma)
    plt.title(fname)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Gamma')
    plt.show()
    #plt.savefig(direc + '3d' + fname)
    #%%
'''
    init_vals = [xcarr.max(), xc, 450, 20]  # for [amp, cen, wid]
    best_vals, covar = curve_fit(gaussian, np.arange(len(xcarr)), xcarr ,p0=init_vals)

    stdevs = np.sqrt(np.diag(covar))
    amp = best_vals[0]
    cen = best_vals[1]
    width = best_vals[2] * 5.2/1000

    fig, ax = plt.subplots()
    plt.plot(np.arange(len(xcarr)),xcarr)
    plt.title(fname + " - Along x")
    plt.plot(np.arange(len(xcarr)), gaussian(np.arange(len(xcarr)), *best_vals))
    #plt.annotate("Waist " + width + " +- " + stdevs[2] * 5.2/1000 + " mm", (0,0) )
    at = AnchoredText(
        f"Waist {width:0.2f} +- {stdevs[2]*5.2/100:0.2f} mm", prop=dict(size=15), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    #plt.show()
    plt.savefig((fname + "_x.png"), format='png')
    """
    print(fname + " - Along x" + ":")
    print('Amp:', amp, '+-',stdevs[0])
    print('Cen:', cen, '+-',stdevs[1])
    print('Width:', width, '+-',stdevs[2])
    print('Or', width * 5.2/1000, 'mm +-', stdevs[2]*5.2/1000)
    """
    plt.close()

    init_vals = [ycarr.max(), yc, 450, 20]  # for [amp, cen, wid]
    best_vals, covar = curve_fit(gaussian, np.arange(len(ycarr)), ycarr ,p0=init_vals)

    stdevs1 = np.sqrt(np.diag(covar))
    amp1 = best_vals[0]
    cen1 = best_vals[1]
    width1 = best_vals[2] * 5.2/1000

    fig, ax = plt.subplots()
    plt.plot(np.arange(len(ycarr)),ycarr)
    plt.title(fname + " - Along y")
    plt.plot(np.arange(len(ycarr)), gaussian(np.arange(len(ycarr)), *best_vals))
    at = AnchoredText(
        f"Waist {width1:0.2f} +- {stdevs1[2]*5.2/100:0.2f} mm", prop=dict(size=15), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    #plt.show()
    plt.savefig((fname + "_y.png"), format='png')
    """
    print(fname + " - Along y" + ":")
    print('Amp:', amp1, '+-',stdevs[0])
    print('Cen:', cen1, '+-',stdevs[1])
    print('Width:', width1, '+-',stdevs[2])
    print('Or', width1 * 5.2/1000, 'mm +-', stdevs[2]*5.2/1000) 
    """
    plt.close()

    arr = np.asarray(image)

    line = arr.sum(axis=0)

    line = line[0:1200]

    init_vals = [120000, 700, 400, 16000]  # for [amp, cen, wid]
    best_vals, covar = curve_fit(gaussian, np.arange(len(line)), line, p0=init_vals)

    stdevs2 = np.sqrt(np.diag(covar))
    amp = best_vals[0]
    cen = best_vals[1]
    width2 = best_vals[2] * 5.2/1000

    avg_wid = (width + width1)/2
    #print(avg_wid-width2* 5.2/1000)
    if "Pos1" in fname:
        for i in Pos1DF.columns:
            if i in fname:
                for j in Pos1DF.index:
                    if j in fname:
                        Pos1DF.loc[j,i] = avg_wid
                        Pos1DFerr.loc[j,i] = stdevs[2] * 5.2/1000
        for i in Pos1DF2.columns:
            if i in fname:
                for j in Pos1DF2.index:
                    if j in fname:
                        Pos1DF2.loc[j,i] = width2 * 5.2/1000
                        Pos1DFerr2.loc[j, i] = stdevs2[2] * 5.2/1000
    if "Pos2" in fname:
        for i in Pos2DF.columns:
            if i in fname:
                for j in Pos2DF.index:
                    if j in fname:
                        Pos2DF.loc[j,i] = avg_wid
                        Pos2DFerr.loc[j, i] = stdevs1[2]
        for i in Pos2DF2.columns:
            if i in fname:
                for j in Pos2DF2.index:
                    if j in fname:
                        Pos2DF2.loc[j,i] = width2
                        Pos2DFerr2.loc[j, i] = stdevs[2] * 5.2/1000



print("Plotting Spot Sizes")
for i in Pos1DF.columns:
    plt.plot(Pos1DF.loc[:,i], ls = '--', label='Pos1')
    plt.plot(Pos1DF2.loc[:, i], ls='--', label='Pos1_2')
    plt.plot(Pos2DF.loc[:, i], ls = '--', label='Pos2')
    plt.plot(Pos2DF2.loc[:, i], ls='--', label='Pos2_2')
    plt.legend()
    plt.title(i+" - Pos 1 & 2")
    plt.ylabel("Spot Size (mm)")
    plt.xlabel("Rep. Rate")
    plt.savefig((direc + i + "RepRate.png"), format='png')
    plt.close()

    foo = 0
    for j in Pos1DF.loc[:,i]:
        if j > foo:
            foo = j
    for j in Pos2DF.loc[:,i]:
        if j > foo:
            foo = j
    for k in Pos1DF2.loc[:,i]:
        if k > foo:
            foo = k
    for l in Pos2DF2.loc[:,i]:
        if l > foo:
            foo = l

    plt.plot(Pos1DF.index, Pos1DF.loc[:,i]/foo, ls = '--', label='Pos1')
    plt.plot(Pos1DF2.index,Pos1DF2.loc[:,i]/foo, ls = '--', label='Pos1_2')
    plt.plot(Pos2DF.index,Pos2DF.loc[:, i]/foo, ls = '--', label='Pos2')
    plt.plot(Pos2DF2.index,Pos2DF2.loc[:, i] / foo, ls='--', label='Pos2_2')
    plt.legend()
    plt.title(i + " - Pos 1 & 2")
    plt.ylabel("Normalized Spot Size")
    plt.xlabel("Rep. Rate")
    plt.savefig((direc + i + "RepRateNorm.png"), format='png')
    plt.close()

for i in Pos1DF.index:
    plt.plot(Pos1DF.loc[i,:], ls = '--', label='Pos1')
    plt.plot(Pos1DF2.loc[i, :], ls='--', label='Pos1_2')
    plt.plot(Pos2DF.loc[i,:], ls = '--', label='Pos2')
    plt.plot(Pos2DF2.loc[i, :], ls='--', label='Pos2_2')
    plt.legend()
    plt.title(i+" - Pos 1 & 2")
    plt.ylabel("Spot Size (mm)")
    plt.xlabel("Pulse Energy")
    plt.savefig((direc + i + "PulseE.png"), format='png')
    plt.close()

    foo = 0
    for j in Pos1DF.loc[i,:]:
        if j > foo:
            foo = j
    for j in Pos2DF.loc[i,:]:
        if j > foo:
            foo = j
    for k in Pos1DF2.loc[i,:]:
        if k > foo:
            foo = k
    for l in Pos2DF2.loc[i,:]:
        if l > foo:
            foo = l
    #print("foo: ", foo)
    #print(Pos1DF.loc[i,:])
    #print(Pos1DF.loc[i,:] / foo)
    plt.plot(Pos1DF.columns, Pos1DF.loc[i,:] / foo, ls = '--', label='Pos1')
    plt.plot(Pos1DF2.columns, Pos1DF2.loc[i, :] / foo, ls='--', label='Pos1_2')
    plt.plot(Pos2DF.columns, Pos2DF.loc[i,:] / foo, ls = '--', label='Pos2')
    plt.plot(Pos2DF2.columns, Pos2DF2.loc[i, :] / foo, ls='--', label='Pos2_2')
    plt.legend()
    plt.title(i + " - Pos 1 & 2")
    plt.ylabel("Normalized Spot Size")
    plt.xlabel("Pulse Energy")
    plt.savefig((direc + i + "PulseENorm.png"), format='png')
    plt.close()

print("done")





