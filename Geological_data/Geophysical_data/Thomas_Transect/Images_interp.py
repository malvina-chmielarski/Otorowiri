# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:34:47 2024

@author: 00098687
"""

import numpy as np
import scipy as sp
import pylab as plt
from skimage.measure import label

im = plt.imread('Figure_11.png')

#plt.imshow(im)

dum = np.zeros(np.shape(im)[:2])

dum = im[:,:,2]

dum[dum == 1] = 0.9

dum[dum <= 0.01] = 1.

dum[dum < 1] = 0

#dum[279:283,68] = 0

#set up coords
pixel_x = 10000./(513 - 367)
pixel_x0 = 68 

pixel_TWT = 4.7/(425-73)

labs = label(dum,background = 0.,connectivity = 1)
#plt.imshow(labs, vmax = 1)



Faults = np.zeros_like(labs)
Faults[labs == 36] = 1
Faults[labs == 47] = 2 
Faults[labs == 37] = 3 # Badaminnna
Faults[labs == 44] = 4# Turtle Dove?
Faults[labs == 43] = 5
Faults[labs == 38] = 6
Faults[labs == 42] = 7
Faults[labs == 39] = 8
Faults[labs == 41] = 9
Faults[labs == 40] = 10 #Muchea

#plt.imshow(Faults)
F1 = []
F2 = []
F3 = []
F4 = []
F5 = []
F6 = []
F7 = []
F8 = []
F9 = []
F10 = []

for i in range(62, np.shape(Faults)[0],1):
    for j in range(70,np.shape(Faults)[1],1):
        #print(i,j)
        if Faults[i,j] == 1:
            if j < 200 and i < 300:
                F1.append([j,i])
        if Faults[i,j] == 2:
            F2.append([j,i])
        if Faults[i,j] == 3:
            F3.append([j,i])
        if Faults[i,j] == 4:
            F4.append([j,i])
        if Faults[i,j] == 5:
            F5.append([j,i])
        if Faults[i,j] == 6:
            F6.append([j,i])
        if Faults[i,j] == 7:
            F7.append([j,i])
        if Faults[i,j] == 8:
            F8.append([j,i])
        if Faults[i,j] == 9:
            F9.append([j,i])
        if Faults[i,j] == 10:
            F10.append([j,i])                

F1 = np.array(F1)
#plt.plot(F1[:,0], F1[:,1],'r-', lw = 3)

F2 = np.array(F2)
#plt.plot(F2[:,0], F2[:,1],'r-', lw = 3)

F3 = np.array(F3)
#plt.plot(F3[:,0], F3[:,1],'r-', lw = 3)

F4 = np.array(F4)
#plt.plot(F4[:,0], F4[:,1],'r-', lw = 3)

F5 = np.array(F5)
#plt.plot(F5[:,0], F5[:,1],'r-', lw = 3)

F6 = np.array(F6)
#plt.plot(F6[:,0], F6[:,1],'r-', lw = 3)

F7 = np.array(F7)
#plt.plot(F7[:,0], F7[:,1],'r-', lw = 3)

F8 = np.array(F8)
#plt.plot(F8[:,0], F8[:,1],'r-', lw = 3)

F9 = np.array(F9)
#plt.plot(F9[:,0], F9[:,1],'r-', lw = 3)

F10 = np.array(F10)
#plt.plot(F10[:,0], F10[:,1],'r-', lw = 3)


im = plt.imread('plan_view.png')
plt.imshow(im)

x = [573,590,590,614,611,629,629,636,636,651,662,688]
y = [653,650,638,632,621,616,611,608,606,603,601,594]
x = np.array(x)
y = np.array(y)
plt.plot(x,y,lw = 3)

lat_pix =(31.3229039-30)/(632-329)
long_pix = 2/(686-214)

x = 114 + (x-214) * long_pix
y = 30 + (y-329) * lat_pix

coord = np.zeros((len(x),2))
coord[:,0] = x
coord[:,1] = -y

plt.savetxt('transect_points.dat', coord)

coords = np.array([[359431.7876,	6523505.397],
                   [366261.0667,	6525047.184],
                   [366186.8127,	6530854.706],
                   [375826.5281,	6533877.524],
                   [374553.3203,	6539186.539],
                   [381787.2866,	6541690.431],
                   [381760.0886,	6544110.109],
                   [384569.2347,	6545593.271],
                   [384558.6242,	6546561.133],
                   [390598.6072,	6548077.54],
                   [395029.9163,	6549090.553],
                   [405497.1961,	6552577.244]])

