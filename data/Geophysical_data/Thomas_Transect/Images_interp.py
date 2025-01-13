# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:34:47 2024

@author: 00098687
"""

import numpy as np
import scipy as sp
import pylab as plt
from skimage.measure import label

class transect:
    
    def __init__(self,coord,px0,px1,z1,z2,pz1,pz2):
        self.coord = coord
        self.px0 = px0
        self.px1 = px1
        self.z1 = z1
        self.z2 = z2
        self.pz1 = pz1
        self.pz2 = pz2
        self.dzdp = (z2-z1)/(pz2-pz1)
        
        L = []

        for i in range(len(coords)-1):
            L.append(np.sqrt((coords[i,0]-coords[i+1,0])**2 + 
                             (coords[i,1]-coords[i+1,1])**2))

        L = np.array(L)
        self.L = np.cumsum(L)/np.sum(L)    
        
    def xyz(self,px,pz):
        frac = (px - self.px0)/(self.px1 - self.px0)
        n = np.argmin(np.abs(self.L-frac))
        if self.L[n]> frac: n-=1
        if n< 0: n = 0
        frac2 = (frac-self.L[n])/(self.L[n+1]-self.L[n])
        x = (1.-frac2) * self.coord[n ,0] + frac2 * self.coord[n+1,0]
        y = (1.-frac2) * self.coord[n ,1] + frac2 * self.coord[n+1,1]
        z = self.z1 + (pz - self.pz1) * self.dzdp
        return(x,y,z)
        
                
        
        
        
        

im = plt.imread('Figure_11.png')

#plt.imshow(im)

#print(ljhvljv)

dum = np.zeros(np.shape(im)[:2])

dum = np.copy(im[:,:,2])

dum[dum == 1] = 0.9

dum[dum <= 0.01] = 1.

dum[dum < 1] = 0

dum2 = np.copy(im[:,:,2])
dum2[dum2 < 0.134] = 1.01
dum2[dum2<1] = 0
dum3 = np.copy(im[:,:,0])
dum2[dum3 < 0.874] = 0.
dum4 = np.copy(im[:,:,1])
dum2[dum4 != 0] = 0

#plt.imshow(dum2) 

BU = []
for i in range(np.shape(dum)[0]):
    for j in range(np.shape(dum)[1]):
        if dum2[i,j] > 0.1:
            BU.append([j,i])
BU = np.array(BU)

dum2 = np.copy(im[:,:,0])
dum3 = np.copy(im[:,:,1])
dum4 = np.copy(im[:,:,2])


#print(jhl)
dum2[dum3>0.075] = 0
dum2[dum3<0.073] = 0
#plt.imshow(dum2)
dum2[dum4>0.39] = 0
dum2[dum4<0.3] = 0


dum2[dum2>0.19] = 1
#plt.imshow(dum2) 
SPS = []

for i in range(np.shape(dum)[0]):
    for j in range(np.shape(dum)[1]):
        if dum2[i,j] > 0.1:
            SPS.append([j,i])
            
SPS = np.array(SPS)

dum2 = np.copy(im[:,:,0])
dum3 = np.copy(im[:,:,1])
dum4 = np.copy(im[:,:,2])


#print(jhl)
dum2[dum3<0.95] = 0
dum2[dum3>0.96] = 0
plt.imshow(dum2)
dum2[dum4<0.51] = 0
dum2[dum4>0.52] = 0


dum2[dum2>0.96] = 1

plt.imshow(dum2)

YAR = []

for i in range(np.shape(dum)[0]):
    for j in range(np.shape(dum)[1]):
        if dum2[i,j] > 0.1:
            YAR.append([j,i])
            
YAR = np.array(YAR)

#print(ugvvluv)

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

plt.imshow(Faults)
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
plt.plot(F1[:,0], F1[:,1],'r-', lw = 3)

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
plt.plot(F10[:,0], F10[:,1],'r-', lw = 3)


"""im = plt.imread('plan_view.png')
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

plt.savetxt('transect_points.dat', coord)"""

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

px0 = 68
px1 = 815

BoorP = 184
AM3P = 119
Boorz = 0 - 2333.
AM3z = 27. - 729.

T = transect(coords,px0,px1,Boorz,AM3z,BoorP,AM3P)
x,y,z = T.xyz(69,AM3P)

x,y,z = [],[],[]
for i in range(len(F3)):
    xx,yy,zz = T.xyz(F3[i,0],F3[i,1])
    x.append(xx)
    y.append(yy)
    z.append(zz)

X = []
for i in range(len(BU)):
    x,y,z = T.xyz(BU[i,0],BU[i,1])
    X.append([x,y,z])
X = np.array(X)
plt.savetxt('BU_thomas.dat',X)

X = []
for i in range(len(YAR)):
    x,y,z = T.xyz(YAR[i,0],YAR[i,1])
    X.append([x,y,z])
X = np.array(X)
plt.savetxt('Yar_base_thomas.dat',X)


X = []
for i in range(len(SPS)):
    x,y,z = T.xyz(SPS[i,0],SPS[i,1])
    X.append([x,y,z])
X = np.array(X)
plt.savetxt('Top_SPS_thomas.dat',X)
