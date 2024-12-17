#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import glob
import flopy
import math
import numbers
import matplotlib.image as mpimg
from math import pi

from scipy.interpolate import interp1d
import pandas as pd
import pickle
import shapefile
import matplotlib as mpl
from collections import OrderedDict
import subprocess as sup
from flopy.utils.triangle import Triangle as Triangle
import geopandas as gpd
from shapely.geometry import LineString,Point,Polygon,shape
import fiona
from datetime import datetime

# logfunc = lambda e: np.log10(e)

def find_kji(cell,nlay,nrow,ncol): #cellid is zerobased
    cellid = cell - 1
    k = math.floor(cellid/(ncol*nrow)) # Zero based
    j = math.floor((cellid - k*ncol*nrow)/ncol) # Zero based
    i = cellid - k*ncol*nrow - j*ncol
    return(k,j,i) # ZERO BASED!

def find_cellid(k,j,i,nlay,nrow,ncol): # returns zero based cell id
    return(i + j*ncol + k*ncol*nrow)

def x_to_col(x, delr):
    return(int(x/delr))

def y_to_row(y, delc):
    return(int(y/delc))

def z_to_lay(z, delz, zmax):
    return(int((zmax - z)/delz))

def lay_to_z(botm, top, lay, icpl=0):
    pillar = botm[:,icpl]
    if lay != 0:
        cell_top = pillar[lay-1]
    if lay == 0:
        cell_top = top[icpl]
    cell_bot = pillar[lay]
    dz = cell_top - cell_bot
    z = cell_top - dz/2        
    return(z)

def RCA(x,X):
    #x is a point that we want to see if it is in  the polygon
    #X is the vertices of the polygon
    dim=len(X)
    #print(dim)
    #we are going to run a line to the left and see is it intercepts the line between 2 vertices
    ct=0   
    for i in range(dim-1):
        #print(X[i])
        if X[i][1]<X[i+1][1]:
            if x[1]>X[i][1] and  x[1]<=X[i+1][1]:
                xx=X[i][0]+(x[1]-X[i][1])/(X[i+1][1]-X[i][1])*(X[i+1][0]-X[i][0])
                if xx >= x[0]:
                    ct= ct + 1
        else:
            if x[1]>X[i+1][1] and  x[1]<=X[i][1]:
                xx=X[i][0]+(x[1]-X[i][1])/(X[i+1][1]-X[i][1])*(X[i+1][0]-X[i][0])
                if xx >= x[0]:
                    ct= ct + 1           
    return(ct % 2)

def disucell_to_disvcell(geomodel, disucell): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
    return(disvcell)

def disvcell_to_disucell(geomodel, disvcell): # zerobased
    disucell = geomodel.cellid_disu.flatten()[disvcell]
    return disucell

def xyz_to_discell(x, y, x0, y1, dx, dy):
    col = int((x - x0)/dx)
    row = int((y1 - y)/dy)
    cell_coords = (dx/2 + col*dx, (y1 - dy/2) - row * dy)
    #print('Actual xy = %s %s, Cell col is %i row is %i, Cell centre is %s' %(x,y,col,row,cell_coords))
    return (col, row, cell_coords)

def disucell_to_xyz(geomodel, disucell): # zerobased
    import math
    nlay, ncpl = geomodel.cellid_disv.shape
    disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
    lay  = math.floor(disvcell/ncpl) # Zero based
    icpl = math.floor(disvcell - lay * ncpl) # Zero based
    x = geomodel.vgrid.xcellcenters[icpl]
    y = geomodel.vgrid.ycellcenters[icpl]
    z = geomodel.vgrid.zcellcenters[lay, icpl]
    return(x,y,z)

def xyz_to_disucell(geomodel, x,y,z): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    point = Point(x,y,z)
    lay, icpl = geomodel.vgrid.intersect(x,y,z)
    disvcell = icpl + lay*ncpl
    disucell = geomodel.cellid_disu.flatten()[disvcell]
    return disucell
        
# Writing and processing MODFLOW arrays

def write_input_files(gwf,modelname):
    import flopy
    headfile = '{}.hds'.format(modelname)
    head_filerecord = [headfile]
    budgetfile = '{}.bud'.format(modelname)
    budget_filerecord = [budgetfile]
    saverecord, printrecord = [('HEAD', 'ALL'), ('BUDGET', 'ALL')], [('HEAD', 'ALL')]
    oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(gwf, pname='oc', saverecord=saverecord, head_filerecord=head_filerecord,
                                                budget_filerecord=budget_filerecord, printrecord=printrecord)

def ch_flow(chdflow):
    flow_in, flow_out = 0., 0.
    for j in range(len(chdflow)):        
        if chdflow[j][2]>0: flow_out += chdflow[j][2]
        if chdflow[j][2]<0: flow_in  += chdflow[j][2]      
    return((flow_in, flow_out))

def get_q_disu(spd, flowja, gwf, staggered):
    import math
    import flopy
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spd, gwf)
    # if cross-connections, recalculate qx taking into account overlap areas
    if staggered:
        gp = d2d.get_gridprops_disu6()
        iac = gp["iac"]
        ja = gp["ja"]
        ihc = gp["ihc"]
        topbycell = gp["top"]
        botbycell = gp["bot"]
        hwva = gp["hwva"]
        iconn = -1
        icell = -1
        for il in iac:
            icell += 1
            qxnumer = 0.
            qxdenom = 0.
            for ilnbr in range(il):
                iconn += 1
                if ihc[iconn] == 2:
                    inbr = ja[iconn]
                    if (inbr == icell):
                        continue
                    dz = min(topbycell[icell], topbycell[inbr]) - max(botbycell[icell], botbycell[inbr])
                    qxincr = flowja[iconn] / (hwva[iconn] * dz)
                    # equal weight given to each face, but could weight by distance instead
                    if (inbr < icell):
                        qxnumer += qxincr
                    else:
                        qxnumer -= qxincr
                    qxdenom += 1.
            qx[icell] = qxnumer / qxdenom

    #print(len(spd))
    qmag, qdir = [], []
    for i in range(len(spd)):
        qmag.append(np.sqrt(qx[i]**2 + qy[i]**2 + qz[i]**2))
        qdir.append(math.degrees(math.atan(qz[i]/qx[i])))      
    return(qmag,qx,qy,qz,qdir)

