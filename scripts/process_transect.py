import pandas as pd
from shapely.geometry import LineString
import geopandas as gpd
import numpy as np
import flopy
from loopflopy.mesh import Mesh

class Transect:
    def __init__(self, crs, name, x0, y0, x1, y1, ncol, delc): # lay bottoms 1D array of layer bottoms

        self.name = name
        self.crs = crs
        
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.length = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        #self.angrot = np.arctan2(self.y1 - self.y0, self.x1 - self.x0)
        self.angrot = np.degrees(np.arctan((self.y0 - self.y1)/(self.x0 - self.x1)))
        print('angrot ', self.angrot)
        
        # Horizontal discretisation
        self.ncol = ncol #len(delr) # number of columns in transect
        self.nrow = 1
        self.ncpl = self.ncol * self.nrow
        delr = self.length / ncol
        self.delr = delr * np.ones(self.ncol, dtype=float)
        self.delc = delc * np.ones(self.nrow, dtype=float)

        sg = flopy.discretization.StructuredGrid(delr=self.delr, delc=self.delc, 
                                                 top=np.ones((self.nrow, self.ncol), dtype=float), 
                                                 botm=np.zeros((1, self.nrow, self.ncol), dtype=float), 
                                                 xoff = self.x0, yoff = self.y0, angrot = self.angrot)

        # Turn structured grid into DISV (its what I know!)
        xyzcenters = sg.xyzcellcenters
        xcenters = xyzcenters[0][0]
        ycenters = xyzcenters[1][0]
        iverts = sg.iverts
        verts = sg.verts

        cell2d = []
        xcyc = [] # added 
        for icpl in range(self.ncpl):
            xc = xcenters[icpl]
            yc = ycenters[icpl]
            iv1, iv2, iv3, iv4 = iverts[icpl]
            cell2d.append([icpl, xc, yc, 5, iv1, iv2, iv3, iv4, iv1])
            xcyc.append((xc, yc))
        
        vertices = []
        for v in range(len(verts)):
            i,j = verts[v]
            vertices.append([v, i, j]) # need to make 1 based
  
        self.coords = [[x0, y0], [x1, y1]]
        self.ls = LineString([[x0, y0], [x1, y1]])
        self.gdf = gpd.GeoDataFrame({'geometry': [self.ls]}, crs=self.crs)
        self.sg = sg
        self.cell2d = cell2d
        self.xcyc = xcyc
        self.xc, self.yc = list(zip(*self.xcyc))
        self.vertices = vertices
        self.xcenters, self.ycenters = xcenters, ycenters
        self.idomain = np.ones((self.ncpl))

        print(f'\nTRANSECT {name} length: {self.length}')
        print('x0 = ', x0, ' ,y0 = ', y0)
        print('x1 = ', x1, ' ,y1 = ', y1)
        print('angrot ', self.angrot)
        print('ncol = ', ncol)

        # Calculate distance along trasect for each cell
        start_x = self.xc[0]
        start_y = self.yc[0]
        delr = self.delr[0]

        self.L = [] # Create a list to store distances
        for cell in range(self.ncpl):
            distance = np.sqrt((start_x - self.xc[cell])**2 + (start_y - self.yc[cell])**2 + delr/2)
            self.L.append(distance)

    def make_vertical(self, top, lay_bottoms):
        top = top.reshape(1,-1)

        # Vertical discretisation
        self.nlay = len(lay_bottoms)
        botm = np.zeros((self.nlay, self.nrow, self.ncol))
        for lay in range(self.nlay):
            botm[lay, : ] = lay_bottoms[lay] * np.ones((self.ncpl))

        sg = flopy.discretization.StructuredGrid(delr=self.delr, delc=self.delc, 
                                                 top=top, botm=botm,
                                                 xoff = self.x0, yoff = self.y0, angrot = self.angrot)

        # Turn structured grid into DISV (its what I know!)
        xyzcenters = sg.xyzcellcenters
        xcenters = xyzcenters[0][0]
        ycenters = xyzcenters[1][0]
        iverts = sg.iverts
        verts = sg.verts

        cell2d = []
        xcyc = [] # added 
        for icpl in range(self.ncpl):
            xc = xcenters[icpl]
            yc = ycenters[icpl]
            iv1, iv2, iv3, iv4 = iverts[icpl]
            cell2d.append([icpl, xc, yc, 5, iv1, iv2, iv3, iv4, iv1])
            xcyc.append((xc, yc))
        
        vertices = []
        for v in range(len(verts)):
            i,j = verts[v]
            vertices.append([v, i, j]) # need to make 1 based

        self.coords = [[self.x0, self.y0], [self.x1, self.y1]]
        self.ls = LineString([[self.x0, self.y0], [self.x1, self.y1]])
        self.gdf = gpd.GeoDataFrame({'geometry': [self.ls]}, crs=self.crs)
        self.sg = sg
        self.cell2d = cell2d
        self.xcyc = xcyc
        self.xc, self.yc = list(zip(*self.xcyc))
        self.vertices = vertices
        self.xcenters, self.ycenters = xcenters, ycenters
        self.idomain = np.ones((self.ncpl))

        self.top = self.sg.top.flatten()
        self.botm = self.sg.botm.squeeze(axis=1)

        self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, 
                                                     cell2d=self.cell2d, 
                                                     ncpl = self.ncpl, 
                                                     top = self.top,
                                                     botm = self.botm,
                                                     #top=np.ones((self.nrow, self.ncol), dtype=float), 
                                                     #botm=np.zeros((1, self.nrow, self.ncol), dtype=float), 
                                                     nlay = self.nlay)
        self.gi = flopy.utils.GridIntersect(self.vgrid)

    def resample_to_top(self, tif_fname):
        # Resample topo to transect mesh
        topo = flopy.utils.Raster.load(tif_fname) #'../data/dem/goldcoast_aoi_cropped.tif')
        print(self.crs)
        print(topo.crs)
        self.topo = topo.resample_to_grid(self.vgrid, band=topo.bands[0], method="nearest", extrapolate_edges=True,)


    def map_cells(self, mesh3d):

        cells = mesh3d.gi.intersects(self.ls)["cellids"]
        
        self.ibd = np.zeros(mesh3d.ncpl)
        for cell in cells: 
            self.ibd[cell] = 1

        self.cells = cells
        self.cells_str = [str(cell) for cell in cells]