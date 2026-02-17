import pandas as pd
import numpy as np
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import loopflopy.utils as utils
import pickle
from scipy.interpolate import griddata

class Inputs:
    def __init__(self):

            self.data_label = "DataBaseClass"


    def process_rch(self, geomodel):

        rec = []
        for icpl in range(geomodel.mesh.ncpl):
            lay = 0
            cell_disv = icpl + lay*geomodel.mesh.ncpl
            cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
            rch = 0.001 
            rec.append(((0, icpl), rch)) 

        self.rch_rec = {}      
        self.rch_rec[0] = rec 


    def process_ic(self):
        self.strt = 199. # start with a watertable of 199m everywhere #wt.reshape(1,len(wt))


    def process_ghb(self, geomodel, props):
        # Create array where non -1 values in cellid_disu are replaced with xcellcenters
        # First, create a copy of cellid_disu
        x_array = geomodel.cellid_disu.copy().astype(float)
        x_array = np.where(geomodel.cellid_disu != -1, geomodel.vgrid.xcellcenters[geomodel.cellid_disu], np.nan)
        # Find indices where cellid_disu is not -1
        valid_indices = np.where(geomodel.cellid_disu != -1, )
        print('valid indices = ', valid_indices[1])

        # Get the corresponding cell indices for xcellcenters
        # Since cellid_disu contains cell IDs, we need to map them to column indices
        cell_cols = geomodel.cellid_disu[valid_indices] % geomodel.ncpl  # Get column index from cell ID

        # Replace non -1 values with corresponding xcellcenters
        x_array[valid_indices] = geomodel.vgrid.xcellcenters[cell_cols]

        # Set -1 values to NaN for clarity (optional)
        x_array[geomodel.cellid_disu == -1] = np.nan

        # Find cells with minimum x coordinate
        lays, icpls = np.where(x_array == min(x_array[valid_indices]))
        cell_tuples = list(zip(lays, icpls)) # Zip them up into tuples
        print(f"Number of cells with minimum x coordinate: {len(cell_tuples)}")
        print(f"Cell tuples (lay, icpl): {cell_tuples}")

        ghb_cells = []
        for lay, icpl in cell_tuples:
            print(geomodel.cellid_disu[lay, icpl])
            ghb_cells.append(geomodel.cellid_disu[lay, icpl])

        self.ghb_rec = []

        # WEST BOUNDARY
        boundary = "west"

        ref_head = 200.#float(geomodel.top_geo[0, 3] - 1) # 1m below ground surface on west
        print(icpl)

        for lay, icpl in cell_tuples:
            
            if ref_head > geomodel.botm[lay, icpl]: # if head is not below cell bottom...
                cell_disv = icpl + lay*geomodel.mesh.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                if cell_disu != -1: # if cell is not pinched out...
                    lith = geomodel.lith_disv[lay, icpl]
                    if lith != -1: # Don't do for cells in the air!
                        
                        K = props.hk[props.lithid == lith].values[0] # hydraulic conductivity for this lithology
                        A = 50*geomodel.mesh.delc[0] # WEST BOUNDARY
                        L = 10. # Length of boundary in m, for conductance calculation
                        conductance = K*A/L 
                        print(f'GHB cell {cell_disu} at layer {lay}, icpl {icpl}, head {ref_head}, K {K}, conductance {conductance}')
                        self.ghb_rec.append([cell_disu, ref_head, 1000000000]) # node, stage, conductance 

    def process_chd(self, geomodel, props):
        # Create array where non -1 values in cellid_disu are replaced with xcellcenters
        # First, create a copy of cellid_disu
        x_array = geomodel.cellid_disu.copy().astype(float)

        # Find indices where cellid_disu is not -1
        valid_indices = np.where(geomodel.cellid_disu != -1)

        # Get the corresponding cell indices for xcellcenters
        # Since cellid_disu contains cell IDs, we need to map them to column indices
        cell_cols = geomodel.cellid_disu[valid_indices] % geomodel.ncpl  # Get column index from cell ID

        # Replace non -1 values with corresponding xcellcenters
        x_array[valid_indices] = geomodel.vgrid.xcellcenters[cell_cols]

        # Set -1 values to NaN for clarity (optional)
        x_array[geomodel.cellid_disu == -1] = np.nan

        # Find cells with maximum x coordinate
        lays, icpls = np.where(x_array == max(x_array[valid_indices]))
        cell_tuples = list(zip(lays, icpls)) # Zip them up into tuples
        print(f"Number of cells with maximum x coordinate: {len(cell_tuples)}")
        print(f"Cell tuples (lay, icpl): {cell_tuples}")

        chd_cells = []
        for lay, icpl in cell_tuples:
            print(geomodel.cellid_disu[lay, icpl])
            chd_cells.append(geomodel.cellid_disu[lay, icpl])

        self.chd_rec = []

        # WEST BOUNDARY
        boundary = "west"

        ref_head = 300.#float(geomodel.top_geo[0, 3] - 1) # 1m below ground surface on west
        print(icpl)

        for lay, icpl in cell_tuples[0:1]:
            
            if ref_head > geomodel.botm[lay, icpl]: # if head is not below cell bottom...
                cell_disv = icpl + lay*geomodel.mesh.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                if cell_disu != -1: # if cell is not pinched out...
                    lith = geomodel.lith_disv[lay, icpl]
                    if lith != -1: # Don't do for cells in the air!
                        print(f'CHD cell {cell_disu} at layer {lay}, icpl {icpl}, head {ref_head}')
                        self.chd_rec.append([cell_disu, ref_head]) # node, stage, conductance 