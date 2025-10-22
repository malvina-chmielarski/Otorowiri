import pandas as pd
import numpy as np
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import loopflopy.utils as utils
import pickle
import geopandas as gpd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import flopy
import json
from scipy.spatial import KDTree

class Data:
    def __init__(self):

            self.data_label = "DataBaseClass"

    def process_rch(self, geomodel, mode, mesh, precipitation_df, steady_veg_json):
        # Recharge values in woody areas (identified in the veg_YEAR_cells file) should be <12 mm/yr
        # Recharge in regular areas (i.e. on the surface but not in veg_YEAR_cells file) should be 20-50 mm/yr

        rec = []

        if mode == 'converge':
            for icpl in range(geomodel.ncpl): #this is ALL the cells in the top layer of the model
                #geomodel.idomain[(0,0)] = 0
                lay = 0
                cell_disv = icpl + lay*geomodel.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                if cell_disu == -1: # if cell is not pinched out...
                    continue # skip pinched out cells
                rch = 0.0003  # 0.035 --> 35mm/yr, 0.0000001 allows for convergence
                rec.append(((cell_disu), rch))
            print('Recharge is', rec)
            #print('you are definitely changing parameters')
            self.rch_rec = {}
            self.rch_rec[0] = rec

        if mode == 'steady':

            # Define the period precipitation for the steady state timestamp
            steady_state_timestamp = "1969_Wet" #from process_filtering outcomes = "Using earliest timeframe with ≥10 bores: 1969_Wet"
            total_rainfall = precipitation_df.loc[
                precipitation_df['Class'] == steady_state_timestamp, 
                'Total Rainfall (mm)'
                ].sum()
            # the total rainfall needs to be converted from mm in a variable numbers of months to mm/yr
            num_months = precipitation_df.loc[
                precipitation_df['Class'] == steady_state_timestamp, 
                'Date'
                ].nunique()
            print(num_months, 'months of data for', steady_state_timestamp)
            annualised_rainfall = (total_rainfall / num_months) * 12  # Convert to mm/yr as the rate for cells
            print(f"Annualised rainfall for {steady_state_timestamp}: {annualised_rainfall} mm/yr")

            #Define the woody vegetation domain for 1972 (our proxy for 'steady state' since this is the earliest vegetation data we have)
            with open(steady_veg_json, 'r') as f:
                woody_cells = json.load(f)
            woody_cells = np.array(woody_cells, dtype = int) #this gives an array of cells that are woody vegetation
            print(woody_cells)
            print(len(woody_cells), 'woody cells')

            # vegetation coefficients
            #veg_coeff = {"woody": 0.05, "non-woody": 0.15} # recharge coefficients for woody and non-woody areas

            #create slope for each cell
            # x and y are stored in mesh as xc and yc
            centroids = np.array([mesh.xc, mesh.yc]).T  # shape (ncpl, 2)
            cell_elevation = geomodel.top_geo  # shape (ncpl,)
            tree = KDTree(centroids)
            distances, _ = tree.query(centroids, k=2)
            avg_spacing = np.mean(distances[:, 1])  # average cell spacing
            neighbors = [tree.query_ball_point(c, r=avg_spacing * 1.5) for c in centroids] # Get neighbors within 1.5× spacing
            slopes = np.zeros_like(cell_elevation)
            for i, elev_i in enumerate(cell_elevation):
                neigh_ids = neighbors[i]
                if len(neigh_ids) <= 1:
                    continue
                slope_sum = 0
                count = 0
                for j in neigh_ids:
                    if j == i:
                        continue
                    elev_j = cell_elevation[j]
                    dx = np.linalg.norm(centroids[i] - centroids[j])
                    if dx > 0:
                        slope_sum += abs(elev_i - elev_j) / dx
                        count += 1
                slopes[i] = slope_sum / count if count > 0 else 0
            slope_factor = 1 - (slopes / slopes.max()) * 0.5 # Normalize slopes to range [0.5, 1.0] as a factor

            for icpl in range(geomodel.ncpl): #this is ALL the cells in the top layer of the model
                lay = 0
                cell_disv = icpl + lay*geomodel.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                if cell_disu == -1: # if cell is not pinched out...
                    continue # skip pinched out cells
                #rch = 0.000001  # 0.035 --> 35mm/yr, 0.0000001 allows for convergence
                if icpl in woody_cells:
                    cell_precip = (0.05 * annualised_rainfall)/3000  # should stay around 12mm/yr in woody areas so 0.012
                else:
                    cell_precip = (0.15 * annualised_rainfall)/3000 # about three times the woody area recharge so 0.036
                #rch = cell_precip
                rch = cell_precip * slope_factor[icpl] # Apply slope factor to the recharge
                rec.append(((cell_disu), rch))
            print("recharge matrix is:", rec)
            print("recharge for woody cell 17 is:", rec[17])
            print("recharge for non-woody cell 20 is:", rec[20])
            self.rch_rec = {}      
            self.rch_rec[0] = rec

        elif mode == 'transient':
            # Define the period precipitation for the transient timestamp

            self.rch_rec = {}      
            self.rch_rec[0] = rec

    def process_evt(self, geomodel, mode, steady_veg_json):
       
        #  fixed_cell (boolean) indicates that evapotranspiration will not be
        #      reassigned to a cell underlying the cell specified in the list if the
        #      specified cell is inactive.

        evt = []

        if mode == 'converge':
            evt_cells = np.arange(geomodel.ncpl) # Assume evapotranspiration occurs in top layer of model (and no pinched out cells)
        
            depth = 2    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
            rate = 5e-3  # ET max (m/d)
    
            for cell in evt_cells:
                disucell = utils.disvcell_to_disucell(geomodel, cell) # zerobased
                surface = geomodel.top_geo[cell] # ground elevation at the cell
                evt.append([disucell, surface, rate, depth])
            self.evt_rec = {}      
            self.evt_rec[0] = evt
        
        if mode == 'steady':

            #Define the woody vegetation domain for 1972 (our proxy for 'steady state' since this is the earliest vegetation data we have)
            with open(steady_veg_json, 'r') as f:
                woody_cells = json.load(f)
            woody_cells = np.array(woody_cells, dtype = int) #this gives an array of cells that are woody vegetation
            print(woody_cells)
            print(len(woody_cells), 'woody cells')

            for icpl in range(geomodel.ncpl): #this is ALL the cells in the top layer of the model
                lay = 0
                cell_disv = icpl + lay*geomodel.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                #print ('cell_disu', cell_disu)
                if icpl in woody_cells:
                    surface = geomodel.top_geo[icpl] # ground elevation at the cell
                    depth = 0.5    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                    rate = 1e-3  # ET max (m/d)
                else:
                    surface = geomodel.top_geo[icpl] # ground elevation at the cell
                    depth = 2    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                    rate = 1e-4  # ET max (m/d)
                if cell_disu != -1: # if cell is not pinched out...
                    evt.append([cell_disu, surface, rate, depth])
            print("evt is", evt)
            print("evt for woody cell 17 is", evt[17])
            self.evt_rec = {}
            self.evt_rec[0] = evt
        
        #if mode == 'transient':

            #clearing years

            #post-clearing, pre-pumping years

            #post-pumping years

    def process_wel(self, geomodel, mesh, spatial, wel_q, wel_qlay):
                  # geo layer pumping from
        
        ## Assume screening pumping well across entire geological layer, ## Find top and bottom of screen 
        self.wel_screens = []
        self.spd_wel = []
        for n in range(spatial.npump):
            icpl = mesh.wel_cells[n]
            print(icpl)
            if wel_qlay == 0:
                wel_top = geomodel.top[wel_cell]  
            else:   
                wel_top = geomodel.botm[(wel_qlay[n])* geomodel.nls-1, icpl]
            wel_bot = geomodel.botm[(wel_qlay[n] + 1) * geomodel.nls-1, icpl]   
            self.wel_screens.append((wel_top, wel_bot))
                   
            if geomodel.vertgrid == 'vox':
                nwell_cells = int((wel_top - wel_bot)/geomodel.dz)
                for lay in range(int((geomodel.top_geo[icpl]-wel_top)/geomodel.dz), int((geomodel.top_geo[icpl]-wel_top)/geomodel.dz) + nwell_cells):   
                    cell_disv = icpl + lay*mesh.ncpl
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    self.spd_wel.append([cell_disu, wel_q[n]/nwell_cells])
    
            if geomodel.vertgrid == 'con':        
                nwell_cells = geomodel.nls # For this research, assume pumping across entire geological layer
                for wel_lay in range(wel_qlay[n] * geomodel.nls, (wel_qlay[n] + 1) * geomodel.nls): # P.geo_pl = geological pumped layer                    
                    cell_disv = icpl + wel_lay*mesh.ncpl
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    self.spd_wel.append([cell_disu, wel_q[n]/nwell_cells])
                    
            if geomodel.vertgrid == 'con2':       
                lay = 0
                well_layers = []
                nwell_cells = 0
                
                while geomodel.botm[lay, icpl] >= wel_top-0.1: # above top of screen
                    lay += 1
                while geomodel.botm[lay, icpl] > wel_bot: # above bottom of screen
                    if geomodel.idomain[lay, icpl] != -1: # skips pinched out cells
                        nwell_cells += 1
                        well_layers.append(lay)
                    lay += 1
                
                for lay in well_layers:
                    cell_disv = icpl + lay*mesh.ncpl
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    self.spd_wel.append([cell_disu, wel_q[n]/nwell_cells])      
    
        print(self.wel_screens)

    def process_ic(self, geomodel):
        '''top_cells = np.arange(geomodel.ncpl)
        starting_WL = []
        for cell in top_cells:
            disucell = utils.disvcell_to_disucell(geomodel, cell) # zerobased
            initial_WL = geomodel.top_geo[cell] - 10 # ground elevation at the cell
            starting_WL.append([disucell, initial_WL])
        self.strt = {}      
        self.strt[0] = starting_WL'''

        self.strt = 210 #geomodel.top_geo - 1 # Initial water table 1m below ground surface #215

    def process_chd(self, geomodel, mesh):
   
        xcoords = np.zeros((mesh.ncpl,))  # Initialize an array to hold x-coordinates of drain cells
        for cell in self.drain_cells:
            xcoords[cell] = mesh.xcyc[cell][0]  # appends the x-coordinate of each drain cell
        west_x = np.min(xcoords[xcoords != 0])  # Find the minimum x-coordinate of drain cells
        print('West x-coordinate of drain cells:', west_x)
        chd_cell = np.where(xcoords == west_x)[0][0]  # Find the cell index of the westernmost drain cell
        print('Westernmost drain cell index:', chd_cell)

        chd_ibd = np.zeros(mesh.ncpl, dtype=int)
        chd_ibd[chd_cell] = 1  # Set the westernmost drain cell to 1 in the chd_ibd array

        head = geomodel.top_geo[chd_cell] - 5 #assumes head 5m below land surface
        print('Head in CHD cell :', head)
        chd_rec = []
        chd_rec.append([chd_cell, head])
        
        self.chd_rec = chd_rec
        self.chd_ibd = chd_ibd
         
    def process_ghb(self, geomodel, mesh, props): # Coast line
        self.ghb_rec = []
        for icpl in mesh.ghb_west_cells: # for each chd_cell in plan...
            for geo_lay in range(geomodel.nlg): # for each geological layer....
                if not np.isnan(props.ghb_west_stage[geo_lay]):               # if a constant head value exists....
                    for model_lay in geomodel.model_layers[geo_lay]: # then for each flow model layer...
                        cell_disv = icpl + model_lay*mesh.ncpl # find the disv cell...
                        cell_disu = utils.disvcell_to_disucell(geomodel, cell_disv) # convert to the disu cell..
                        if cell_disu != -1:
                            self.ghb_rec.append([cell_disu, props.ghb_west_stage[geo_lay], 
                                                 props.ghb_west_cond[geo_lay]]) # node, stage, conductance

    def process_drn_linestrings(self, spatial):

        gdf = gpd.read_file('../data/data_shp/Model_Streams.shp')
        gdf.to_crs(epsg=28350, inplace=True)
        gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)

        ##Arrowsmith River polygons
        Arrowsmith_gdf = gdf[((gdf['something'] == 'Arrowsmith_1') | (gdf['something'] == 'Arrowsmith_2') | (gdf['something'] == 'Arrowsmith_3'))]
        ls1 = Arrowsmith_gdf.iloc[0].geometry
        ls2 = Arrowsmith_gdf.iloc[1].geometry
        ls3 = Arrowsmith_gdf.iloc[2].geometry

        '''##Small Creek polygons
        Small_creek_gdf = gdf[gdf['something'] == 'Another_Creek']
        ls4 = Small_creek_gdf.iloc[0].geometry

        ##Sand Plain polygons
        Sand_plain_gdf = gdf[gdf['something'] == 'Sand_Plain_Creek_1']
        ls5 = Sand_plain_gdf.iloc[0].geometry

        ##Fault trace polygons
        gdf = gpd.read_file('../data/data_shp/southern_fault.shp')
        gdf.to_crs(epsg=28350, inplace=True)
        gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)
        
        southern_fault_gdf = gdf.iloc[[0]]
        ls6 = southern_fault_gdf.iloc[0].geometry'''

        linestrings = [ls1, ls2, ls3] #, ls4, ls5, ls6]
        labels = ['ls1', 'ls2', 'ls3'] #, 'ls4', 'ls5', 'ls6']
        def plot_linestrings(lines, labels):
            fig, ax = plt.subplots() 
            for line, label in zip(lines, labels):
                x, y = line.xy
                ax.plot(x, y, '-o', ms = 2, label = label)  # You can set color, linestyle, etc.

            ax.set_aspect('equal')
            ax.legend()
            plt.show()
        plot_linestrings(linestrings, labels)

        print("the lengths of the linestrings are:", sum([line.length for line in linestrings]))
        return linestrings

    def get_drain_cells(self, linestrings, geomodel):
        ixs = flopy.utils.GridIntersect(geomodel.vgrid, method="vertex")
        cellids = []
        for seg in linestrings:
            v = ixs.intersect(seg, sort_by_cellid=True)
            cellids += v["cellids"].tolist()
        intersection_rg = np.zeros(geomodel.vgrid.shape[1:])
        for loc in cellids:
            intersection_rg[loc] = 1

        # intersect stream segs to simulate as drains
        ixs = flopy.utils.GridIntersect(geomodel.vgrid, method="vertex")
        drn_cellids = []
        drn_lengths = []
        i = 0
        for seg in linestrings:
            v = ixs.intersect(LineString(seg), sort_by_cellid=True)
            drn_cellids += v["cellids"].tolist()
            drn_lengths += v["lengths"].tolist()
            i+=1
        print('Number of drain cells = ', len(drn_cellids))

        ibd = np. zeros((geomodel.ncpl), dtype=int)
        for i, cellid in enumerate(drn_cellids):
            ibd[cellid] = 1

        print('the drain lengths are ', drn_lengths)
        print('the sum of the drain lengths is ', sum(drn_lengths))

        self.drain_cells = drn_cellids
        return ibd, drn_cellids, drn_lengths

    def make_drain_rec(self, geomodel, setting, drn_cellids, drn_lengths):
        self.drn_rec = []

        # not sure what the next few lines are about, but copied from here: https://flopy.readthedocs.io/en/latest/Notebooks/mf6_parallel_model_splitting_example.html
        # I'm guessing that dv0 is depth of drain, and the "leakance" is based on head difference between middle and bottom of drain
        if setting == 'unconfined':
            riv_depth = 2.0 # I think this means depth of drain? This is the river stage
            leakance = 1.0 / (0.5 * riv_depth)  # kv / b --> the higher the leakance, the more water can flow through the drain
            for icpl, length in zip(drn_cellids, drn_lengths):
                model_lay = 0 # drain in top flow model layer
                cell_disv = icpl + model_lay*geomodel.ncpl # find the disv cell...
                cell_disu = utils.disvcell_to_disucell(geomodel, cell_disv) # convert to the disu cell...
                land_surface = geomodel.top_geo[icpl] # ground elevation at the cell
                drain_elevation = land_surface - riv_depth # bottom of drain elevation
                width = 10 # Assume a constant width of 10m for all drains
                conductance = leakance * length * width

                if cell_disu != -1: # if cell is not pinched out...
                    self.drn_rec.append((cell_disu, drain_elevation, conductance))
                
                print(f"Drain cell {icpl}: length={length}, conductance={conductance}")
        
        elif setting == 'surficial confinement':
            ###here there will be two kinds of drain cells - the ones that correspond to the surface confinement, and the ones that are still part of the surface drainage
            with open(steady_veg_json, 'r') as f:
                woody_cells = json.load(f)
            woody_cells = np.array(woody_cells, dtype = int) #this gives an array of cells that are woody vegetation
            print(woody_cells)
            print(len(woody_cells), 'woody cells')
            
            depth_of_surficial_confinement = 10.0
            #also no leakance, just conductance
            for icpl in zip(drn_cellids, drn_lengths): #here bring in the cell ids of the surficial confinement layer
                model_lay = 0
                cell_disv = icpl + model_lay*geomodel.ncpl
                cell_disu = utils.disvcell_to_disucell(geomodel, cell_disv)
                land_surface = geomodel.top_geo[icpl]
                drain_elevation = land_surface #- depth_of_surficial_confinement # keep drain elevation as the top of the geomodel surface
                #width not included in the surface confinement layer???
                conductance = 100.0 #/ (0.5 * #depth_of_surficial_confinement) --> number itself doesn't matter (just needs to control the pressure of system)
                if cell_disu != -1: # if cell is not pinched out...
                    self.drn_rec.append((cell_disu, drain_elevation, conductance))

                    # if a cell is both river and confinement, then use river depth but confinement conductance