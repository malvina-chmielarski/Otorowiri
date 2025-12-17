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
import os

#bringing in the PEST parameter sheet which can then be used to control all the unknowns
PEST_unknowns = pd.read_excel('../data/data_pest/pest_parameters_otorowiri.xlsx', sheet_name='pars')

class Data:
    def __init__(self):

            self.data_label = "DataBaseClass"

    def process_rch(self, geomodel, mode, mesh, precipitation_df, steady_veg_json, veg_json_folder):
        # Recharge values in woody areas (identified in the veg_YEAR_cells file) should be <12 mm/yr
        # Recharge in regular areas (i.e. on the surface but not in veg_YEAR_cells file) should be 20-50 mm/yr

        #find the 'centroids' 

        rec = []
        rec_for_plot = np.zeros_like((geomodel.top_geo))

        if mode == 'converge':
            for icpl in range(geomodel.ncpl): #this is ALL the cells in the top layer of the model
                #geomodel.idomain[(0,0)] = 0
                lay = 0
                cell_disv = icpl + lay*geomodel.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                if cell_disu == -1: # if cell is not pinched out...
                    continue # skip pinched out cells
                rch = 0.0003  # 0.035 --> 35mm/yr, 0.0000001 allows for convergence #0.0003 is what the figures are made with 
                rec.append(((cell_disu), rch))
            print('Recharge is', rec)
            #print('you are definitely changing parameters')
            self.rch_rec = {}
            self.rch_rec[0] = rec
            self.rec_for_plot = rec_for_plot

        if mode == 'steady':

            steady_state_timestamp = "1969_Wet" #from process_filtering outcomes = "Using earliest timeframe with ≥10 bores: 1969_Wet"
            annualised_rainfall = precipitation_df.loc[precipitation_df['Timestamp'] == steady_state_timestamp, 'Annualised_Rainfall_mm_per_yr'].values[0]

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
            print("slope factor is:", slope_factor)
            print("slopes are:", slopes)

            for icpl in range(geomodel.ncpl): #this is ALL the cells in the top layer of the model
                lay = 0
                cell_disv = icpl + lay*geomodel.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                if cell_disu == -1: # if cell is not pinched out...
                    continue # skip pinched out cells
                #rch = 0.000001  # 0.035 --> 35mm/yr, 0.0000001 allows for convergence
                if icpl in woody_cells:

                    cell_precip = (0.01 * annualised_rainfall)/(1000*365)  # should stay around 12mm/yr in woody areas so 0.012
                else:
                    cell_precip = (0.06 * annualised_rainfall)/(1000*365) # about three times the woody area recharge so 0.036
                #rch = cell_precip
                rch = cell_precip * slope_factor[icpl] # Apply slope factor to the recharge
                rec.append(((cell_disu), rch))
                rec_for_plot[icpl] = rch
            print("recharge matrix is:", rec)
            print("recharge for woody cell 17 is:", rec[17])
            print("recharge for non-woody cell 20 is:", rec[20])
            self.rch_rec = {}      
            self.rch_rec[0] = rec
            self.rec_for_plot = rec_for_plot

        elif mode == 'transient':

            # --- rainfall / periods ---
            rainfall_df = pd.read_excel('../data/data_precipitation/transient_rainfall_projection.xlsx') # rainfall_df is expected to have columns: Timestamp, Annualised_Rainfall_mm_per_yr, Period_Year

            # --- geometry / slope factor (static, compute once) ---
            centroids = np.array([mesh.xc, mesh.yc]).T              # shape (ncpl, 2)
            cell_elevation = geomodel.top_geo                       # shape (ncpl,)
            ncpl = geomodel.ncpl

            # build KDTree and neighbor list
            tree = KDTree(centroids)
            distances, _ = tree.query(centroids, k=2)
            avg_spacing = np.mean(distances[:, 1])
            neighbors = [tree.query_ball_point(c, r=avg_spacing * 1.5) for c in centroids]

            # compute local slope for each cell (mean absolute slope to neighbors)
            slopes = np.zeros(ncpl, dtype=float)
            for i, elev_i in enumerate(cell_elevation):
                neigh_ids = neighbors[i]
                if len(neigh_ids) <= 1:
                    slopes[i] = 0.0
                    continue
                slope_sum = 0.0
                count = 0
                for j in neigh_ids:
                    if j == i:
                        continue
                    elev_j = cell_elevation[j]
                    dx = np.linalg.norm(centroids[i] - centroids[j])
                    if dx > 0:
                        slope_sum += abs(elev_i - elev_j) / dx
                        count += 1
                slopes[i] = slope_sum / count if count > 0 else 0.0

            # normalize slopes to slope_factor in range [0.5, 1.0] using your formula
            max_slope = slopes.max() if slopes.size and slopes.max() > 0 else 1.0
            slope_factor = 1.0 - (slopes / max_slope) * 0.5   # array same length as ncpl

            # --- vegetation coefficients (static mapping) ---
            veg_coeff = {"woody": 0.01, "nonwoody": 0.06}

            # Precompute mapping icpl -> cell_disu and active mask
            # Your original code used: cell_disv = icpl + lay * geomodel.ncpl  and then cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
            # If model is single layer (lay = 0) the mapping is simply:
            # cell_disu = geomodel.cellid_disu.flatten()[icpl]
            # but preserve the general approach to skip inactive cells.
            ncpl = geomodel.ncpl
            cellid_disu_flat = geomodel.cellid_disu.flatten()
            icpl_to_cell_disu = np.full(ncpl, -1, dtype=int)
            active_mask = np.zeros(ncpl, dtype=bool)
            for icpl in range(ncpl):
                # if you had multiple layers, adjust lay accordingly; here we assume top layer (lay=0)
                cell_disv = icpl   # lay*ncpl + icpl  (lay==0)
                try:
                    cell_disu = int(cellid_disu_flat[cell_disv])
                except Exception:
                    cell_disu = -1
                icpl_to_cell_disu[icpl] = cell_disu
                active_mask[icpl] = (cell_disu != -1)

            # --- prepare veg file list once and sort by year available ---
            veg_json_folder = veg_json_folder  # ensure this variable exists in your scope
            available_files = sorted([
                f for f in os.listdir(veg_json_folder)
                if f.startswith("veg_") and f.endswith("_cells.json")
            ])
            # parse years in filenames like 'veg_1950_cells.json'
            available_years = []
            year_to_file = {}
            for fname in available_files:
                parts = fname.split('_')
                # attempt to parse year from second token
                if len(parts) >= 2:
                    try:
                        y = int(parts[1])
                        available_years.append(y)
                        year_to_file[y] = fname
                    except Exception:
                        continue
            available_years = sorted(available_years)

            # --- Build transient recharge dictionary expected by FloPy: {iper: [(cellid, rch), ...], ...} ---
            self.rch_rec = {}

            # iterate through rainfall_df rows in order (should be 150 rows)
            for iper, row in enumerate(rainfall_df.itertuples()):
                timestamp = getattr(row, 'Timestamp', None)
                rainfall = getattr(row, 'Annualised_Rainfall_mm_per_yr', None)
                year = int(getattr(row, 'Period_Year', np.nan))

                # find veg file for this year; if missing choose last available earlier year
                woody_cells = np.array([], dtype=int)
                if year in year_to_file:
                    veg_path = os.path.join(veg_json_folder, year_to_file[year])
                    with open(veg_path, 'r') as fh:
                        woody_cells = np.array(json.load(fh), dtype=int)
                else:
                    # find last available year <= year
                    earlier_years = [y for y in available_years if y <= year]
                    if earlier_years:
                        chosen_year = max(earlier_years)
                        veg_path = os.path.join(veg_json_folder, year_to_file[chosen_year])
                        with open(veg_path, 'r') as fh:
                            woody_cells = np.array(json.load(fh), dtype=int)
                        print(f"No veg file for {year}, using {chosen_year} ({veg_path})")
                    else:
                        # no veg data available - assume all non-woody
                        woody_cells = np.array([], dtype=int)
                        print(f"No vegetation data available before {year}; assuming all non-woody for period {iper} ({timestamp})")

                # Build a boolean mask for woody cells (icpl indices)
                woody_mask = np.zeros(ncpl, dtype=bool)
                if woody_cells.size:
                    # if woody_cells are already icpl indices this is fine; if they are cell_disu, adapt accordingly
                    # we assume the JSON lists icpl indices (0..ncpl-1). If not, you'll need to map cell_disu back to icpl.
                    woody_mask[woody_cells] = True

                # Build recharge list for this stress period
                rec = []
                for icpl in range(ncpl):
                    if not active_mask[icpl]:
                        continue  # skip inactive cells
                    cell_disu = icpl_to_cell_disu[icpl]
                    if cell_disu == -1:
                        continue

                    # choose veg coefficient
                    if woody_mask[icpl]:
                        coeff = veg_coeff["woody"]
                    else:
                        coeff = veg_coeff["nonwoody"]

                    # convert rainfall (mm/yr) -> cell recharge. You used /3000.0 earlier; preserve that behavior
                    # final units depend on your model convention — adjust if needed.
                    cell_precip = (coeff * float(rainfall)) / (1000*365)

                    # apply slope modifier (array)
                    rch_val = cell_precip * float(slope_factor[icpl])

                    # append tuple (disu_index, recharge_value)
                    rec.append((int(cell_disu), float(rch_val)))

                # store for this stress period
                self.rch_rec[iper] = rec
                print(f"\nPeriod {iper}: {timestamp}, Year {year}")
                print(f"Number of active cells: {len(rec)}")
                print("Sample of recharge values (first 10 cells):")
                print(rec[:10])  # shows first 10 tuples (cell_disu, rch)
                #print(f"Period {iper}: {timestamp}, Year {year}, woody_cells = {woody_cells.size}, active_cells = {len(rec)}")
            
            print("rch_rec min key:", min(self.rch_rec.keys()))
            print("rch_rec max key:", max(self.rch_rec.keys()))
            #print("TDIS nper:", len(flowmodel.perioddata))

            print(f"Total transient periods generated: {len(self.rch_rec)}")
    
    def process_evt(self, geomodel, mode, steady_veg_json):
       
        #  fixed_cell (boolean) indicates that evapotranspiration will not be
        #      reassigned to a cell underlying the cell specified in the list if the
        #      specified cell is inactive.

        evt = []

        if mode == 'converge':
            evt_cells = np.arange(geomodel.ncpl) # Assume evapotranspiration occurs in top layer of model (and no pinched out cells)
        
            depth = 2    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
            rate = 9e-3  # ET max (m/d)
    
            for cell in evt_cells:
                disucell = utils.disvcell_to_disucell(geomodel, cell) # zerobased
                surface = geomodel.top_geo[cell] # ground elevation at the cell
                evt.append([disucell, surface, rate, depth])
            self.evt_rec = {}      
            self.evt_rec[0] = evt
        
        if mode == 'steady':

            average_evaporation_rate = 0.00312  # Average ET max (m/d) for steady state --> taken from 1950_Wet

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
                    depth = 5    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                    rate = average_evaporation_rate * 0.6  # ET max (m/d)
                else:
                    surface = geomodel.top_geo[icpl] # ground elevation at the cell
                    depth = 2    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                    rate = average_evaporation_rate * 1.3 # ET max (m/d)
                if cell_disu != -1: # if cell is not pinched out...
                    evt.append([cell_disu, surface, rate, depth])
            print("evt is", evt)
            print("evt for woody cell 17 is", evt[17])
            self.evt_rec = {}
            self.evt_rec[0] = evt
        
        if mode == 'transient':

            #get the average evaporation rate from the evaporation dataframe
            evaporation_df = pd.read_excel('../data/data_evaporation/seasonal_average_evaporation.xlsx')
            for index, row in evaporation_df.iterrows():
                if 'Evaporation_m_per_day' in row:
                    average_evaporation_rate = row['Evaporation_m_per_day']  # Average ET max (m/d) for transient periods

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
                    depth = 5    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                    rate = average_evaporation_rate * 0.6  # ET max (m/d)
                else:
                    surface = geomodel.top_geo[icpl] # ground elevation at the cell
                    depth = 2    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                    rate = average_evaporation_rate * 1.3  # ET max (m/d)
                if cell_disu != -1: # if cell is not pinched out...
                    evt.append([cell_disu, surface, rate, depth])
            print("evt is", evt)
            print("evt for woody cell 17 is", evt[17])
            self.evt_rec = {}
            self.evt_rec[0] = evt

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

        self.strt = 215 #geomodel.top_geo - 1 # Initial water table 1m below ground surface #215
        print("IC value is:", self.strt)

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

    def make_drain_rec(self, geomodel, setting, drn_cellids, drn_lengths, surface_confinement_json):

        # not sure what the next few lines are about, but copied from here: https://flopy.readthedocs.io/en/latest/Notebooks/mf6_parallel_model_splitting_example.html
        # I'm guessing that dv0 is depth of drain, and the "leakance" is based on head difference between middle and bottom of drain
        if setting == 'unconfined':
            self.drn_rec = []
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
            # --- River drains ---
            riv_depth = 2.0
            leakance = 1.0 / (0.5 * riv_depth)
            river_drains = {}

            for icpl, length in zip(drn_cellids, drn_lengths):
                model_lay = 0  # top layer
                cell_disv = icpl + model_lay * geomodel.ncpl
                cell_disu = utils.disvcell_to_disucell(geomodel, cell_disv)

                if cell_disu != -1:
                    land_surface = geomodel.top_geo[icpl]
                    drain_elevation = land_surface - riv_depth
                    width = 10
                    conductance = leakance * length * width
                    river_drains[cell_disu] = (drain_elevation, conductance)

            # --- Surface confinement drains ---
            with open(surface_confinement_json, 'r') as f:
                confinement_cells = json.load(f)
            confinement_cells = np.array(confinement_cells, dtype=int)
            confinement_drains = {}
            print(confinement_cells)

            for cell_disu in confinement_cells:
                if cell_disu >= geomodel.ncpl:
                    continue  # ignore invalid indices
                land_surface = geomodel.top_geo[cell_disu]
                drain_elevation = land_surface
                conductance = 0.0001
                confinement_drains[cell_disu] = (drain_elevation, conductance)

            # --- Merge river and confinement drains ---
            ###add the river and confinement lists together and then get rid of repeats with rule
            '''merged_drains = river_drains.copy()
            for cell, (conf_elev, conf_cond) in confinement_drains.items():
                if cell in merged_drains:
                    river_elev, _ = merged_drains[cell]
                    merged_drains[cell] = (river_elev, conf_cond)  # river elevation, confinement conductance
                else:
                    merged_drains[cell] = (conf_elev, conf_cond)'''
            
            merged_drains = river_drains.copy()
            for cell, (conf_elev, conf_cond) in confinement_drains.items():
                if cell in merged_drains:
                    river_elev, old_cond = merged_drains[cell]
                    merged_drains[cell] = (river_elev, conf_cond)
                    print(f"Cell {cell} in both river & confinement: keep elevation {river_elev}, override conductance {old_cond} -> {conf_cond}")
                else:
                    merged_drains[cell] = (conf_elev, conf_cond)
                    print(f"Cell {cell} only in confinement: add with elevation {conf_elev}, conductance {conf_cond}")

            self.drn_rec = []
            drn_for_plot = np.zeros(geomodel.ncpl)

            for cell, (elev, cond) in merged_drains.items():
                print(f"Drain cell {cell}: elevation={elev}, conductance={cond}")
                self.drn_rec.append((cell, elev, cond))
                drn_for_plot[cell] = 1 #max(drn_for_plot[cell], cond)

            self.drn_for_plot = drn_for_plot

            print(f"Total river drains: {len(river_drains)}")
            print(f"Total confinement drains: {len(confinement_drains)}")
            print(f"Total merged drains: {len(merged_drains)}")
            print("Number of nonzero cells in drn_for_plot:", np.count_nonzero(drn_for_plot))
            print("Unique conductance values (up to 20):", np.unique(drn_for_plot[drn_for_plot > 0])[:20])