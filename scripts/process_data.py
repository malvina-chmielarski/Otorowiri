import pandas as pd
import numpy as np
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import loopflopy.utils as utils
import pickle
import geopandas as gpd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import flopy

class Data:
    def __init__(self):

            self.data_label = "DataBaseClass"

    def process_rch(self, geomodel):
   
        rec = []

        for icpl in range(geomodel.ncpl):
            lay = 0
            cell_disv = icpl + lay*geomodel.ncpl
            cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
            rch = 0.000000001  # 0.035 --> 35mm/yr
            if cell_disu != -1: # if cell is not pinched out...
                rec.append(((0, icpl), rch)) # Assume for now 35mm over entire year

        self.rch_rec = {}      
        self.rch_rec[0] = rec 

    def process_evt(self, geomodel):
       
        #  fixed_cell (boolean) indicates that evapotranspiration will not be
        #      reassigned to a cell underlying the cell specified in the list if the
        #      specified cell is inactive.

        evt_cells = np.arange(geomodel.ncpl) # Assume evapotranspiration occurs in top layer of model (and no pinched out cells)
       
        depth = 10    # extinction depth (m)
        rate = 1e-6  # ET max (m/d)
 
        evt = [] # Create a list to store the evapotranspiration records
        for cell in evt_cells:
            disucell = utils.disvcell_to_disucell(geomodel, cell) # zerobased
            surface = geomodel.top_geo[cell] # ground elevation at the cell
            evt.append([disucell, surface, rate, depth])
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
        self.strt = 200 #geomodel.top_geo - 1 # Initial water table 1m below ground surface 

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

        linestrings = [ls1, ls2, ls3]
        labels = ['ls1', 'ls2', 'ls3']
        def plot_linestrings(lines, labels):
            fig, ax = plt.subplots() 
            for line, label in zip(lines, labels):
                x, y = line.xy
                ax.plot(x, y, '-o', ms = 2, label = label)  # You can set color, linestyle, etc.

            ax.set_aspect('equal')
            ax.legend()
            plt.show()
        plot_linestrings(linestrings, labels)
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

        self.drain_cells = drn_cellids
        return ibd, drn_cellids, drn_lengths


    def make_drain_rec(self, geomodel, drn_cellids, drn_lengths):
        
        # not sure what the next few lines are about, but copied from here: https://flopy.readthedocs.io/en/latest/Notebooks/mf6_parallel_model_splitting_example.html
        # I'm guessing that dv0 is depth of drain, and the "leakance" is based on head difference between middle and bottom of drain
        riv_depth = 10. # I think this means depth of drain? This is the river stage
        leakance = 100.0 / (0.5 * riv_depth)  # kv / b
        self.drn_rec = []
        for icpl, length in zip(drn_cellids, drn_lengths):
            model_lay = 0 # drain in top flow model layer
            cell_disv = icpl + model_lay*geomodel.ncpl # find the disv cell...
            cell_disu = utils.disvcell_to_disucell(geomodel, cell_disv) # convert to the disu cell..
            land_surface = geomodel.top_geo[icpl] # ground elevation at the cell
            drain_elevation = land_surface - riv_depth # bottom of drain elevation
            width = 10 # Assume a constant width of 10m for all drains
            conductance = leakance * length * width

            if cell_disu != -1: # if cell is not pinched out...
                self.drn_rec.append((cell_disu, drain_elevation, conductance))

'''
        stageleft = 10.0
        stageright = 10.0
        bound_sp1 = []
        for il in range(nlay):
            condleft = hk * (stageleft - zbot) * delc
            condright = hk * (stageright - zbot) * delc
            for ir in range(nrow):
                bound_sp1.append([il, ir, 0, stageleft, condleft])
                bound_sp1.append([il, ir, ncol - 1, stageright, condright])
        print("Adding ", len(bound_sp1), "GHBs for stress period 1.")
        
        
        # General-Head Boundaries
        ghb_period = {}
        ghb_period_array = []
        for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
            for row in range(0, 15):
                ghb_period_array.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
        ghb_period[0] = ghb_period_array
        ghb = flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_period,)
        ts_recarray = []
        fd = open(os.path.join(data_pth, "tides.txt"))
        for line in fd:
            line_list = line.strip().split(",")
            ts_recarray.append((float(line_list[0]), float(line_list[1])))
        ghb.ts.initialize(
            filename="tides.ts",
            timeseries=ts_recarray,
            time_series_namerecord="tides",
            interpolation_methodrecord="linear",
        )
        obs_recarray = {
            "ghb_obs.csv": [
                ("ghb-2-6-10", "GHB", (1, 5, 9)),
                ("ghb-3-6-10", "GHB", (2, 5, 9)),
            ],
            "ghb_flows.csv": [
                ("Estuary2", "GHB", "Estuary-L2"),
                ("Estuary3", "GHB", "Estuary-L3"),
            ],
        }
        ghb.obs.initialize(
            filename=f"{model_name}.ghb.obs",
            print_input=True,
            continuous=obs_recarray,
        )'''