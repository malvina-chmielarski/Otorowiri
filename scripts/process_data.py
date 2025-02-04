import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape

class Data:
    def __init__(self):

            self.data_label = "DataBaseClass"

    def process_rch(self, mesh):
        #fname = '../data/data_climate/IDCJAC0001_007176_Data12.csv'
        #df = pd.read_csv(fname)
        #df = df[df.Year > 2018]
        #df = df[df.Year < 2024]
        #df = df.fillna(0)
        #df = df.reset_index()
        #mean_rch = df.Annual.mean()/1000/365 # DAILY RCH AVERAGE
        
        mean_rch = 35/365 # DAILY RCH AVERAGE
        self.rch_rec = {}      
        self.rch_rec[0] = [((0, icpl), mean_rch) for icpl in range(mesh.ncpl)] # Assume for now 35mm over entire year

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

    def process_ic(self, geomodel, mesh):
        self.strt = 500. #geomodel.top_geo - 1. # start with a watertable of 1m everywhere #wt.reshape(1,len(wt))
        
    def process_chd(self, geomodel, mesh):
        self.chd_rec = []
        for icpl in mesh.chd_cells:
            z = 0
            x,y,z = mesh.xcyc[icpl][0], mesh.xcyc[icpl][1], z
            point = Point(x,y,z)
            lay, icpl = geomodel.vgrid.intersect(x,y,z)
            cell_disv = icpl + lay*mesh.ncpl
            cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
            self.chd_rec.append([cell_disu, 0])    

    def process_ghb(self, geomodel, mesh):
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
        )