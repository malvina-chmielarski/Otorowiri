import numpy as np
import pandas as pd
import flopy
import pickle
import dill
from datetime import datetime
import loopflopy.utils as utils

#unpickle (project, observations, mesh, geomodel) --> this will hold all the information from the base model

class geomodel:
    pass

class project:
    pass

class mesh:
    pass

class structuralmodel:
    pass

class observations:
    pass

#Import inputs
fname = '../modelfiles/otorowiri_geomodel.pkl'
print('Importing geomodel from ', fname)
with open(fname, 'rb') as f:
    geomodel = pickle.load(f)

fname = '../modelfiles/otorowiri_mesh.pkl'
print('Importing mesh from ', fname)
with open(fname, 'rb') as f:
    mesh = pickle.load(f)

fname = '../modelfiles/otorowiri_structuralmodel.pkl'
print('Importing structuralmodel from ', fname)
with open(fname, 'rb') as f:
    structuralmodel = dill.load(f)

fname = '../modelfiles/otorowiri_observations.pkl'
print('Importing observations from ', fname)
with open(fname, 'rb') as f:
    observations = pickle.load(f)
print(type(observations))
print(dir(observations))
print(observations.obs_rec)

fname = '../modelfiles/otorowiri_project.pkl'
print('Importing project from ', fname)
with open(fname, 'rb') as f:
    project = pickle.load(f)

fname_params_names   = '../pest/pest_parameters_otorowiri.xlsx' # Parameter information <--this is the master spreadsheet
fname_params         = '../pest/parameters.par' # Parvals in list
fname_rawmodeloutput = '../modelfiles/steadymodel_observations.csv' # Raw flow model output <--this is the observations csv that comes from running the model (initial run)
fname_modelled       = '../pest/model_observations.txt' # Manipulated model output
#fname_measured       = '../data/data_pest/measured_groundwater.xlsx' # Raw measured values (generated in main script)
#fname_measured       = ''

#---------------------------------------------------------------------------------------------------------------------------#

######### Bring in all the necessary spreadsheets and information ##########

#bringing in the PEST parameter sheet which can then be used to control all the unknowns
PEST_unknowns_df = pd.read_excel('../data/data_pest/pest_parameters_otorowiri.xlsx', sheet_name='pars')

# Recharge values in woody areas (identified in the veg_YEAR_cells file) should be <12 mm/yr
# Recharge in regular areas (i.e. on the surface but not in veg_YEAR_cells file) should be 20-50 mm/yr

precipitation_df = pd.read_excel('../data/data_precipitation/transient_rainfall_projection.xlsx')
steady_state_timestamp = "1969_Wet" #from process_filtering outcomes = "Using earliest timeframe with ≥10 bores: 1969_Wet"
annualised_rainfall = precipitation_df.loc[precipitation_df['Timestamp'] == steady_state_timestamp, 'Annualised_Rainfall_mm_per_yr'].values[0]

slope_factor_df = pd.read_csv('../data/data_precipitation/slope_factor.csv')
slope_factor = slope_factor_df['slope_Factor'].values

steady_state_vegetation_df = pd.read_csv('../data/data_specialcells/veg_1970_cells.csv') #<-- this is really like 1972
woody_cells = steady_state_vegetation_df['cell_id'].dropna().astype(int).values #this gives an array of cells that are woody vegetation
print(woody_cells) #Define the woody vegetation domain for 1972 (our proxy for 'steady state' since this is the earliest vegetation data we have)
print(len(woody_cells), 'woody cells')

drn_cell_df = pd.read_excel('../data/data_specialcells/drain_cells.xlsx')
drn_cells = drn_cell_df['cell_id'].dropna().astype(int).values
drn_lengths = drn_cell_df['drain_length'].dropna().astype(float).values

#---------------------------------------------------------------------------------------------------------------------------#

######### Define all the different packages ##########

######### Set the hydraulic parameters ##########

# Reset matplotlib to defaults
import matplotlib as mpl
mpl.rcdefaults()

# FILL CELL PROPERTIES
props = pd.read_excel(structuralmodel.geodata_fname, sheet_name = 'strat')
props = props.drop(index=[0]).reset_index()#inplace=True) # drop first row as it is ground
geomodel.hk_perlay = props.hk.tolist() #[100., 100., 100.]
geomodel.vk_perlay = props.vk.tolist() #[5., 5., 5.]
geomodel.ss_perlay = props.ss.tolist()
geomodel.sy_perlay = props.sy.tolist()
geomodel.iconvert_perlay = [0., 0., 0.] #props.iconvert.tolist()

geomodel.fill_cell_properties(mesh)
print(geomodel.hk_perlay)
print(geomodel.iconvert_perlay)  # Check the properties are filled correctly
props

######### Define the RCH package ##########

rch_data = []
rch_for_plot = np.zeros_like((geomodel.top_geo))

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
    rch_data.append(((cell_disu), rch))
    rch_for_plot[icpl] = rch
print("recharge matrix is:", rch_data)
print("recharge for woody cell 17 is:", rch_data[17])
print("recharge for non-woody cell 20 is:", rch_data[20])

######### Define the EVT package ##########

evt_data = []
average_evaporation_rate = 0.00312  # Average ET max (m/d) for steady state --> taken from 1950_Wet

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
        evt_data.append([cell_disu, surface, rate, depth])
print("evt is", evt_data)
print("evt for woody cell 17 is", evt_data[17])

######### Define the IC package ##########

strt = 215 #geomodel.top_geo - 1 # Initial water table 1m below ground surface #215
print("IC value is:", strt)

########## Define the DRN package ##########

drn_data = []

riv_depth = 2.0 # I think this means depth of drain? This is the river stage
leakance = 1.0 / (0.5 * riv_depth)  # kv / b --> the higher the leakance, the more water can flow through the drain
for icpl, length in zip(drn_cells, drn_lengths):
    model_lay = 0 # drain in top flow model layer
    cell_disv = icpl + model_lay*geomodel.ncpl # find the disv cell...
    cell_disu = utils.disvcell_to_disucell(geomodel, cell_disv) # convert to the disu cell...
    land_surface = geomodel.top_geo[icpl] # ground elevation at the cell
    drain_elevation = land_surface - riv_depth # bottom of drain elevation
    width = 10 # Assume a constant width of 10m for all drains
    conductance = leakance * length * width

    if cell_disu != -1: # if cell is not pinched out...
        drn_data.append((cell_disu, drain_elevation, conductance))
        print(f"Drain cell {icpl}: length={length}, conductance={conductance}")

#---------------------------------------------------------------------------------------------------------------------------#

######### Write the flow model ##########

'''class Flowmodel:
    
    def __init__(self, scenario, project, observations, mesh, geomodel, *args, **kwargs):     

        self.scenario = scenario
        self.project = project
        self.mesh = mesh
        self.geomodel = geomodel
        self.observations = observations
        self.lith = geomodel.lith
        self.logk11 = geomodel.logk11
        self.logk33 = geomodel.logk33
        
    def write_flowmodel(self, transient = False, xt3d = True, staggered = True, **kwargs):
        
        print('   Writing simulation and gwf for ', self.scenario, ' ...')
        print('xt3d = ', xt3d)

        self.xt3d = xt3d
        self.staggered = staggered
        self.newtonoptions = ['UNDER_RELAXATION']

        t0 = datetime.now()
       
        for key, value in kwargs.items():
            setattr(self, key, value)    
        print('mf6 executable expected: ', self.project.mfexe)
        # -------------- SIM -------------------------
        sim = flopy.mf6.MFSimulation(sim_name = 'sim', 
                                     version = 'mf6',
                                     exe_name = self.project.mfexe, 
                                     sim_ws = self.project.workspace)

        # -------------- TDIS -------------------------
      
        if not transient: 
            tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim)      
            
        if transient:  
            tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, nper=len(self.perioddata), perioddata=self.perioddata)
        
        # -------------- IMS -------------------------
        # Make linear solver (inner) an order of magnitude tighter than non-linear solver (outer)
        if not transient: 
            ims = flopy.mf6.ModflowIms(sim, print_option='SUMMARY', 
                                       complexity    = 'Moderate',
                                       outer_dvclose = 1e-2, 
                                       inner_dvclose = 1e-3, 
                                       outer_maximum = 400, 
                                       linear_acceleration = "BICGSTAB",
                                       preconditioner_levels=5, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                                       preconditioner_drop_tolerance=0.01, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                                       number_orthogonalizations=2,
                                      )
        if transient: 
            ims = flopy.mf6.ModflowIms(sim, print_option='SUMMARY', 
                                       complexity    = 'Moderate',
                                       outer_dvclose = 1e-2, 
                                       inner_dvclose = 1e-3, 
                                       outer_maximum = 60, 
                                       linear_acceleration = "BICGSTAB",
                                       preconditioner_levels=5, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                                       preconditioner_drop_tolerance=0.01, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                                       number_orthogonalizations=2, # NORTH - increase if hard!
                                       )

        # -------------- GWF -------------------------
        gwf = flopy.mf6.ModflowGwf(sim, 
                                   modelname=self.scenario, 
                                   save_flows=True, 
                                   newtonoptions = self.newtonoptions,) 

        # -------------- DIS -------------------------       

        from loopflopy import disv2disu
        #from importlib import reload
        #reload(disv2disu)
        Disv2Disu = disv2disu.Disv2Disu           
         
        dv2d = Disv2Disu(self.mesh.vertices, self.mesh.cell2d, self.geomodel.top_geo, self.geomodel.botm, 
                         staggered=self.staggered, disv_idomain = self.geomodel.idomain,)
        disu_gridprops = dv2d.get_gridprops_disu6()
        self.disu_gridprops = disu_gridprops.copy()  # Save for later use
        disu = flopy.mf6.ModflowGwfdisu(gwf, **disu_gridprops) # This is the flow package
        # -------------- NPF -------------------------

        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf, 
                                                       xt3doptions = self.xt3d, 
                                                       k = self.geomodel.k11, 
                                                       k22  = self.geomodel.k22, 
                                                       k33  = self.geomodel.k33, 
                                                       angle1 = self.geomodel.angle1, 
                                                       angle2 = self.geomodel.angle2,
                                                       angle3 = self.geomodel.angle3, 
                                                       #angle1 = 0., angle2 = 0., angle3 = 0.,
                                                       icelltype = self.geomodel.iconvert, #.astype(int), #had to change to int for mf6
                                                       save_flows = True, 
                                                       save_specific_discharge = True,)
                                                       #dev_minimum_saturated_thickness = 1)# try 0.1 then 0.001... no more than 1m!
        
        # -------------- IC -------------------------
        ic = flopy.mf6.ModflowGwfic(gwf, strt = strt)
              
        # -------------- RCH-------------------------
        if self.rch:
            rch = flopy.mf6.modflow.mfgwfrch.ModflowGwfrch(gwf, 
                                                           maxbound = len(rch_data),
                                                           stress_period_data = rch_data,)
            #print("RCH created with", len(self.data.rch_rec), "boundaries.")
            
        # -------------- EVT-------------------------
        if self.evt:

            evt = flopy.mf6.ModflowGwfevt(gwf,
                                            maxbound = len(evt_data),
                                            stress_period_data = evt_data)      
        # -------------- DRN -------------------------
        if self.drn:
            drn = flopy.mf6.ModflowGwfdrn(gwf, 
                                          maxbound = len(drn_data),
                                          stress_period_data = drn_data,
                                          save_flows= True,
                                          print_input = True,
                                          print_flows = True,
                                          boundnames= True,
                                          )
        
        # -------------- OBS -------------------------
        if self.obs: 
            csv_file = self.scenario + "_observations.csv" # To write observation to
            obs = flopy.mf6.ModflowUtlobs(gwf, 
                                          filename=self.scenario, 
                                          print_input=True, 
                                          continuous={csv_file: self.observations.obs_rec},) 
        # ------------ OC ---------------------------------
        oc = flopy.mf6.ModflowGwfoc(gwf, 
                                    budget_filerecord='{}.bud'.format(self.scenario), 
                                    head_filerecord='{}.hds'.format(self.scenario),
                                    saverecord=[('HEAD', 'LAST'),('BUDGET', 'ALL')], 
                                    printrecord = [("BUDGET", "LAST")] #printrecord=None,) #this is what Kerry had
                                    )
        
        # -------------- WRITE SIMULATION -------------------------
            
        sim.write_simulation(silent = True)   
        run_time = datetime.now() - t0
        print('   Time taken to write flow model = ', run_time.total_seconds())

         # --------------------------------------------------------
        return(sim)

    def run_flowmodel(self, sim, transient = False):
    
        t0 = datetime.now()
        print('Running simulation for ', self.scenario, ' ...')

        success, buff = sim.run_simulation(silent = True)   
        print('Model success = ', success)
        run_time = datetime.now() - t0
        print('   run_time = ', run_time.total_seconds())
        
        def process_results():
                        
            gwf = sim.get_model(self.scenario)
            package_list = gwf.get_package_list()
            print(package_list)
            times = gwf.output.head().get_times()
            head = gwf.output.head().get_data()[-1] # last time
            print('head results shape ', head.shape)
            
            bud = gwf.output.budget()
            spd = bud.get_data(text='DATA-SPDIS')[0]
            if self.chd:
                chdflow = bud.get_data(text='CHD')[-1]
                self.chdflow = chdflow
            obs_data = gwf.obs

            self.gwf = gwf
            self.head = head
            self.obsdata = obs_data
            self.spd = spd
            
            self.runtime = run_time.total_seconds()
            
        if success:
            process_results()

        else:
            print('   Re-writing IMS - Take 2')
            sim.remove_package(package_name='ims')
            
            ims = flopy.mf6.ModflowIms(sim, print_option='ALL', 
                                   complexity    = 'Complex',
                                   outer_dvclose = 1e-4, 
                                   inner_dvclose = 1e-6, 
                                   outer_maximum = 60,
                                   inner_maximum = 60,
                                   linear_acceleration = "BICGSTAB",
                                   backtracking_number = 10,
                                   backtracking_tolerance = 100, #1.01 (aggressive) to 10000
                                   backtracking_reduction_factor = 0.5, # 0.1-0.3, or 0.9 when non-linear convergence HARD 
                                   #preconditioner_levels=10, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                                   #preconditioner_drop_tolerance=0.01, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                                   #number_orthogonalizations=10,
                                   ) # NORTH - increase if hard!)

            sim.ims.write()
            success2, buff = sim.run_simulation(silent = True)   
            print('Model success2 = ', success2)
            
            if success2:
                process_results()
     
            
            else:
                print('   Re-writing IMS - Take 3')
                
                if transient:   # Increase number of timesteps to help convergence
                    future_years = 5
                    nts_future = future_years * 12
                    tdis_future = [(future_years * 365, nts_future, 1.2)] # period length, number of timesteps, tsmult
                    sim.remove_package(package_name='tdis')
                    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, nper=len(tdis_future), perioddata=tdis_future)
                
                # More aggressive solver settings
                sim.remove_package(package_name='ims')
                ims = flopy.mf6.ModflowIms(sim, print_option='ALL', 
                            complexity    = 'Complex',
                            outer_dvclose = 1e-4, 
                            inner_dvclose = 1e-6, 
                            outer_maximum = 60,
                            inner_maximum = 300,
                            linear_acceleration = "BICGSTAB",
                            reordering_method=['RCM'],
                            #no_ptcrecord = ['ALL'],
                            under_relaxation = 'DBD',
                            under_relaxation_kappa = 0.2, #0.05 (aggressive) to 0.3
                            under_relaxation_theta = 0.7, # 0.5 - 0.9
                            under_relaxation_gamma = 0.1, # 0-0.2 doesnt make big difference
                            under_relaxation_momentum = 0.001, #0-0.001 doesn't make big difference
                            backtracking_number = 15,
                            backtracking_tolerance = 1.1, #1.01 (aggressive) to 10000
                            backtracking_reduction_factor = 0.7, # 0.1-0.3, or 0.9 when non-linear convergence HARD 
                            preconditioner_levels=18, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                            preconditioner_drop_tolerance=0.00001, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                            number_orthogonalizations=10,)
                sim.ims.write()
                success3, buff = sim.run_simulation(silent = True)   
                print('Model success3 = ', success3)
                
                if success3:
                    process_results()
    
scenario = 'steadymodel'
#fm is object; Flowmodel is class
fm = Flowmodel(scenario, project, observations, mesh, geomodel)
#utils.print_object_details(fm)

# Write and run flow model files
sim = fm.write_flowmodel(chd = False, #not necessary when all model boundaries have implied boundary conditions
                         wel = False, 
                         obs = True, #put false to not record heads at the chosen cells
                         rch = True, 
                         evt = True, 
                         drn = True, 
                         ghb = False, 
                         xt3d = True,
                         staggered = False, # True made "fully connected". "False" is essentially DISV.
                        )

fm.run_flowmodel(sim)'''