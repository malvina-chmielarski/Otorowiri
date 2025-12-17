import numpy as np
import pandas as pd
import os
import flopy
import pickle
import dill
from datetime import datetime
import loopflopy.utils as utils

fname_inputs         = '../data/data_pest/inputs.pkl' # Base model inputs
fname_params_names   = '../data//data_pest/pest_parameters_otorowiri.xlsx' # Parameter information <--this is the master spreadsheet
fname_params         = '../pest/parameters.par' # Parvals in list
fname_rawmodeloutput = '../modelfiles/transientmodel_observations.csv' # Raw flow model output <--this is the observations csv that comes from running the model (initial run)
fname_measured       = '../data/data_pest/measured_groundwater.xlsx' # Raw measured values (generated in main script)

class Inputs:
    pass

def preprocessing(fname_inputs, fname_params, fname_params_names):

    # Import inputs
    print('Importing inputs from ', fname_inputs)
    with open(fname_inputs, 'rb') as f:
        inputs = dill.load(f)

    # Import parameters values
    print('Importing parameter values from ', fname_params)
    with open(fname_params, 'r') as f:
        lines = f.readlines()
    par_values = []
    for line in lines:  
        par_values.append(float(line.strip()))
    print(par_values)

    # Import parameter names
    print('Importing parameter names from ', fname_params_names)
    pars_df = pd.read_excel(fname_params_names, sheet_name = 'pars')
    par_names = pars_df.parnme.tolist()
    print(par_names)
    
    # Incorporate new parameters into model
    class Parameters:
        def __init__(self, names, values):
            for name, value in zip(names, values):
                setattr(self, name, value)
    pars = Parameters(par_names, par_values)
    print('pars = ', par_names, par_values)

    #---------------------------------------------------------------------------------------------------------------------------#

    ######### Bring in all the necessary spreadsheets and information ##########

    # Recharge values in woody areas (identified in the veg_YEAR_cells file) should be <12 mm/yr
    # Recharge in regular areas (i.e. on the surface but not in veg_YEAR_cells file) should be 20-50 mm/yr

    precipitation_df = pd.read_excel('../data/data_precipitation/transient_rainfall_projection.xlsx')
    evaporation_df = pd.read_excel('../data/data_evaporation/seasonal_average_evaporation.xlsx')

    slope_factor_df = pd.read_csv('../data/data_precipitation/slope_factor.csv')
    slope_factor = slope_factor_df['slope_Factor'].values

    veg_folder = '../data/data_specialcells/'  # ensure this variable exists in your scope
    available_files = sorted([
        f for f in os.listdir(veg_folder)
        if f.startswith("veg_") and f.endswith("_cells.csv")
        ])
    available_veg_years = []
    year_to_file = {}
    for fname in available_files:
        parts = fname.split('_')
        # attempt to parse year from second token
        if len(parts) >= 2:
            try:
                y = int(parts[1])
                available_veg_years.append(y)
                year_to_file[y] = fname
            except Exception:
                continue
    available_veg_years = sorted(available_veg_years)

    def get_woody_cells(year):
            if year in year_to_file:
                use_year = year
            else:
                earlier_years = [y for y in available_veg_years if y <= year]
                if len(earlier_years) == 0:
                    raise ValueError(f"No vegetation data available for year {year} or earlier.")
                use_year = max(earlier_years)
            veg_path = os.path.join(veg_folder, year_to_file[use_year])
            df = pd.read_csv(veg_path)
            if "cell_id" not in df.columns:
                raise KeyError(f"'cell_id' column missing in {veg_path}")
            woody_cells = df["cell_id"].dropna().astype(int).values
            return woody_cells

    drn_cell_df = pd.read_excel('../data/data_specialcells/drain_cells.xlsx')
    drn_cells = drn_cell_df['cell_id'].dropna().astype(int).values
    drn_lengths = drn_cell_df['drain_length'].dropna().astype(float).values

    #---------------------------------------------------------------------------------------------------------------------------#

    ######### Define all the different packages ##########

    ######### Set the hydraulic parameters ##########

    ncpl = inputs.geomodel.ncpl #<--this is the total number of cells we are dealing with in the top layer
    total_cells = inputs.geomodel.ncpl * inputs.geomodel.nlay

    k11_data = np.full(total_cells, pars.k_kp, dtype = float)
    k22_data = np.full(total_cells, pars.k_kp, dtype = float)
    k33_data = np.full(total_cells, pars.vk_kp, dtype = float)
    ss_data = np.full(total_cells, pars.ss, dtype = float)
    sy_data = np.full(total_cells, pars.sy, dtype = float)
    iconvert_data = np.full(total_cells, inputs.iconvert, dtype = float)

    ######### Define the RCH package ##########

    rch_data = {}

    for iper, row in enumerate(precipitation_df.itertuples()):
        timestamp = row.Timestamp
        precipitation = row.Annualised_Rainfall_mm_per_yr
        year = int(row.Period_Year)

        # vegetation mask
        woody_cells = get_woody_cells(year)
        woody_mask = np.zeros(ncpl, dtype=bool)
        if woody_cells.size > 0:
            woody_mask[woody_cells] = True

        rec = []
        for icpl in range(ncpl):
            # veg coefficient
            if woody_mask[icpl]:
                coeff = pars.rch_woody_coeff
            else:
                coeff = pars.rch_nonwoody_coeff

            cell_precip = (coeff * float(precipitation)) / (1000 * 365) # rainfall → m/day
            rch_val = cell_precip * slope_factor[icpl] # apply slope factor
            rec.append((icpl, rch_val)) # DISV uses icell2d index directly

        rch_data[iper] = rec
        print(f"Period {iper}, year {year}: {len(rec)} recharge entries")
        print(f"\nPeriod {iper}: {timestamp}, Year {year}")
        print(f"Number of active cells: {len(rec)}")
        print("Sample of recharge values (first 10 cells):")
        print(rec[:10])  # shows first 10 tuples (cell_disu, rch)
        #print(f"Period {iper}: {timestamp}, Year {year}, woody_cells = {woody_cells.size}, active_cells = {len(rec)}")
            
    print("rch_rec min key:", min(rch_data.keys()))
    print("rch_rec max key:", max(rch_data.keys()))
    #print("TDIS nper:", len(flowmodel.perioddata))

    print(f"Total transient periods generated: {len(rch_data)}")

    ######### Define the EVT package ##########

    evt_data = {}

    #get the average evaporation rate from the evaporation dataframe
    for iper, row in enumerate(evaporation_df.itertuples()):
        timestamp = row.Class
        average_evaporation_rate = row.Evaporation_m_per_day # Average ET max (m/d) for transient periods
        year = int(row.Period_Year)
        
        # vegetation mask
        woody_cells = get_woody_cells(year)
        woody_mask = np.zeros(ncpl, dtype=bool)
        if woody_cells.size > 0:
            woody_mask[woody_cells] = True
        
        evt = []
        for icpl in range(ncpl): #this is ALL the cells in the top layer of the model
            surface = inputs.geomodel.top_geo[icpl] # ground elevation at the cell
            if icpl in woody_cells:
                depth = pars.evt_woody_depth    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                rate = average_evaporation_rate * pars.evt_woody_multiplier  # ET max (m/d)
            else:
                depth = pars.evt_nonwoody_depth    # extinction depth (m) --> this needs to be smaller for evapotranspiration to occur sooner (i.e more evap power)
                rate = average_evaporation_rate * pars.evt_nonwoody_multiplier  # ET max (m/d)
            evt.append([icpl, surface, rate, depth])

        evt_data[iper] = evt     
        print("evt is", evt)
        print("evt for woody cell 17 is", evt[17])

    ######### Define the IC package ##########

    strt = 215 #geomodel.top_geo - 1 # Initial water table 1m below ground surface #215
    print("IC value is:", strt)

    ########## Define the DRN package ##########

    drn_data = []

    for icpl, length in zip(drn_cells, drn_lengths):
        land_surface = inputs.geomodel.top_geo[icpl] # ground elevation at the cell
        drain_elevation = land_surface - pars.river_depth # bottom of drain elevation
        conductance = length * pars.river_width

        drn_data.append((icpl, drain_elevation, conductance))
        print(f"Drain cell {icpl}: length={length}, conductance={conductance}")

    #---------------------------------------------------------------------------------------------------------------------------#

    #pickle everything back up
    inputs.k11_data = k11_data
    inputs.k22_data = k22_data
    inputs.k33_data = k33_data
    inputs.ss_data = ss_data
    inputs.sy_data = sy_data
    inputs.iconvert_data = iconvert_data

    inputs.rch_data = rch_data 
    inputs.evt_data = evt_data
    inputs.strt = strt
    inputs.drn_data = drn_data

    # Pickle inputs again with updated ones
    with open(fname_inputs, 'wb') as f:
        pickle.dump(inputs, f)

    return inputs

#---------------------------------------------------------------------------------------------------------------------------#

######### Write the flow model ##########

def write_and_run_model(inputs):
    print(f'Writing and running model: {inputs.modelname}')
    
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
       
        tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, nper=len(self.perioddata), perioddata=self.perioddata)
        
        # -------------- IMS -------------------------
        # Make linear solver (inner) an order of magnitude tighter than non-linear solver (outer)
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
                                                       k = k11_data, 
                                                       k22  = k22_data, 
                                                       k33  = k33_data, 
                                                       icelltype = iconvert_data, #.astype(int), #had to change to int for mf6
                                                       save_flows = True, 
                                                       save_specific_discharge = True,)
                                                       #dev_minimum_saturated_thickness = 1)# try 0.1 then 0.001... no more than 1m!
        
        # -------------- IC -------------------------
        ic = flopy.mf6.ModflowGwfic(gwf, strt = strt)

        # -------------- STO -------------------------

        sto = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(gwf, 
                                                        storagecoefficient=None, 
                                                        iconvert=self.geomodel.iconvert, 
                                                        ss = self.geomodel.ss, 
                                                        sy = self.geomodel.sy)
        
        # -------------- WEL ------------------------- 
        if self.wel:                                              
            wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf, 
                                                            print_input=True, 
                                                            print_flows=True, 
                                                            stress_period_data = self.data.spd_wel, 
                                                            save_flows=True,) 
              
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

fm.run_flowmodel(sim)