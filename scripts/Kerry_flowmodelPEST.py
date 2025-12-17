import pandas as pd
import flopy
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
import dill

fname_inputs         = '../inputs/inputs.pkl' # Base model inputs
fname_params_names   = '../inputs/pest_parameters.xlsx' # Parameter information
fname_params         = '../pest/parameters.par' # Parvals in list
fname_rawmodeloutput = '../modelfiles/A_observations.csv' # Raw flow model output
fname_modelled       = '../pest/model_observations.txt' # Manipulated model output
fname_measured       = '../inputs/measured.pkl' # Raw measured values (generated in main script)

# Import inputs and parameters, and prepare model
# Convert parameters to model inputs

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

    # Import parameter names
    print('Importing parameter names from ', fname_params_names)
    pars_df = pd.read_excel(fname_params_names, sheet_name = 'pars')
    par_names = pars_df.parnme.tolist()
 
    # Incorporate new parameters into model
    class Parameters:
        def __init__(self, names, values):
            for name, value in zip(names, values):
                setattr(self, name, value)
    pars = Parameters(par_names, par_values)
    print('pars = ', par_names, par_values)

    # Make HK and SS arrays
    hk_zones = [pars.HK_0, pars.HK_1, pars.HK_2, pars.HK_3, pars.HK_4]
    ss_zones = [pars.SS_0, pars.SS_1, pars.SS_2, pars.SS_3, pars.SS_4]
    hk = np.array(hk_zones)[inputs.zone_array]
    ss = np.array(ss_zones)[inputs.zone_array]
 
    # GHB 1 - River boundary (GHB1_COND_0)
    # W4 is at left side of transect (col = 0)
    # Left column (river) = River stage * conductance

    ghb1_spd= {}
    for sp in range(inputs.nper):
        ghb_rec = []
        stage = inputs.river_stage[sp]
        for lay in range(inputs.transect.nlay):
            if stage > inputs.transect.botm[lay,0]:
                col = 0 # Left column
                ref_head = stage
                cond = pars.GHB1_COND_0
                ghb_rec.append([(lay, col), ref_head, cond]) # Left column
        ghb1_spd[sp] = ghb_rec

    # GHB 2 - upgradient side
    # Reference head * GHB conductance (GHB_COND_0)
    ghb2_spd= {}
    for sp in range(inputs.nper):
        ghb_rec = []
        for lay in range(inputs.transect.nlay):
            if ref_head > inputs.transect.botm[lay,0]:
                col = inputs.transect.ncol -1 # Upgradient is right column
                ref_head = 0.6 # Minimum WL in W1 in all data collected for July was 0.64 m
                cond = pars.GHB2_COND_0
                ghb_rec.append([(lay, col), ref_head, cond]) # Left column
        ghb2_spd[sp] = ghb_rec

    # GHB 3 - Flooding boundary (GHB_3)
    # If surface water depth > 0.05m, then cell becomes surface water elevation * multiplier (RCH_MULT_0, etc)
    cond_zones = [pars.GHB3_COND_0, pars.GHB3_COND_1, pars.GHB3_COND_2, pars.GHB3_COND_3, pars.GHB3_COND_4]
    cond_array = np.array(cond_zones)[inputs.zone_array[0]] # top layer only
    ghb3_flag = np.zeros((inputs.nper, inputs.transect.ncpl))
    print(ghb3_flag.shape, inputs.nper, inputs.transect.ncpl)
    ghb3_spd = {}
    for sp in range(inputs.nper):
        ghb_rec = []
        count = 0
        for icpl in range(inputs.transect.ncpl):
            stage = inputs.floodplain_stage[sp][icpl]
            #stage = inputs.river_stage[sp]
            ground = inputs.transect.top[icpl]
            depth = stage - ground #inputs.floodplain_elevation
            if depth > 0.03:# If stage is at least 30 cm higher than ground level...and above transect GL...
                if icpl != 0 and icpl != inputs.transect.ncpl-1 and icpl > int(70/inputs.transect.delr[0]): # Columns at end are already boundaries... Won't apply GHB in first 50m!
                    ref_head = stage
                    cond = cond_array[icpl]
                    ghb_rec.append([(0, icpl), ref_head, cond])
                    ghb3_flag[sp, icpl] = 1
                    count += 1
        ghb3_spd[sp] = ghb_rec

    # Assign to inputs
    inputs.hk = hk
    inputs.ss = ss
    inputs.ghb1_spd = ghb1_spd # River boundary
    inputs.ghb2_spd = ghb2_spd # Upgradient boundary
    inputs.ghb3_spd = ghb3_spd # Flooding boundary
    inputs.ghb3_flag = ghb3_flag

    # Pickle inputs again with updated ones
    with open(fname_inputs, 'wb') as f:
        pickle.dump(inputs, f)
    return inputs

def write_and_run_model(inputs):
    print(f'Writing and running model: {inputs.modelname}')

    # Write MODFLOW 6 model
    
    sim = flopy.mf6.MFSimulation(sim_name='sim',
                                 exe_name=inputs.project.mfexe,
                                 version='mf6',
                                 sim_ws=inputs.project.workspace)

    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim,
                                                nper = len(inputs.perioddata),
                                                perioddata=inputs.perioddata)

    gwf = flopy.mf6.ModflowGwf(sim,
                               modelname=inputs.modelnames
                               save_flows=True)

    ims = flopy.mf6.ModflowIms(sim,
                               print_option='SUMMARY',
                               complexity='complex',
                               outer_dvclose=1.e-6,
                               inner_dvclose=1.e-6)  

    disv = flopy.mf6.ModflowGwfdisv(gwf, length_units='METERS',
                                    nlay = inputs.transect.nlay, ncpl=inputs.transect.ncpl, nvert=len(inputs.transect.vertices),
                                    top = inputs.transect.top, botm = inputs.transect.botm, vertices = inputs.transect.vertices, cell2d  =inputs.transect.cell2d)

    npf = flopy.mf6.ModflowGwfnpf(gwf,
                                  k = inputs.hk,
                                  k22 = inputs.hk,
                                  k33 = inputs.hk,
                                  icelltype = inputs.iconvert,
                                  save_flows=True,
                                  save_specific_discharge = True)

    ic = flopy.mf6.ModflowGwfic(gwf, strt = inputs.ic)#.reshape(1, -1)
   
    # GHB 1 = River boundary
    ghb1 = flopy.mf6.ModflowGwfghb(gwf, stress_period_data = inputs.ghb1_spd,
                                   save_flows=True, pname='GHB1',
                                   filename = f'{inputs.modelname}.ghb1') # River boundary

    # GHB 2 = Upgradient boundary
    ghb2 = flopy.mf6.ModflowGwfghb(gwf, stress_period_data = inputs.ghb2_spd,
                                   save_flows=True, pname='GHB2',
                                   filename = f'{inputs.modelname}.ghb2') # Upgradient boundary

    # GHB 3 = Flooding boundary
    ghb3 = flopy.mf6.ModflowGwfghb(gwf, stress_period_data = inputs.ghb3_spd,
                                   save_flows=True, pname='GHB3',
                                   filename = f'{inputs.modelname}.ghb3') # Upgradient boundary
    
    #rch = flopy.mf6.modflow.mfgwfrch.ModflowGwfrch(gwf, maxbound = len(rch_spd), stress_period_data = rch_spd,)

    sto = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(gwf, storagecoefficient=None,
                                                        iconvert=inputs.iconvert,
                                                        ss = inputs.ss,
                                                        sy = inputs.ss,
                                                        #steady_state={0: True},
                                                        #transient={1: True, 2: True},
                                                        )

    csv_file = inputs.modelname + "_observations.csv" # To write observation to

    obs = flopy.mf6.ModflowUtlobs(gwf,
                                  filename=inputs.modelname, print_input=True,
                                  continuous={csv_file: inputs.obs_rec},)

    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord='{}.cbc'.format(inputs.modelname),
                                head_filerecord='{}.hds'.format(inputs.modelname),
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                                printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')])

    sim.write_simulation(silent = False)
 
    success, buff = sim.run_simulation(silent = False)

    return sim, gwf

 

def interpolate_data(data, predefined_times):
    interpolated_data = []

    for bore_id in data['bore_id'].unique():
        bore_data = data[data['bore_id'] == bore_id].copy()
        bore_data = bore_data.sort_values('datetime')

        # Create a DataFrame with predefined times

        interp_df = pd.DataFrame({'datetime': predefined_times})

        # Merge with bore data and interpolate

        merged = pd.concat([bore_data[['datetime', 'obsval']], interp_df]).sort_values('datetime')
        merged = merged.drop_duplicates(subset=['datetime'])
        merged['obsval_interp'] = merged['obsval'].interpolate(method='linear')

        # Extract only the predefined times
        result = merged[merged['datetime'].isin(predefined_times)].copy()
        result['bore_id'] = bore_id

        # Add other columns from original data
        bore_info = bore_data.iloc[0]
        for col in ['x', 'y', 'z', 'icpl', 'Transect', 'Type']:
            if col in bore_data.columns:
                result[col] = bore_info[col]

        interpolated_data.append(result[['datetime', 'bore_id', 'obsval_interp']])    

    return pd.concat(interpolated_data, ignore_index=True)

def read_model_obs_and_interpolate(modelname):
    # MEASURED

    measured = pd.read_pickle(fname_measured)
    print('len(measured):', len(measured))

    # MODELLED
    modelled = pd.read_csv(fname_rawmodeloutput)
    modelled = modelled.melt(id_vars=['time'], var_name='bore_id', value_name='obsval')
    modelled['datetime'] = inputs.start + pd.to_timedelta(modelled['time'], unit='D') # Convert time column to datetime by adding time (in days) to start date
    modelled['bore_id'].unique()

    # INTERPOLATE MODELLED TO MEASURED TIMES

    interpolated_list = []

    for bore in measured.bore_id.unique():
        print(bore)
        filt_obs = measured[measured.bore_id == bore]
        filt_modelled = modelled[modelled.bore_id == bore]
        filt_modelled_obs = interpolate_data(filt_modelled, filt_obs.datetime)
        filt_modelled_obs = filt_modelled_obs.dropna(subset=['obsval_interp'])
        interpolated_list.append(filt_modelled_obs[['bore_id', 'datetime', 'obsval_interp']])

    # Combine all interpolated data
    interpolated_all = pd.concat(interpolated_list, ignore_index=True)

    merged = measured.merge(interpolated_all, on=['bore_id', 'datetime'], how='left')
    merged = merged.dropna(subset=['obsval_interp']) # This is where data exists but the model did not cover this time
    merged.rename(columns={'obsval': 'measured', 'obsval_interp': 'modelled'}, inplace=True)
    #merged['obsnme'] = merged['bore_id'].astype(str) + '_' + merged.index.astype(str)
    print(merged)

    # Pickle the obsnames in a dataframe
    with open('../modelfiles/obsnames.pkl', 'wb') as f:
        pickle.dump(merged['obsnme'].tolist(), f)

    # Write interpolated observations to text file
    modelled_values = merged.modelled.tolist()
    with open(fname_modelled, 'w+') as f:
        for i in range(len(modelled_values)):
            f.write(f'{str(modelled_values[i])}\n')

    print('\n----------------------------------------\n')    
    print(f'Measured values written as a list to {fname_measured}')
    print(f'Modelled values written as a list to {fname_modelled}\n')
    print(f'Number of measured/modelled observations = {len(modelled_values)}')
    print('----------------------------------------\n')

# Run the workflow
inputs = preprocessing(fname_inputs, fname_params, fname_params_names)
sim, gwf = write_and_run_model(inputs)
read_model_obs_and_interpolate(inputs.modelname)








 