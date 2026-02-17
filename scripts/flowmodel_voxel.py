import numpy as np
from datetime import datetime
import flopy
import matplotlib.pyplot as plt

class Flowmodel:

    
    def __init__(self, scenario, project, inputs, mesh, geomodel, observations = None, *args, **kwargs):     

        self.scenario = scenario
        self.project = project
        self.inputs = inputs
        self.mesh = mesh
        self.geomodel = geomodel
        self.observations = observations

    def write_flowmodel(self, transient = False, 
                        xt3d = True, 
                        staggered = True, 
                        chd = False, 
                        rch = False,
                        obs = False, 
                        wel = False, 
                        ghb = False, 
                        evt = False, 
                        zbud = False,
                        newtonoptions = ['UNDER_RELAXATION'], 
                        **kwargs):


        # Set all method parameters as instance attributes
        params = locals().copy()  # Get all local variables (method parameters)
        params.pop('self')        # Remove 'self' from the dictionary
        params.pop('kwargs')      # Remove 'kwargs' from the dictionary
        
        for key, value in params.items():
            setattr(self, key, value)
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        t0 = datetime.now()
        print('Writing simulation and gwf for ', self.scenario, ' ...')
        print('    xt3d = ', xt3d)
        print('    staggered = ', self.staggered)
        print('    transient = ', transient)
        print('    mf6 executable expected: ', self.project.mfexe)
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

        disv = flopy.mf6.ModflowGwfdisv(gwf,
                            nlay=self.geomodel.nlay, 
                            ncpl=self.geomodel.mesh.ncpl, 
                            nvert=len(self.geomodel.mesh.vertices),
                            top=self.geomodel.top, 
                            botm=self.geomodel.botm, 
                            cell2d=self.geomodel.mesh.cell2d,
                            vertices=self.geomodel.mesh.vertices,
                            idomain = self.geomodel.idomain)
        
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
                                                       icelltype = self.geomodel.iconvert,
                                                       save_flows = True, 
                                                       save_specific_discharge = True,)
                                                       #dev_minimum_saturated_thickness = 1)# try 0.1 then 0.001... no more than 1m!
        
        # -------------- IC -------------------------
        ic = flopy.mf6.ModflowGwfic(gwf, strt = self.inputs.strt)

         # --------------  STO -------------------------
    
        if not transient:
            sto = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(gwf, 
                                                    storagecoefficient=None, 
                                                    iconvert=self.geomodel.iconvert, 
                                                    ss = self.geomodel.ss, 
                                                    sy = self.geomodel.sy,
                                                    steady_state={0: True})
        # -------------- WEL / STO -------------------------
        if transient and self.wel:  

            sto = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(gwf, 
                                                           storagecoefficient=None, 
                                                           iconvert=self.geomodel.iconvert, 
                                                           ss = self.geomodel.ss, 
                                                           sy = self.geomodel.sy,
                                                           #steady_state={0: True}
                                                           )
        if self.wel:
            wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf, 
                                                            print_input=True, 
                                                            print_flows=True, 
                                                            stress_period_data = self.inputs.spd_wel, 
                                                            save_flows=True,) 
                
        # -------------- GHB-------------------------
        if self.ghb:
            
            ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(gwf, 
                                                           maxbound = len(self.inputs.ghb_rec),
                                                           stress_period_data = self.inputs.ghb_rec,)
        # -------------- CHD-------------------------
        if self.chd:
            
            chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(gwf, 
                                                           maxbound = len(self.inputs.chd_rec),
                                                           stress_period_data = self.inputs.chd_rec,)
        
        # -------------- RCH-------------------------
        if self.rch:
            rch = flopy.mf6.modflow.mfgwfrch.ModflowGwfrch(gwf, 
                                                           maxbound = len(self.inputs.rch_rec),
                                                           stress_period_data = self.inputs.rch_rec,)          

        # -------------- EVT-------------------------
        if self.evt:

            evt = flopy.mf6.modflow.mfgwfevt.ModflowGwfevt(gwf,
                                  print_input=True,
                                  print_flows=True,
                                  save_flows=True,
                                  fixed_cell=True, #If True, ET does not shift to underlying active cell if the specified one is inactive
                                  pname='EVT',
                                  stress_period_data= self.inputs.evt_rec)
        
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
                                    printrecord=None,)
        
        # -------------- WRITE SIMULATION -------------------------
            
        sim.write_simulation(silent = True)   
        run_time = datetime.now() - t0
        print('Time taken to write flow model = ', run_time.total_seconds())

         # --------------------------------------------------------
        return(sim, gwf)

    def run_flowmodel(self, sim, transient = False):
        """
        Execute MODFLOW 6 simulation with automatic solver fallback strategies.
        
        Runs the MODFLOW 6 simulation with progressive solver adjustments if
        convergence fails. Implements a three-tier strategy from moderate to
        very aggressive solver settings to handle difficult convergence cases.
        
        Parameters
        ----------
        sim : flopy.mf6.MFSimulation
            MODFLOW 6 simulation object to run.
        transient : bool, optional
            Whether this is a transient simulation (default: False).
            Affects solver strategy if convergence fails.
        
        Notes
        -----
        Convergence Strategy:
        
        **First Attempt:** Uses original solver settings from write_flowmodel()
        
        **Second Attempt (if first fails):**
        - Increases solver complexity to 'Complex'
        - Tightens convergence criteria (1e-4 outer, 1e-6 inner)
        - Enables backtracking with moderate settings
        
        **Third Attempt (if second fails):**
        - Very tight convergence criteria (1e-6 outer, 1e-7 inner)
        - Aggressive under-relaxation (DBD method)
        - Enhanced preconditioner settings
        - For transient: Increases timesteps to aid convergence
        
        Post-Processing (on successful run):
        - Extracts hydraulic heads from output
        - Processes specific discharge vectors
        - Extracts boundary flows (CHD, GHB) if applicable
        - Stores observation data if observations enabled
        - Calculates qx, qy, qz velocity components
        
        Sets Attributes
        ---------------
        gwf : flopy.mf6.ModflowGwf
            Groundwater flow model object.
        head : ndarray
            Hydraulic heads for final time step.
        spd : ndarray
            Specific discharge data structure.
        qx, qy, qz : ndarray
            Specific discharge components in x, y, z directions.
        chdflow : ndarray
            Constant head boundary flows (if CHD package used).
        ghbflow : ndarray
            General head boundary flows (if GHB package used).
        obsdata : object
            Observation data (if OBS package used).
        runtime : float
            Total execution time in seconds.
        
        Examples
        --------
        >>> # Run steady-state model
        >>> fm.run_flowmodel(sim)
        >>> print(f"Model converged in {fm.runtime:.2f} seconds")
        >>> print(f"Head range: {fm.head.min():.2f} to {fm.head.max():.2f} m")
        >>>
        >>> # Run transient model
        >>> fm.run_flowmodel(sim, transient=True)
        """
    
        t0 = datetime.now()

        print('Running simulation for ', self.scenario, ' ...')

        success, buff = sim.run_simulation(silent = True)   
        print('Model success = ', success)
        run_time = datetime.now() - t0
        print('run_time = ', run_time.total_seconds())
        
        def process_results():
                        
            gwf = sim.get_model(self.scenario)
            package_list = gwf.get_package_list()
            print(package_list)
            times = gwf.output.head().get_times()
            head = gwf.output.head().get_data() # last time
            print('head results shape ', head.shape)
            
            bud = gwf.output.budget()
            print('Type bud ', type(bud))
            spd = bud.get_data(text='DATA-SPDIS')[0]
            
            if self.chd == True:
                chdflow = bud.get_data(text='CHD')[-1]
                self.chdflow = chdflow

            if self.ghb == True:
                ghbflow = bud.get_data(text='GHB')[-1]
                self.ghbflow = ghbflow

            if self.obs == True:
                obs_data = gwf.obs
                self.obsdata = obs_data

            self.gwf = gwf
            self.head = head
            self.spd = spd
            self.qx, self.qy, self.qz = flopy.utils.postprocessing.get_specific_discharge(self.spd, self.gwf)
            self.runtime = run_time.total_seconds()
            
        if success:
            process_results()

        else:
            print('\nRe-writing IMS - Take 2 (SOMEWHAT MORE AGGRESSIVE)')
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
                print('\nRe-writing IMS - Take 3 (VERY AGGRESSIVE)')
                
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
                            outer_dvclose = 1e-6, #1e-4
                            inner_dvclose = 1e-7, #1e-6
                            outer_maximum = 300,
                            inner_maximum = 300,
                            linear_acceleration = "BICGSTAB",
                            reordering_method=['RCM'],
                            no_ptcrecord = ['ALL'],
                            under_relaxation = 'DBD',
                            under_relaxation_kappa = 0.2, #0.05 (aggressive) to 0.3
                            under_relaxation_theta = 0.5, # 0.5 - 0.9 Changes how quickly relaxation factor changes
                            under_relaxation_gamma = 0.1, # 0-0.2 doesnt make big difference
                            under_relaxation_momentum = 0.001, #0-0.001 doesn't make big difference
                            backtracking_number = 15,
                            backtracking_tolerance = 1.05, #1.01 (aggressive) to 10000
                            backtracking_reduction_factor = 0.7, # 0.1-0.3, or 0.9 when non-linear convergence HARD 
                            preconditioner_levels=18, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                            preconditioner_drop_tolerance=0.00001, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                            number_orthogonalizations=10,)
                sim.ims.write()
                success3, buff = sim.run_simulation(silent = True)   
                print('Model success3 = ', success3)
                
                if success3:
                    process_results()

    def get_watertable(self, geomodel, heads, hdry=-1e30):
        """
        Extract water table elevation from 3D hydraulic head results.
        
        Determines the water table elevation at each model cell by finding
        the uppermost active cell with a valid hydraulic head value.
        
        Parameters
        ----------
        geomodel : Geomodel
            Geological model containing grid structure and active cell information.
        heads : list of ndarray
            List containing hydraulic head arrays. Typically [head_array] where
            head_array has shape (ncell_disu,) for unstructured grids.
        hdry : float, optional
            Dry cell indicator value (default: -1e30).
        
        Returns
        -------
        watertable : ndarray
            Water table elevation for each cell column (ncpl,).
            Values of -999 indicate no water table found (all cells dry/inactive).
        
        Notes
        -----
        Algorithm:
        1. Convert DISU head results to DISV format for layer-by-layer processing
        2. For each cell column, search from top layer downward
        3. Find first active cell (idomain=1) with valid head (not hdry)
        4. Record that head value as water table elevation
        
        This method is essential for:
        - Mapping phreatic surfaces in unconfined aquifers
        - Calculating depth to groundwater
        - Visualizing groundwater flow patterns
        - Post-processing for ecological or agricultural applications
        
        Examples
        --------
        >>> # Extract water table from steady-state results
        >>> watertable = fm.get_watertable(geomodel, [fm.head])
        >>> print(f"Water table range: {watertable[watertable>-999].min():.1f} to "
        ...       f"{watertable[watertable>-999].max():.1f} m")
        >>>
        >>> # Use with multiple time steps (transient)
        >>> for t, head_t in enumerate(all_heads):
        ...     wt = fm.get_watertable(geomodel, [head_t])
        ...     print(f"Time {t}: Mean water table = {wt[wt>-999].mean():.2f} m")
        """
        nlay, ncpl = geomodel.cellid_disv.shape
    
        head_disv = -999 * np.ones((geomodel.ncell_disv))  
        watertable = -999 * np.ones(ncpl)
    
        for disucell in range(geomodel.ncell_disu):
            disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
            head_disv[disvcell] = heads[0][disucell]   
        head_disv = head_disv.reshape((nlay, ncpl))
    
        for icpl in range(ncpl):
            for lay in range(nlay): 
                if geomodel.idomain[lay, icpl] == 1:    # if present
                    if head_disv[lay, icpl] != hdry:    # if not dry
                        watertable[icpl] = head_disv[lay, icpl] 
                        break           
    
        return watertable

    def plot_watertable(self, spatial, geomodel, flowmodel, watertable, 
                        plot_grid = False, fname = '../figures/watertable.png', **kwargs):
        """
        Create a plan view plot of the water table elevation.
        
        Generates a colored contour map showing water table elevations across
        the model domain with optional overlay of wells and grid lines.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing well locations and domain boundaries.
        geomodel : Geomodel
            Geological model containing grid structure for plotting.
        flowmodel : Flowmodel
            Flow model object (typically self) containing scenario information.
        watertable : ndarray
            Water table elevations for each cell (from get_watertable()).
        plot_grid : bool, optional
            Whether to overlay model grid lines (default: False).
        fname : str, optional
            Output filename for saving the plot (default: '../figures/watertable.png').
        **kwargs
            x0, y0, x1, y1 : float
                Plot extent boundaries (default: uses spatial domain bounds).
        
        Examples
        --------
        >>> # Plot water table with default settings
        >>> watertable = fm.get_watertable(geomodel, [fm.head])
        >>> fm.plot_watertable(spatial, geomodel, fm, watertable)
        >>>
        >>> # Plot with grid overlay and custom extent
        >>> fm.plot_watertable(spatial, geomodel, fm, watertable,
        ...                    plot_grid=True, x0=700000, x1=710000,
        ...                    fname='../figures/detailed_watertable.png')
        """
        x0 = kwargs.get('x0', spatial.x0)
        y0 = kwargs.get('y0', spatial.y0)
        x1 = kwargs.get('x1', spatial.x1)
        y1 = kwargs.get('y1', spatial.y1)
        
        fig = plt.figure(figsize = (8,6))
        ax = plt.subplot(111)
        ax.set_title(flowmodel.scenario, size = 10)
        mapview = flopy.plot.PlotMapView(modelgrid=geomodel.vgrid)#, layer = layer)
        plan = mapview.plot_array(watertable, cmap='Spectral', alpha=0.8)#, vmin = vmin, vmax = vmax)
        if plot_grid:
            mapview.plot_grid(color = 'black', lw = 0.4)
        #if vectors:
        #    mapview.plot_vector(flowmodel.spd["qx"], flowmodel.spd["qy"], alpha=0.5)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('y (m)', size = 10)
        plt.colorbar(plan, shrink = 0.4)
           
        if hasattr(spatial, 'xyobsbores'):
            for j in range(spatial.nobs):
                ax.plot(spatial.xyobsbores[j][0], spatial.xyobsbores[j][1],'o', ms = '4', c = 'black')
                #ax.annotate(spatial.idobsbores[j], (spatial.xyobsbores[j][0], spatial.xyobsbores[j][1]+100), c='black', size = 10) #, weight = 'bold')
        
        for j in range(spatial.npump):
            ax.plot(spatial.xypumpbores[j][0], spatial.xypumpbores[j][1],'o', ms = '4', c = 'red')
            #ax.annotate(spatial.idpumpbores[j], (spatial.xypumpbores[j][0], spatial.xypumpbores[j][1]+100), c='red', size = 10) #, weight = 'bold')
            
        #ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color = 'black', lw = 1)
        plt.tight_layout() 
        plt.savefig(fname)

    def plot_plan(self, spatial, mesh, array, layer, extent = None, 
                  vmin = None, vmax = None, vectors = None):
        """
        Create a plan view plot of any model array with optional flow vectors.
        
        Generates a plan view visualization of specified model results (heads,
        hydraulic conductivity, etc.) with optional flow vector overlay and
        well locations.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing well locations.
        mesh : Mesh
            Mesh object for grid plotting (if needed).
        array : str
            Name of the flowmodel attribute to plot (e.g., 'head', 'logk11').
        layer : int
            Model layer to plot (may not be used depending on array type).
        extent : list, optional
            Plot extent as [x0, x1, y0, y1] (default: None).
        vmin, vmax : float, optional
            Color scale limits (default: None, auto-scale).
        vectors : bool, optional
            Whether to overlay flow vectors (default: None/False).
        
        Notes
        -----
        This general-purpose plotting method can visualize:
        - Hydraulic heads ('head')
        - Hydraulic conductivity ('logk11', 'logk33')
        - Any other flowmodel attribute arrays
        - Flow vectors (qx, qy) if requested
        
        Examples
        --------
        >>> # Plot hydraulic heads
        >>> fm.plot_plan(spatial, mesh, 'head', 0, vmin=50, vmax=150)
        >>>
        >>> # Plot with flow vectors
        >>> fm.plot_plan(spatial, mesh, 'head', 0, vectors=True,
        ...              extent=[700000, 710000, 6200000, 6210000])
        """
        
        fig = plt.figure(figsize = (8,6))
        ax = plt.subplot(111)
        ax.set_title(self.scenario, size = 10)
        mapview = flopy.plot.PlotMapView(model=self.gwf)#, layer = layer)
        plan = mapview.plot_array(getattr(self, array), cmap='Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
        if vectors:
            mapview.plot_vector(self.spd["qx"], self.spd["qy"], alpha=0.5)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('y (m)', size = 10)
        plt.colorbar(plan, shrink = 0.4)
           
        for j in range(spatial.nobs):
            ax.plot(spatial.xyobsbores[j][0], spatial.xyobsbores[j][1],'o', ms = '4', c = 'black')
            #ax.annotate(spatial.idobsbores[j], (spatial.xyobsbores[j][0], spatial.xyobsbores[j][1]+100), c='black', size = 10) #, weight = 'bold')
        
        for j in range(spatial.npump):
            ax.plot(spatial.xypumpbores[j][0], spatial.xypumpbores[j][1],'o', ms = '4', c = 'red')
            #ax.annotate(spatial.idpumpbores[j], (spatial.xypumpbores[j][0], spatial.xypumpbores[j][1]+100), c='red', size = 10) #, weight = 'bold')
            
        if mesh.plangrid == 'car': mesh.sg.plot(ax = ax, color = 'black', lw = 0.2) 
        if mesh.plangrid == 'tri': mesh.tri.plot(ax = ax, edgecolor='black', lw = 0.2)
        if mesh.plangrid == 'vor': mesh.vor.plot(ax = ax, edgecolor='black', lw = 0.2)
        ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color = 'black', lw = 1)
        plt.tight_layout() 
        plt.savefig('../figures/plan_%s.png' % array)

    def plot_transect(self, mesh, geomodel, array, title = None, grid = True,
                      vmin = None, vmax = None, lithology = False, contours = False, levels = None, xlim = None, ylim = None,
                      vectors = None, kstep = None, hstep = None, normalize = True, figsize = (12,4), label = None,
                      **kwargs):

        x0 = kwargs.get('x0', mesh.x0)
        y0 = kwargs.get('y0', mesh.y0)
        x1 = kwargs.get('x1', mesh.x1)
        y1 = kwargs.get('y1', mesh.y1)
        z0 = kwargs.get('z0', geomodel.z0)
        z1 = kwargs.get('z1', geomodel.z1)
    
        fig = plt.figure(figsize = figsize)
        ax = plt.subplot(111)
        
        if title is not None:
            ax.set_title(title, size = 10)
      
        xsect = flopy.plot.PlotCrossSection(model=self.gwf, line={"line": [(x0, y0),(x1, y1)]}, 
                                            #extent = [x0, x1, z0, z1], 
                                            geographic_coords=True)
        csa = xsect.plot_array(a = array, cmap = 'Spectral', alpha=0.8, vmin = vmin, vmax = vmax)

        if contours:
            cp = xsect.contour_array(a=array, levels=levels, linewidths=0.5, colors='black')
            ax.clabel(cp, inline=1, fontsize=8) # Adds labels

        if grid:
            xsect.plot_grid(lw = 0.5, color = 'black') 

        if lithology:
            for lay in range(geomodel.nlay):
                bot = geomodel.botm_geo[lay]
                xsect.plot(mesh.xc, bot)

        if vectors:
            xsect.plot_vector(self.qx, self.qy, self.qz, 
                              kstep = kstep,
                              hstep = hstep,
                              normalize=normalize, 
                              color="black")
        
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('z (m)', size = 10)
        
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([z0,z1])
        if xlim: 
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([x0,x1])

        plt.colorbar(csa, shrink = 0.6, label = label)

        plt.tight_layout()  
        plt.savefig('../figures/transect_%s.png' % title)
        plt.show()    
    

