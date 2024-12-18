#!/usr/bin/env python
# coding: utf-8
import rasterio
import rasterio.mask
import fiona
import numpy as np
import matplotlib.pyplot as plt

def merge_rasters():    
    def asc_to_tif(fname):
        if not os.path.isdir(fname + '.tif'): 
            with rasterio.open(fname + '.asc') as src:
                data = src.read(1)#, masked=True)  # Read the first band
                #data[data == src.nodata] = -9999  # Example: Change NoData to -9999
                meta = src.meta.copy()  # Copy metadata
            meta.update(driver="GTiff") # Update metadata to set the driver to GeoTIFF
            meta.update(crs="EPSG:28350")
            with rasterio.open(fname + '.tif', "w", **meta) as dest: # Save the data to a new GeoTIFF file
                dest.write(data, 1)  # Write the data to the first band
                
    asc_to_tif('../data/data_dem/Hydro_Enforced_1_Second_DEM')
    asc_to_tif('../data/data_dem/Hydro_Enforced_1_Second_DEM_1')
    
    with rasterio.open("../data/data_dem/Hydro_Enforced_1_Second_DEM.tif") as src1, rasterio.open("../data/data_dem/Hydro_Enforced_1_Second_DEM_1.tif") as src2:
        # Read the data from the first band of each file
        data1 = src1.read(1, masked = True)
        data2 = src2.read(1, masked = True)
    
        # Get the transform and bounds for aligning the plots if they match
        extent1 = [src1.bounds.left, src1.bounds.right, src1.bounds.bottom, src1.bounds.top]
        extent2 = [src2.bounds.left, src2.bounds.right, src2.bounds.bottom, src2.bounds.top]
    
    # Plot the two rasters on the same plot
    plt.figure(figsize=(10, 8))
    plt.imshow(data1, cmap="Blues", extent=extent1, alpha=0.6)
    plt.imshow(data2, cmap="Oranges", extent=extent2, alpha=0.6)
    
    # Add a colorbar and labels
    plt.colorbar(label="Data Values", shrink = 0.4)
    plt.title("Overlay of Two GeoTIFFs")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
    # MERGE
    import rasterio
    from rasterio.merge import merge
    
    raster_files = ['../data/data_dem/Hydro_Enforced_1_Second_DEM.tif', '../data/data_dem/Hydro_Enforced_1_Second_DEM_1.tif'] # Define the list of raster file paths you want to merge
    src_files_to_merge = [rasterio.open(f) for f in raster_files] # Open the rasters and store them in a list
    merged_raster, out_transform = merge(src_files_to_merge, nodata = True) # Merge the rasters
    
    out_meta = src_files_to_merge[0].meta.copy() # Update the metadata for the new merged raster
    out_meta.update(crs="EPSG:28350")
    out_meta.update({
        "driver": "GTiff",
        "height": merged_raster.shape[1],
        "width": merged_raster.shape[2],
        "transform": out_transform})
    
    with rasterio.open("../data/data_dem/merged_raster.tif", "w", **out_meta) as dest: # Save the merged raster to a new file
        dest.write(merged_raster)
    
    for src in src_files_to_merge: # Close the opened source files
        src.close()

def crop_raster():
    with fiona.open("../data/data_shp/Extent.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    
    with rasterio.open("../data/data_dem/merged_raster.tif") as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    with rasterio.open("../data/data_dem/merged_cropped_raster.tif", "w", **out_meta) as dest:
        dest.write(out_image)


def plot_geotiff(tif_file):
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        nodata_value = src.nodata
        print(nodata_value)
    if nodata_value is not None:
        masked_data = np.ma.masked_equal(data, nodata_value)
    else:
        masked_data = data  # If no NoData value is set, use the data as is
    
    plt.figure(figsize=(10, 8))
    plt.imshow(masked_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Data Values', shrink = 0.5)
    plt.title('GeoTIFF Visualization (NoData Values Excluded)')
    plt.xlabel('Column Number')
    plt.ylabel('Row Number')
    plt.show()

def create_faultfunction():    
    ## ADD FAULT (this chunk given to me directly by Lachlan Grose to make an ellipsoid fault)
    from LoopStructural.modelling.features.fault._fault_function import CubicFunction, FaultDisplacement, Composite
    hw = CubicFunction()
    hw.add_cstr(0, 1)
    hw.add_grad(0, 0)
    hw.add_cstr(1, 0)
    hw.add_grad(1, 0)
    hw.add_max(1)
    fw = CubicFunction()
    fw.add_cstr(0, -1)
    fw.add_grad(0, 0)
    fw.add_cstr(-1, 0)
    fw.add_grad(-1, 0)
    fw.add_min(-1)
    gyf = CubicFunction()
    gyf.add_cstr(-1, 0)
    gyf.add_cstr(1, 0)
    gyf.add_cstr(-0.2, 1)
    gyf.add_cstr(0.2, 1)
    gyf.add_grad(0, 0)
    gyf.add_min(-1)
    gyf.add_max(1)
    gzf = CubicFunction()
    gzf.add_cstr(-1, 0)
    gzf.add_cstr(1, 0)
    gzf.add_cstr(-0.2, 1)
    gzf.add_cstr(0.2, 1)
    gzf.add_grad(0, 0)
    gzf.add_min(-1)
    gzf.add_max(1)
    gxf = Composite(hw, fw)
    fault_displacement = None
    fault_displacement = FaultDisplacement(gx=gxf, gy=gyf, gz=gzf)
    faultfunction = fault_displacement
    return(faultfunction)


def process_obs_steady(P, M):
    import numpy as np
    import os
    import pandas as pd
    M.hobs_steady = np.zeros((P.nobs, P.nzobs, 1), dtype = float)
    fname = str(M.modelname + "_steady.csv")
    csv_file = os.path.join(P.workspace, fname)
    data_set = pd.read_csv(csv_file)#, header=1)
    df = pd.DataFrame(data_set)
    a = df.to_numpy()
    hobs = a[0][1:(P.nobs*P.nzobs+1)]
    for ob_bore in range(P.nobs):
        for z in range(P.nzobs):
            n = ob_bore*P.nzobs + z
            M.hobs_steady[ob_bore, z] = hobs[n]
    return(M.hobs_steady)

def process_obs_past(P, M):
    import numpy as np
    import os
    import pandas as pd
    M.hobs_past = np.zeros((P.nobs, P.nzobs, P.nts_past), dtype = float) #P.nts_past-1)
    fname = str(M.modelname + "_past.csv")
    csv_file = os.path.join(P.workspace, fname)
    data_set = pd.read_csv(csv_file, header=0)
    data_frames = pd.DataFrame(data_set)
    hobs = np.array(data_frames.values)
    hobs = hobs[:,1:]
    hobs = np.swapaxes(hobs, 0, 1)
    for ob_bore in range(P.nobs):
        for z in range(P.nzobs):
            n = ob_bore*P.nzobs + z
            M.hobs_past[ob_bore, z, :] = hobs[n,:]
    return(M.hobs_past)

def process_obs_future(P, M):
    import numpy as np
    import os
    import pandas as pd
    M.hobs_future = np.zeros((P.nobs, P.nzobs, P.nts_future), dtype = float) # P.nts_future-1)
    fname = str(M.modelname + "_future.csv")
    csv_file = os.path.join(P.workspace, fname)
    data_set = pd.read_csv(csv_file, header=0)
    data_frames = pd.DataFrame(data_set)
    hobs = np.array(data_frames.values)
    hobs = hobs[:,1:]
    
    hobs = np.swapaxes(hobs, 0, 1)
    for ob_bore in range(P.nobs):
        for z in range(P.nzobs):
            n = ob_bore*P.nzobs + z
            M.hobs_future[ob_bore, z, :] = hobs[n,:]
    return(M.hobs_future)

# Plot observations
def plot_observations(heads, modelnames, ylim = None):
    import matplotlib.pyplot as plt
    colors = ['gray', 'red', 'green', 'purple', 'orange', 'blue']
    n = len(heads)
    fig = plt.figure(figsize = (12,3))
    fig.suptitle('Steady state heads')

    for ob in range(P.nobs): # New figure for each obs bore (OB1, OB2, OB3, OB4)
        ax = plt.subplot(1,5,ob+1)#, aspect = 'equal')
        ax.set_title(P.idobsbores[ob], size = 10) 
        for i in range(n): # for each option
            hobs = heads[i]   # extract obs data
            ax.plot(P.zobs[ob], hobs[ob,:,0], '-o', markersize = 4, c = colors[i], label = modelnames[i])

        ax.set_xlabel('Obs Depth (m)')
        if ylim != None:
            ax.set_ylim(ylim)
        if ob > 0: ax.set_yticks([])
        if ob == 0: ax.set_ylabel('Head (m)')
        #ax.set_xticks([0,1,2,3,4])
        #ax.set_xticklabels(P.zobs)
        #ax.set_xlim([10,30])
        #ax.set_yticks([30, 40, 50, 60])

    from matplotlib.lines import Line2D
    legend_markers = []
    for i in range(len(options)):
        legend_markers.append((Line2D([0], [0], marker='o', markersize = 4, color=colors[i])))
    ax = plt.subplot(1,5,5, aspect = 'equal')
    ax.set_axis_off()
    ax.legend(legend_markers, modelnames, loc="center", fontsize = 9)#, ncols = 3, bbox_to_anchor=[0.5, 0.7])
    plt.tight_layout()
    plt.show()


def find_watertable_disu(P, M, layer):
    model = M.gwf
    water_table = flopy.utils.postprocessing.get_water_table(M.gwf.output.head().get_data())
    M.heads_disv = -1e30 * np.ones_like(M.idomain, dtype=float) 
    for i, h in enumerate(water_table):
        if math.isnan(h) == False: 
            M.heads_disv[M.cellid_disu==i] = h        
    return(M.heads_disv[layer])

def plot_head_diff(P, M, heads1, heads2, vmin = None, vmax = None): 
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, aspect="equal")
    #M = pinchout_models[0]      
    pmv = flopy.plot.PlotMapView(modelgrid=M.vgrid)
    H = pmv.plot_array(heads1 - heads2, cmap = 'Spectral', alpha = 0.6)#vmin = vmin, vmax = vmax, 

    for j in range(len(P.xyobsbores)):
        ax.plot(P.xyobsbores[j][0], P.xyobsbores[j][1],'o', ms = '4', c = 'black')
        ax.annotate(P.idobsbores[j], (P.xyobsbores[j][0], P.xyobsbores[j][1]+100), c='black', size = 12) #, weight = 'bold')

    for j in range(len(P.xypumpbores)):
        ax.plot(P.xypumpbores[j][0], P.xypumpbores[j][1],'o', ms = '4', c = 'red')
        ax.annotate(P.idpumpbores[j], (P.xypumpbores[j][0], P.xypumpbores[j][1]+100), c='red', size = 12) #, weight = 'bold')

    if M.plan == 'car': P.sg.plot(ax=ax, edgecolor='black', lw = 0.2)
    if M.plan == 'tri': P.tri.plot(ax=ax, edgecolor='black', lw = 0.2)
    if M.plan == 'vor': P.vor.plot(ax=ax, edgecolor='black', lw = 0.2)
    ax.set_title('Head diffence between two most extreme structural models', size = 10)
    plt.colorbar(H, shrink = 0.4)


# Plot observations
def param_vs_struct(param_obs_heads, pinchout_obs_heads, xlim = None, ylim = None):

    fig = plt.figure(figsize = (12,3))
    fig.suptitle('Steady state heads')

    for ob in range(P.nobs): # New figure for each obs bore (OB1, OB2, OB3, OB4)
        a = np.array(param_obs_heads)[:,ob,:,0] #16,4,5,1
        b = np.array(pinchout_obs_heads)[:,ob,:,0] #6,4,5,1
        a_max = np.nanpercentile(a, 0, axis=0)
        a_min = np.nanpercentile(a, 100, axis=0)
        b_max = np.nanpercentile(b, 0, axis=0)
        b_min = np.nanpercentile(b, 100, axis=0)
    
        ax = plt.subplot(1,5,ob+1)#, aspect = 'equal')
        ax.set_title(P.idobsbores[ob], size = 10) 
        for i in range(16): # for each option
            h = param_obs_heads[i]   # extract obs data
            ax.plot(P.zobs, h[ob,:,0], '-', lw = 0.2, alpha = 0.7, c = 'blue')
        for i in range(6): # for each option
            h = pinchout_obs_heads[i]   # extract obs data
            ax.plot(P.zobs, h[ob,:,0], '-', lw = 0.2, alpha = 0.7, c = 'red')
        ax.fill_between(P.zobs, a_min, a_max, color = 'blue', alpha = 0.1)
        ax.fill_between(P.zobs, b_min, b_max, color = 'red', alpha = 0.1)
        ax.plot(P.zobs, a_min, '-', lw = 2, c = 'blue')
        ax.plot(P.zobs, a_max, '-', lw = 2, c = 'blue')
        ax.plot(P.zobs, b_min, '-', lw = 2, c = 'red')
        ax.plot(P.zobs, b_max, '-', lw = 2, c = 'red')
        
        ax.set_xlabel('Obs Depth (m)')
        if xlim != None: ax.set_xlim(xlim)
        if ylim != None: ax.set_ylim(ylim)
        if ob > 0: ax.set_yticks([])
        if ob == 0: ax.set_ylabel('Head (m)')
        #ax.set_xticks([0,1,2,3,4])
        #ax.set_xticklabels(P.zobs)
        #ax.set_xlim([10,30])
        #ax.set_yticks([30, 40, 50, 60])

    from matplotlib.lines import Line2D
    legend_markers = []
    legend_markers.append((Line2D([0], [0], marker='o', markersize = 4, color='blue')))
    legend_markers.append((Line2D([0], [0], marker='o', markersize = 4, color='red')))
    ax = plt.subplot(1,5,5, aspect = 'equal')
    ax.set_axis_off()
    ax.legend(legend_markers, ['parameter variation', 'structural variation'], loc="center", fontsize = 9)#, ncols = 3, bbox_to_anchor=[0.5, 0.7])
    plt.tight_layout()
    plt.show()

