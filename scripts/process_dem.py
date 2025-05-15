import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import fiona
import rasterio
import rasterio.mask
import pickle
import flopy
from rasterio import shutil as rio_shutil

def make_geotif():
    # Create a geotiff from the DEM
    with rasterio.open("../Data/data_dem/output_hh.asc") as src:
        rio_shutil.copy(src, "../Data/data_dem/output_hh.tif", driver="GTiff")

make_geotif()

'''def crop_geotif():      
    geotiff_fname = "../Data/data_dem/output_hh.tif"
    uncropped_dem = DEM(geotiff_fname = geotiff_fname)
    bbox_path = "../Data/Data_shp//Otorowiri_Model_Extent.shp"
    uncropped_dem.crop_raster(bbox_path)

    geotiff_fname = '../data/data_dem/cropped_raster.tif'
    cropped_dem = DEM(geotiff_fname = geotiff_fname)
    #cropped_dem.plot_geotiff()

    # Resample DEM onto a coarse grid and save as xyz list to be used in structural model
    fname = project.workspace + 'topo_xyz.pkl'
    cropped_dem.topo = cropped_dem.resample_topo(project, structuralmodel, nrow = 20, ncol = 20, fname = fname)   # This creates a pickle file of resampled x,y,z to use in structural model
    #plt.imshow(cropped_dem.topo)
    levels = np.arange(-50, 300, 50)
    #cropped_dem.plot_topo(levels=levels)

    fname = project.workspace + 'topo_xyz.pkl'
    pickleoff = open(fname,'rb')
    topo_xyz = pickle.load(pickleoff)
    pickleoff.close()

        #topo_xyz'''

'''class DEM:
    
    def __init__(self, geotiff_fname):   
        self.geotiff_fname = geotiff_fname


    
    def crop_raster(self, bbox_path):
        with fiona.open(bbox_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        
        with rasterio.open(self.geotiff_fname) as src:
            print(src.crs)
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta
        
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
        
        with rasterio.open("../data/data_dem/cropped_raster.tif", "w", **out_meta) as dest:
            print(dest.crs)
            dest.write(out_image)

    def resample_topo(self, project, structuralmodel, nrow, ncol, fname):
        fine_topo = flopy.utils.Raster.load(self.geotiff_fname)
        delr = np.ones((ncol)) * ((structuralmodel.x1 - structuralmodel.x0)/ncol)
        delc = np.ones((nrow)) * ((structuralmodel.y1 - structuralmodel.y0)/nrow)
        resample_grid = flopy.discretization.structuredgrid.StructuredGrid(delc = delc, delr = delr, xoff = structuralmodel.x0, yoff = structuralmodel.y0)
        self.resample_grid = resample_grid
        resampled_topo = fine_topo.resample_to_grid(resample_grid, band=fine_topo.bands[0], method="linear", extrapolate_edges=True,)

        # Create a list x,y,z to put in structural model
        xyzcenters = resample_grid.xyzcellcenters
        xcenters = xyzcenters[0][0]
        ycenters = [xyzcenters[1][i][0] for i in range(nrow)]
        topo_xyz = []
        for col in range(ncol):
            for row in range(nrow):
                topo_xyz.append((xcenters[col], ycenters[row], resampled_topo[col, row]))
        
        pickle.dump(topo_xyz, open(os.path.join(fname),'wb'))
        return resampled_topo

    def load_topo(self, project):
        pickleoff = open(project.workspace + 'topo.pkl','rb')
        self.topo = pickle.load(pickleoff)
        pickleoff.close()

    def plot_geotiff(self):
        with rasterio.open(self.geotiff_fname) as src:
            data = src.read(1)
            nodata_value = src.nodata
    
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
    
    def plot_topo(self, levels):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot()
        ax.set_title('Top Elevation (m)')
        pmv = flopy.plot.PlotMapView(modelgrid=self.resample_grid)
        t = pmv.plot_array(self.topo)#, ec="0.75")
        cbar = plt.colorbar(t, shrink = 0.5)  
        cg = pmv.contour_array(self.topo, levels=levels, linewidths=0.8, colors="0.75")'''
