import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import  mapping
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from pyproj import CRS

def model_DEM(spatial):
    # Read the DEM file and set the CRS
    asc_path = "C:\\Users\\00105010\\Projects\\Otorowiri\\data\\data_elevation\\rasters_COP30\\output_hh.asc"
    #crs=spatial.epsg
    crs = rasterio.crs.CRS.from_epsg(spatial.epsg) # Set the CRS to EPSG:28350
    model_geom = [mapping(spatial.model_boundary_poly)] # Convert the polygon to a format suitable for rasterio

    with rasterio.open(asc_path) as src:
        # reproject the ASC crs to the model crs
        transform, width, height = calculate_default_transform(
        src.crs, crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height})
        # Reproject to memory
        with MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.bilinear)
                # Mask (clip) the raster to the polygon
                out_image, out_transform = mask(dst, model_geom, crop=True)
                out_meta = dst.meta.copy()
                out_meta.update({"height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
    output_filename = "Otorowiri_Model_DEM.asc"
    output_path = os.path.join("..", "data", "data_dem", output_filename)
    ## Create the gdf for the DEM data    
    spatial.dem_gdf = gpd.GeoDataFrame({'geometry': [spatial.model_boundary_poly]}, crs=crs)
    masked_data = np.ma.masked_equal(out_image, -9999)  # Mask the NoData values
    spatial.dem_gdf['elevation'] = [float(masked_data.mean())]  # Calculate the mean elevation
    # Save the clipped DEM to the new ASC file
    with open(output_path, 'w') as asc_file:
        # Write the ASC header
        asc_file.write(f"ncols         {out_image.shape[2]}\n")
        asc_file.write(f"nrows         {out_image.shape[1]}\n")
        asc_file.write(f"xllcorner     {out_transform[2]}\n")  # bottom-left x
        asc_file.write(f"yllcorner     {out_transform[5]}\n")  # bottom-left y
        asc_file.write(f"cellsize      {out_transform[0]}\n")  # pixel size
        asc_file.write(f"NODATA_value  -9999\n")  # NoData value
        # Write the elevation data
        for row in out_image[0]:  # Assuming you're working with a single-band raster
            asc_file.write(' '.join(map(str, row)) + '\n')
    
    #Write companion .prj file to store CRS metadata
    prj_path = output_path.replace(".asc", ".prj")
    with open(prj_path, 'w') as prj_file:
        prj_file.write(crs.to_wkt())

    # Save the GeoTIFF version of the DEM
    tiff_output_filename = "Otorowiri_Model_DEM.tif"
    tiff_output_path = os.path.join("..", "modelfiles", tiff_output_filename)
    
    spatial.model_DEM = output_path
    spatial.model_DEM_2 = tiff_output_path ###use the geotiff option if the asc option doesn't work

    with rasterio.open(tiff_output_path,
    'w',
    driver='GTiff',
    height=out_image.shape[1],
    width=out_image.shape[2],
    count=1,
    dtype=out_image.dtype,
    crs=crs,
    transform=out_transform,
    nodata=-9999) as dst:
        dst.write(out_image[0], 1)

    #plotting the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    image = rasterio.plot.show(out_image[0], transform=out_transform, cmap='terrain', ax=ax)
    cbar = plt.colorbar(image.get_images()[0], ax=ax, label='Elevation (m)')
    ax.set_title("Clipped Elevation Map")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    plt.show()

    return output_filename