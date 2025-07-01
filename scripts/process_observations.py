import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import loopflopy.utils as utils
import flopy
import math
import os
import rasterio
from scipy.interpolate import griddata

def make_obs_gdf(df, geomodel, mesh, spatial):
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=spatial.epsg)

    mask = gdf.geometry.within(spatial.model_boundary_poly)
    if not mask.all():
        print("The following geometries are NOT within the polygon:")
        print(gdf[~mask])
    else:
        print("All geometries are within the polygon.")
    gdf = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points outside model

    #gdf = gdf[gdf['zobs'] != np.nan] # Don't include obs with no zobs
    gdf = gdf[~gdf['zobs'].isna()] # Don't include obs with no zobs
    gdf = gdf[gdf['zobs'] > geomodel.z0] # Don't include obs deeper than flow model bottom
    gdf = gdf.reset_index(drop=True)

    # Perform the intersection
    gdf['icpl'] = gdf.apply(lambda row: mesh.vgrid.intersect(row.Easting,row.Northing), axis=1)
    gdf['ground'] = gdf.apply(lambda row: geomodel.top_geo[row.icpl], axis=1)
    gdf['model_bottom'] = gdf.apply(lambda row: geomodel.botm[-1, row.icpl], axis=1)
    gdf['zobs-bot'] = gdf.apply(lambda row: row['zobs'] - row['model_bottom'], axis=1)

    for idx, row in gdf.iterrows():
        result = row['zobs'] - row['model_bottom']
        if result < 0:
            print(f"Bore {row['id']} has a zobs elevation below model bottom by: {result} m, removing from obs list")

    gdf = gdf[gdf['zobs-bot'] > 0] # filters out observations that are below the model bottom

    gdf['cell_disv'] = gdf.apply(lambda row: utils.xyz_to_disvcell(geomodel, row.Easting, row.Northing, row.zobs), axis=1)
    gdf['cell_disu'] = gdf.apply(lambda row: utils.disvcell_to_disucell(geomodel, row['cell_disv']), axis=1)  

    gdf['(lay,icpl)'] = gdf.apply(lambda row: utils.disvcell_to_layicpl(geomodel, row['cell_disv']), axis = 1)
    gdf['lay']        = gdf.apply(lambda row: row['(lay,icpl)'][0], axis = 1)
    gdf['icpl']       = gdf.apply(lambda row: row['(lay,icpl)'][1], axis = 1)
    gdf['obscell_xy'] = gdf['icpl'].apply(lambda icpl: (mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]))
    gdf['obscell_z']  = gdf.apply(lambda row: geomodel.zc[row['lay'], row['icpl']], axis=1)
    gdf['obs_zpillar']  = gdf.apply(lambda row: geomodel.zc[:, row['icpl']], axis=1)
    gdf['geolay']       = gdf.apply(lambda row: math.floor(row['lay']/geomodel.nls), axis = 1) # model layer to geolayer

    gdf.rename(columns={'Easting': 'x', 'Northing': 'y', 'zobs': 'z', 'Site Ref' : 'id'}, inplace=True) # to be consistent when creating obs_rec array

    # Make sure no pinched out observations
    if -1 in gdf['cell_disu'].values:
        print('Warning: some observations are pinched out. Check the model and data.')
        print('Number of pinched out observations: ', len(gdf[gdf['cell_disu'] == -1]))
        gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf