import sys
import os
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon,MultiPolygon
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import flopy.plot
import pickle

from shapely.validation import make_valid

def clip_confinement_to_model(spatial, model_boundary, force_reclip = False):
    '''check = gpd.read_file('../data/data_shp/surficial_confinement.shp')
    print(check)
    print(check.shape)
    print(check.columns)
    print(check.geometry.head())'''

    surface_confinement_shp = '../data/data_shp/surficial_confinement.shp'
    output_dir = '../data/data_geology/surface_confinement/'
    os.makedirs(output_dir, exist_ok=True)

    surface_confinement = gpd.read_file(surface_confinement_shp)

    #code to fix any weird wonky 'invalid' shapes
    invalid = ~surface_confinement.is_valid
    print(f"Invalid geometries: {invalid.sum()} out of {len(surface_confinement)}")
    if invalid.sum() > 0:
        print(surface_confinement.loc[invalid, 'id'])
    surface_confinement['geometry'] = surface_confinement.geometry.apply(make_valid)
    print(f"Invalid after repair: {(~surface_confinement.is_valid).sum()}")

    
    cache_path = os.path.join(output_dir, "surface_confinement.pkl") #creating a cache path to save the multipolygon so we don't have to reclip every time

    if os.path.exists(cache_path) and not force_reclip:
        with open(cache_path, "rb") as f:
            combined_multipoly = pickle.load(f)
        setattr(spatial, "surface_confinement_multipoly", combined_multipoly)
        print(f"Loaded multipolygon from cache: {cache_path}")
        return
    
    print("Reclipping surface confinement shapefile to model boundary...")

    
    print("Model boundary CRS:", model_boundary.crs)
    print("Surface confinement CRS:", surface_confinement.crs)
    print("Model boundary bounds:", model_boundary.total_bounds)
    print("Confinement bounds:", surface_confinement.total_bounds)
    if surface_confinement.crs != model_boundary.crs: #spatial 
        surface_confinement = surface_confinement.to_crs(model_boundary.crs)
        
    confinement_clipped = gpd.clip(surface_confinement, model_boundary)

    def to_multipolygon(geom):
        if geom.geom_type == 'Polygon':
            return MultiPolygon([geom])
        elif geom.geom_type == 'MultiPolygon':
            return geom
        else:
            return None
            
    confinement_clipped['geometry'] = confinement_clipped['geometry'].apply(to_multipolygon)
    confinement_clipped = confinement_clipped[confinement_clipped['geometry'].notnull()]

    print(f"Number of features after clipping: {len(confinement_clipped)}")
    if len(confinement_clipped) == 0:
        print("WARNING: No overlapping geometries found between confinement and model boundary.")
        print("Check CRS and geometry overlap.")
        return
        
    full_poly = [] #this will combine all the seperate multipolygons into a single multipolygon file to call later
    for geom in confinement_clipped.geometry:
        full_poly.extend(list(geom.geoms))
    combined_multipoly = MultiPolygon(full_poly)
    setattr(spatial, "surface_confinement_multipoly", combined_multipoly)

    out_shp_path = os.path.join(output_dir, 'surface_confinement_clipped.shp')
    confinement_clipped.to_file(out_shp_path)
    print(f"Saved clipped shapefile: {out_shp_path}")

    with open(cache_path, "wb") as f:
        pickle.dump(combined_multipoly, f)
    print(f"Saved multipolygons to cache: {cache_path}")

def print_clipped_confinement_fig(model_boundary):
    clipped_confinement_file = '../data/data_geology/surface_confinement/surface_confinement_clipped.shp'
    output_dir = '../data/data_geology/surface_confinement/'
    os.makedirs(output_dir, exist_ok=True)
    
    clipped_confinement = gpd.read_file(clipped_confinement_file)

    fig, ax = plt.subplots(figsize=(8, 8))
    model_boundary.boundary.plot(ax=ax, color='black', linewidth=1)
    clipped_confinement.plot(ax=ax, column=None, color='purple', edgecolor='black', linewidth=0.2)
    ax.set_title("Clipped Surface Confinement")
    ax.axis('off')
    fig.tight_layout()
    
    fig_path = os.path.join(output_dir, 'clipped_surface_confinement_fig.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")