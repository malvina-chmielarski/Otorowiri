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

def clip_veg_to_model(spatial, model_boundary):
    #list_of_veg_files = ['../data/data_woody/MC_WC_l1emm_1972_NCAS_sh50_woody_geo_GDA94_MGA50.shp']
    list_of_veg_files = [str(p) for p in Path('../data/data_woody').glob('*.shp')]
    output_dir = '../data/data_woody/clipped_woody/'
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(output_dir, "veg_multipolygons_by_year.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            veg_multipolygons_by_year = pickle.load(f)
        for year, multipoly in veg_multipolygons_by_year.items():
            setattr(spatial, f"{year}_multipoly", multipoly)
        print(f"Loaded multipolygons from cache: {cache_path}")
        return

    veg_multipolygons_by_year = {}

    for shp_path in list_of_veg_files:
        veg = gpd.read_file(shp_path)
        if veg.crs != model_boundary.crs: #spatial 
            veg = veg.to_crs(model_boundary.crs)
        
        veg_clipped = gpd.clip(veg, model_boundary)

        def to_multipolygon(geom):
            if geom.geom_type == 'Polygon':
                return MultiPolygon([geom])
            elif geom.geom_type == 'MultiPolygon':
                return geom
            else:
                return None
            
        veg_clipped['geometry'] = veg_clipped['geometry'].apply(to_multipolygon)
        veg_clipped = veg_clipped[veg_clipped['geometry'].notnull()]

        filename = os.path.basename(shp_path)
        year = filename[12:16]
        
        all_polys = [] #this will combine all the seperate multipolygons into a single multipolygon file to call later
        for geom in veg_clipped.geometry:
            all_polys.extend(list(geom.geoms))

        combined_multipoly = MultiPolygon(all_polys)
        veg_multipolygons_by_year[year] = combined_multipoly
        attr_name = f"{year}_multipoly"
        setattr(spatial, attr_name, combined_multipoly)

        out_shp_path = os.path.join(output_dir, f'vegetation_clipped_{year}.shp')
        veg_clipped.to_file(out_shp_path)
        print(f"Saved: {out_shp_path}")

    with open(cache_path, "wb") as f:
        pickle.dump(veg_multipolygons_by_year, f)
    print(f"Saved multipolygons to cache: {cache_path}")

def print_clipped_veg_figures(model_boundary):
    clipped_veg_files = [str(p) for p in Path('../data/data_woody/clipped_woody').glob('*.shp')]
    output_dir = '../data/data_woody/clipped_woody/veg_figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    for shp_path in clipped_veg_files:
        veg = gpd.read_file(shp_path)
        filename = os.path.basename(shp_path)
        year = filename[:-4][-4:]

        fig, ax = plt.subplots(figsize=(8, 8))
        model_boundary.boundary.plot(ax=ax, color='black', linewidth=1)
        veg.plot(ax=ax, column=None, color='green', edgecolor='black', linewidth=0.2)
        ax.set_title(f"Vegetation Cover {year}")
        ax.axis('off')
        fig.tight_layout()

        fig_path = os.path.join(output_dir, f'vegetation_clipped_{year}.png')
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved figure: {fig_path}")

'''
def project_veg_onto_mesh(mesh, model_boundary):
    clipped_veg_files = [str(p) for p in Path('../data/data_woody/clipped_woody').glob('*.shp')]
    output_dir = '../data/data_woody/clipped_woody/veg_on_mesh_figures/'
    os.makedirs(output_dir, exist_ok=True)

    for shp_path in clipped_veg_files:
        veg = gpd.read_file(shp_path)
        filename = os.path.basename(shp_path)
        year = filename[:-4][-4:]

        fig, ax = plt.subplots(figsize=(8, 8))
        model_boundary.boundary.plot(ax=ax, color='black', linewidth=1)
        pmv = flopy.plot.PlotMapView(modelgrid=mesh.vgrid, ax=ax)
        pmv.plot_grid(lw=0.2, color="gray")
        veg.plot(ax=ax, color='green', edgecolor='black', linewidth=0.5)
        ax.set_title(f"Vegetation on Mesh {year}")
        ax.axis('off')
        fig.tight_layout()

        fig_path = os.path.join(output_dir, f'vegetation_on_mesh_{year}.png')
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

        print(f"Saved: {fig_path}")'''
