import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape

class Observations:
    def __init__(self):

            self.observations_label = "ObservationsBaseClass"
        
    def obs_bores(spatial):   
    obsbore_df = pd.read_excel('../data/data_dwer/Formation picks.xls', sheet_name = 'bore_info')
    obsbore_gdf = gpd.GeoDataFrame(obsbore_df, geometry=gpd.points_from_xy(obsbore_df.Easting, obsbore_df.Northing), crs="epsg:28350")
    obsbore_gdf = gpd.clip(obsbore_gdf, spatial.inner_boundary_poly).reset_index(drop=True)
    spatial.idobsbores = list(obsbore_gdf.ID)
    spatial.xyobsbores = list(zip(obsbore_gdf.Easting, obsbore_gdf.Northing))
    spatial.nobs = len(spatial.xyobsbores)
    spatial.obsbore_gdf = obsbore_gdf
    
    
    def process_obs(self, spatial, geomodel, mesh):

        # Get observation elevation (z) from dataframe
        depth = spatial.obsbore_gdf.zobs_mbgl.tolist()
        zobs = []
        for n in range(spatial.nobs):
            icpl = mesh.obs_cells[n]
            zobs.append(geomodel.top_geo[icpl] - depth[n])
        
        xobs, yobs = spatial.obsbore_gdf.Easting.tolist(), spatial.obsbore_gdf.Northing.tolist(), 
        obslist = list(zip(xobs, yobs, zobs))
    
        # Cretae input arrays
        obs_rec = []
        for i, cell in enumerate(mesh.obs_cells):
            x,y,z = obslist[i][0], obslist[i][1], obslist[i][2]
            point = Point(x,y,z)
            lay, icpl = geomodel.vgrid.intersect(x,y,z)
            cell_disv = icpl + lay*mesh.ncpl
            cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
            obs_rec.append([spatial.idobsbores[i], 'head', (cell_disu+1)])   
    
        self.obs_rec = obs_rec