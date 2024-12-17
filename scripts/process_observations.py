import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape

class Observations:
    def __init__(self):

            self.observations_label = "ObservationsBaseClass"
        
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