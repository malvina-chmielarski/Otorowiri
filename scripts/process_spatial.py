import shapely.geometry as sg
from shapely.geometry import LineString, LinearRing, Point,Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import loopflopy
from loopflopy.mesh_routines import resample_linestring, resample_shapely_poly, resample_gdf_poly
from shapely.affinity import translate

def remove_duplicate_points(polygon):
    # Extract unique points using LinearRing
    linear_ring = LinearRing(polygon.exterior.coords)
    unique_coords = list(linear_ring.coords)
    # Reconstruct the Polygon without duplicates
    return Polygon(unique_coords)

    # Apply to a GeoDataFrame
    #gdf = gpd.GeoDataFrame({'geometry': [polygon]})

def create_polygon_from_xy(x_coords, y_coords):
    coords= list(zip(x_coords, y_coords))
    # Check if the first and last coords are the same (ie if the polygon is closed)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    # Create a Polygon from the coordinates
    return Polygon(coords)

def make_bbox_shp(spatial, x0, x1, y0, y1):
    xcoords = [x0, x1, x1, x0, x0]
    ycoords = [y0, y0, y1, y1, y0]
    bbox = Polygon(list(zip(xcoords, ycoords)))
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs = spatial.epsg)
    bbox_gdf.to_file('../data/data_shp/bbox/bbox.shp')
    spatial.bbox_gdf = bbox_gdf

def model_boundary(spatial, boundary_buff, simplify_tolerance, node_spacing):
    model_boundary_shp_fname = '../data/data_shp/Otorowiri_Model_Extent_update_new.shp'
    model_boundary_gdf = gpd.read_file(model_boundary_shp_fname)
    model_boundary_gdf.to_crs(epsg=spatial.epsg, inplace=True) 
    model_boundary_poly = Polygon(model_boundary_gdf.geometry.iloc[0])
    model_boundary_gs = model_boundary_gdf.geometry.simplify(tolerance=simplify_tolerance, preserve_topology=True) # simplify 
    model_boundary_poly = resample_gdf_poly(model_boundary_gs, node_spacing) # resample 
    
    # Create an inner boundary for meshing
    inner_boundary_poly = model_boundary_poly.buffer(-boundary_buff)
    inner_boundary_gs = gpd.GeoSeries([inner_boundary_poly])
    inner_boundary_poly = resample_gdf_poly(inner_boundary_gs, 0.95*node_spacing) #  A few less nodes in inside boundary
    
    #refinement_boundary_gs = model_boundary_gdf.buffer(5000)   
    #refinement_boundary_poly = resample_poly(refinement_boundary_gs, 3000) 
    spatial.boundary_buff = boundary_buff
    spatial.model_boundary_gdf = model_boundary_gdf
    model_boundary_gdf.to_file('../modelfiles/model_boundary.shp')
    spatial.model_boundary_poly = model_boundary_poly
    spatial.inner_boundary_poly = inner_boundary_poly
    spatial.x0, spatial.y0, spatial.x1, spatial.y1 = model_boundary_poly.bounds

def head_boundary(spatial):    

    # WEST
    inner_boundary_poly = spatial.model_boundary_poly.buffer(-1)
    coords = list(inner_boundary_poly.exterior.coords)
    new_coords = []
    for coord in coords:
        x,y = coord[0], coord[1]
        if x < 380000:
            if y < (spatial.y1 - 5):
                if y > (spatial.y0 + 5):
                    new_coords.append(coord)
 
    chd_west_ls = LineString(new_coords)
    chd_west_gdf = gpd.GeoDataFrame({'geometry': [chd_west_ls]})
    spatial.chd_west_gdf = chd_west_gdf
    spatial.chd_west_ls = chd_west_ls

    # SOUTH
    new_coords = []
    for coord in coords:
        x,y = coord[0], coord[1]
        if y < (spatial.y0 + 5):
            new_coords.append((x,y))

    ghb_south_ls = LineString(new_coords)
    ghb_south_ls = translate(ghb_south_ls, xoff=0, yoff=200)
    ghb_south_gdf = gpd.GeoDataFrame({'geometry': [ghb_south_ls]})
    spatial.ghb_south_gdf = ghb_south_gdf
    spatial.ghb_south_ls = ghb_south_ls

    # NORTH
    new_coords = []
    for coord in coords:
        x,y = coord[0], coord[1]
        if y > (spatial.y1 - 5):
            new_coords.append((x,y))

    ghb_north_ls = LineString(new_coords)
    ghb_north_ls = translate(ghb_north_ls, xoff=0, yoff=-200)
    ghb_north_gdf = gpd.GeoDataFrame({'geometry': [ghb_north_ls]})
    spatial.ghb_north_gdf = ghb_north_gdf
    spatial.ghb_north_ls = ghb_north_ls
    

def head_boundary2(spatial):    
    coast_gdf = gpd.read_file('../data/data_shp/coast/Coastline_LGATE_070.shp')
    coast_gdf.to_crs(epsg=spatial.epsg, inplace=True)
    coast_gdf = gpd.clip(spatial.model_boundary_gdf, spatial.bbox_gdf).reset_index(drop=True)    
    coast_ls = coast_gdf.geometry[0]
    coast_ls = translate(coast_ls, xoff=10, yoff=0)
    chd_west_gdf = gpd.GeoDataFrame({'geometry': [coast_ls]})
    spatial.chd_west_gdf = chd_west_gdf
    spatial.chd_west_ls = coast_ls

def geo_bores(spatial):   
    df = pd.read_excel('../data/data_geology/bore_data.xlsx', sheet_name = 'geo')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=spatial.epsg)
    gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)
    spatial.geobore_gdf = gdf
    spatial.idgeobores = list(gdf.ID)
    spatial.xygeobores = list(zip(gdf.Easting, gdf.Northing))
    spatial.nobs = len(spatial.xygeobores)

'''def obs_bores(spatial):   
    df = pd.read_excel('../data/data_geology/Otorowiri_Model_Geology.xlsx', sheet_name = 'mAHD')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=spatial.epsg)
    gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)
    spatial.obsbore_gdf = gdf
    spatial.idobsbores = list(gdf.ID)
    spatial.xyobsbores = list(zip(gdf.Easting, gdf.Northing))
    spatial.nobs = len(spatial.xyobsbores)'''

def obs_bores(spatial): #, observations):   
    df = pd.read_excel('../data/data_geology/Otorowiri_Model_Geology.xlsx', sheet_name = 'mAHD')
    #df = observations.df_boredetails
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=spatial.epsg)
    gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)
    spatial.obsbore_gdf = gdf
    spatial.idobsbores = list(gdf.ID)
    spatial.xyobsbores = list(zip(gdf.Easting, gdf.Northing))
    spatial.nobs = len(spatial.xyobsbores)

def pump_bores(spatial):    
    df = pd.read_excel('../data/data_geology/Otorowiri_Model_Geology.xlsx', sheet_name = 'pumping_bores')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=spatial.epsg)
    gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)
    spatial.pumpbore_gdf = gdf
    spatial.idpumpbores = list(gdf.ID)
    spatial.xypumpbores = list(zip(gdf.Easting, gdf.Northing))
    spatial.npump = len(spatial.xypumpbores)

def outcrop(spatial, buffer_distance, node_spacing, threshold):
    Otorowiri_1_df = pd.read_excel('../data/data_geology/Otorowiri_outcrop.xlsx', sheet_name = 'Otorowiri_1')
    Otorowiri_1_df = Otorowiri_1_df.dropna(subset=['Easting', 'Northing'])
    Otorowiri_1_raw_poly = create_polygon_from_xy(Otorowiri_1_df['Easting'], Otorowiri_1_df['Northing'])
    print(Otorowiri_1_raw_poly)
    Otorowiri_1_gdf = gpd.GeoDataFrame({'geometry': [Otorowiri_1_raw_poly]}, crs=spatial.epsg)
    Otorowiri_1_gdf = gpd.clip(Otorowiri_1_gdf, spatial.model_boundary_poly).reset_index(drop=True)
    Otorowiri_1_gs = Otorowiri_1_gdf.geometry.simplify(tolerance=100, preserve_topology=True) # simplify
    Otorowiri_1_gs = Otorowiri_1_gdf.geometry.buffer(buffer_distance) # buffer
    Otorowiri_1_poly = unary_union(Otorowiri_1_gs) # union
    Otorowiri_1_poly = resample_shapely_poly(Otorowiri_1_poly, node_spacing) # resample
    from loopflopy.mesh_routines import remove_close_points
    cleaned_coords = remove_close_points(list(Otorowiri_1_poly.exterior.coords), threshold) # Clean the polygon exterior
    Otorowiri_1_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates

    Otorowiri_2_df = pd.read_excel('../data/data_geology/Otorowiri_outcrop.xlsx', sheet_name = 'Otorowiri_2')
    Otorowiri_2_df = Otorowiri_2_df.dropna(subset=['Easting', 'Northing'])
    Otorowiri_2_raw_poly = create_polygon_from_xy(Otorowiri_2_df['Easting'], Otorowiri_2_df['Northing'])
    print(Otorowiri_2_raw_poly)
    Otorowiri_2_gdf = gpd.GeoDataFrame({'geometry': [Otorowiri_2_raw_poly]}, crs=spatial.epsg)
    Otorowiri_2_gdf = gpd.clip(Otorowiri_2_gdf, spatial.model_boundary_poly).reset_index(drop=True)
    Otorowiri_2_gs = Otorowiri_2_gdf.geometry.simplify(tolerance=100, preserve_topology=True) # simplify
    Otorowiri_2_gs = Otorowiri_2_gdf.geometry.buffer(buffer_distance) # buffer
    Otorowiri_2_poly = unary_union(Otorowiri_2_gs) # union
    Otorowiri_2_poly = resample_shapely_poly(Otorowiri_2_poly, node_spacing) # resample
    from loopflopy.mesh_routines import remove_close_points
    cleaned_coords = remove_close_points(list(Otorowiri_2_poly.exterior.coords), threshold) # Clean the polygon exterior
    Otorowiri_2_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates

    Otorowiri_3_df = pd.read_excel('../data/data_geology/Otorowiri_outcrop.xlsx', sheet_name = 'Otorowiri_3')   
    Otorowiri_3_df = Otorowiri_3_df.dropna(subset=['Easting', 'Northing'])
    Otorowiri_3_raw_poly = create_polygon_from_xy(Otorowiri_3_df['Easting'], Otorowiri_3_df['Northing'])
    print(Otorowiri_3_raw_poly)
    Otorowiri_3_gdf = gpd.GeoDataFrame({'geometry': [Otorowiri_3_raw_poly]}, crs=spatial.epsg)
    Otorowiri_3_gdf = gpd.clip(Otorowiri_3_gdf, spatial.model_boundary_poly).reset_index(drop=True)
    Otorowiri_3_gs = Otorowiri_3_gdf.geometry.simplify(tolerance=100, preserve_topology=True) # simplify
    Otorowiri_3_gs = Otorowiri_3_gdf.geometry.buffer(buffer_distance) # buffer
    Otorowiri_3_poly = unary_union(Otorowiri_3_gs) # union
    Otorowiri_3_poly = resample_shapely_poly(Otorowiri_3_poly, node_spacing) # resample
    from loopflopy.mesh_routines import remove_close_points
    cleaned_coords = remove_close_points(list(Otorowiri_3_poly.exterior.coords), threshold) # Clean the polygon exterior
    Otorowiri_3_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates

    spatial.outcrop_poly = [Otorowiri_1_poly, Otorowiri_2_poly, Otorowiri_3_poly]
    spatial.outcrop_gdf = gpd.GeoDataFrame(geometry = [Otorowiri_1_poly, Otorowiri_2_poly, Otorowiri_3_poly])
    spatial.outcrop_nodes = list(spatial.outcrop_gdf.geometry[0].exterior.coords) 

def faults(spatial):  
    faults_gdf = gpd.read_file('../data/data_shp/baragoon_seismic/baragoon_seismic_faults.shp')
    faults_gdf.to_crs(epsg=spatial.epsg, inplace=True)
    faults_gdf = gpd.clip(faults_gdf, spatial.model_boundary_poly).reset_index(drop=True)
    
    #from meshing_routines import remove_close_points
    #threshold = 1000
    #cleaned_coords = remove_close_points(list(streams_poly.exterior.coords), threshold) # Clean the polygon exterior
    #streams_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates
   # 
   # streams_gdf = gpd.GeoDataFrame(geometry=list(streams_multipoly.geoms))

    faults_gdf.loc[int(1), "id"]  = 'Leed1'
    faults_gdf.loc[int(2), "id"] = 'Horstye'
    faults_gdf.loc[int(4), "id"] = 'Yarra1'
    faults_gdf.loc[int(10), "id"] = 'Biggestmeanest'
    faults_gdf = faults_gdf.drop(index=[0, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14])
    faults_gdf = faults_gdf.reset_index(drop = True)
    
    #### FAULTS AS LINESTRINGS
        
    r = 1000 # distance between points
    threshold_distance = 1000 # Don't want fault nodes too close to boundaries.
    faults_ls, faults_coords = [], []
    faults1_coords, faults2_coords = [], []

    for i, ls in enumerate(faults_gdf.geometry): # For each fault...    
        
        # STEP 1: Resample fault to have evenly spaced nodes at distance r
        ls_resample = resample_linestring(ls, r) # Resample linestring

        # STEP 2: Removing nodes too close to inner and outer boundary so mesh doesn't go crazy refined (threshold_distance)
        nodes_to_remove = []
        for p1 in ls_resample:
            for p2 in spatial.inner_boundary_poly.exterior.coords:
                p2 = Point(p2)
                if p1.distance(p2) <= threshold_distance:
                    nodes_to_remove.append(p1)
            for p3 in spatial.model_boundary_poly.exterior.coords:
                p3 = Point(p3)
                if p1.distance(p3) <= threshold_distance:
                    nodes_to_remove.append(p1)
        if nodes_to_remove:
            print('Removing faults nodes on ', faults_gdf.loc[i, 'id'], ' because too close to boundary: ', nodes_to_remove)
        ls_new = [node for node in ls_resample if node not in nodes_to_remove]
        faults_ls.append(ls_new) # list of shapely points

        # STEP 3: Don't include short faults with only 1 node
        p = []
        for point in ls_new:
            x,y = point.x, point.y
            p.append((x,y))
        if len(p) > 1: # just making sure very short dykes with 1 point are not included
            faults_coords.append(p)
    
    faults_nodes = list(itertools.chain.from_iterable(faults_coords))
    faults_multipoint = MultiPoint(faults_nodes)
    
    spatial.faults_gdf = faults_gdf   
    spatial.faults_nodes = faults_nodes
    spatial.faults_multipoint = faults_multipoint

def lakes(spatial):  

    gdf = gpd.read_file('../data/data_shp/EPP_Lakes.shp')
    gdf.to_crs(epsg=28350, inplace=True)
    gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)
    spatial.lakes_gdf = gdf

def river(spatial, buffer_distance, node_spacing, threshold):  

    gdf = gpd.read_file('../data/data_shp/Model_Streams.shp')
    gdf.to_crs(epsg=28350, inplace=True)
    gdf = gpd.clip(gdf, spatial.model_boundary_poly).reset_index(drop=True)

    ##Arrowsmith River polygons
    Arrowsmith_gdf = gdf[((gdf['something'] == 'Arrowsmith_1') | (gdf['something'] == 'Arrowsmith_2') | (gdf['something'] == 'Arrowsmith_3'))]
    Arrowsmith_gs = Arrowsmith_gdf.buffer(buffer_distance)
    Arrowsmith_poly = unary_union(Arrowsmith_gs)
    print(Arrowsmith_poly)
    Arrowsmith_poly = resample_shapely_poly(Arrowsmith_poly, node_spacing) # streams_multipoly = streams_gdf

    from loopflopy.mesh_routines import remove_close_points
    cleaned_coords = remove_close_points(list(Arrowsmith_poly.exterior.coords), threshold) # Clean the polygon exterior
    Arrowsmith_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates
    
    ##Another_Creek River polygons
    Small_creek_gdf = gdf[(gdf['something'] == 'Another_Creek')]
    Small_creek_gs = Small_creek_gdf.buffer(buffer_distance)
    Small_creek_poly = unary_union(Small_creek_gs)
    Small_creek_poly = resample_shapely_poly(Small_creek_poly, node_spacing) # streams_multipoly = streams_gdf

    from loopflopy.mesh_routines import remove_close_points
    cleaned_coords = remove_close_points(list(Small_creek_poly.exterior.coords), threshold) # Clean the polygon exterior
    Small_creek_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates

    ##Sand_Plain_Creek River polygons
    Sand_Plain_Creek_gdf = gdf[(gdf['something'] == 'Sand_Plain_Creek_1')]#| (gdf['something'] == 'Sand_Plain_Creek_2')]
    Sand_Plain_Creek_gs = Sand_Plain_Creek_gdf.buffer(buffer_distance)
    Sand_Plain_Creek_poly = unary_union(Sand_Plain_Creek_gs)
    Sand_Plain_Creek_poly = resample_shapely_poly(Sand_Plain_Creek_poly, node_spacing) # streams_multipoly = streams_gdf

    from loopflopy.mesh_routines import remove_close_points
    cleaned_coords = remove_close_points(list(Sand_Plain_Creek_poly.exterior.coords), threshold) # Clean the polygon exterior
    Sand_Plain_Creek_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates

    spatial.river_poly = [Arrowsmith_poly, Small_creek_poly, Sand_Plain_Creek_poly]
    spatial.river_gdf = gpd.GeoDataFrame(geometry = [Arrowsmith_poly, Small_creek_poly, Sand_Plain_Creek_poly])
    spatial.river_nodes = list(spatial.river_gdf.geometry[0].exterior.coords) 
    
def plot_spatial(spatial, extent = None):    # extent[[x0,x1], [y0,y1]]
    
    fig, ax = plt.subplots(figsize = (7,7))
    ax.set_title('Example spatial files')
       
    x, y = spatial.model_boundary_poly.exterior.xy
    ax.plot(x, y, '-o', ms = 2, lw = 1, color='black')
    x, y = spatial.inner_boundary_poly.exterior.xy
    ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='black')
    if extent: 
        ax.set_xlim(extent[0][0], extent[0][1])
        ax.set_ylim(extent[1][0], extent[1][1])
        
    '''spatial.faults_gdf.plot(ax=ax, markersize = 5, color = 'lightblue', zorder=2)
    for node in spatial.fault_nodes: 
        ax.plot(node[0], node[1], 'o', ms = 3, color = 'lightblue', zorder=2)'''
    
    #spatial.chd_east_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    #spatial.chd_west_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    spatial.obsbore_gdf.plot(ax=ax, markersize = 5, color = 'black', zorder=2)
    spatial.pumpbore_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    #spatial.geobore_gdf.plot(ax=ax, markersize = 12, color = 'green', zorder=2)

    for x, y, label in zip(spatial.obsbore_gdf.geometry.x, spatial.obsbore_gdf.geometry.y, spatial.obsbore_gdf.ID):
        ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, textcoords="offset points")
    for x, y, label in zip(spatial.pumpbore_gdf.geometry.x, spatial.pumpbore_gdf.geometry.y, spatial.pumpbore_gdf.ID):
        ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 10, textcoords="offset points")

    plt.savefig('../figures/spatial.png')

def plot_spatial2(spatial, faults = False, obsbores = False, pumpbores = True, geobores = True, extent = None):    # extent[[x0,x1], [y0,y1]]
    
    fig, ax = plt.subplots(figsize = (7,7))
    ax.set_title('Otorowiri spatial files') 

    x, y = spatial.model_boundary_poly.exterior.xy
    ax.plot(x, y, '-o', ms = 2, lw = 1, color='black')
    x, y = spatial.inner_boundary_poly.exterior.xy
    ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='black')
    if extent: 
        ax.set_xlim(extent[0][0], extent[0][1])
        ax.set_ylim(extent[1][0], extent[1][1])

    spatial.river_gdf.plot(ax=ax, color = 'darkblue', lw = 0.5, zorder=2)
    spatial.outcrop_gdf.plot(ax=ax, color = 'orange', lw = 0.5, zorder=2)
    #spatial.lakes_gdf.plot(ax=ax, color = 'darkblue', zorder=2)
    #spatial.ghb_west_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    #spatial.chd_north_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    #spatial.chd_south_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)

    if faults: 
        spatial.faults_gdf.plot(ax=ax, markersize = 5, color = 'lightblue', zorder=2)
        for node in spatial.faults_nodes: 
            ax.plot(node[0], node[1], 'o', ms = 3, color = 'lightblue', zorder=2)
    
    if obsbores == True:
        spatial.obsbore_gdf.plot(ax=ax, markersize = 5, color = 'black', zorder=2)
        for x, y, label in zip(spatial.obsbore_gdf.geometry.x, spatial.obsbore_gdf.geometry.y, spatial.obsbore_gdf.ID):
            ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, textcoords="offset points")
    
    if pumpbores == True:
        spatial.pumpbore_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
        for x, y, label in zip(spatial.pumpbore_gdf.geometry.x, spatial.pumpbore_gdf.geometry.y, spatial.pumpbore_gdf.ID):
            ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 10, textcoords="offset points")

    if geobores == True:
        spatial.geobore_gdf.plot(ax=ax, markersize = 12, marker = '^', color = 'green', zorder=2)

        for x, y, label in zip(spatial.geobore_gdf.geometry.x, spatial.geobore_gdf.geometry.y, spatial.geobore_gdf.ID):
            ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, textcoords="offset points")