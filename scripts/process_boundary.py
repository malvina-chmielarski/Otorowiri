from shapely.geometry import LineString, LinearRing, Point,Polygon, MultiPolygon, MultiPoint, mapping
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from loopflopy.mesh_routines import resample_linestring
from shapely.ops import split, polygonize

def gather_boundary_linestrings(project):
    fname = '../Data/Data_shp/Fault_eastern_boundary.shp'
    gdf = gpd.read_file(fname)
    gdf.to_crs(epsg=project.crs, inplace=True)
    ls = gdf.geometry[0]
    line1 = LineString(resample_linestring(ls, 1000))  

    #fname = '../Data/Data_shp/Groundwater_divide.shp'
    #gdf = gpd.read_file(fname)
    #gdf.to_crs(epsg=project.crs, inplace=True)
    #ls = gdf.geometry[0]
    #line2 = LineString(resample_linestring(ls, 1000))  
    ######### CHEATING!!!! ###################
    coords = [(350000, 6680000), (388000, 6690000)]
    ls = LineString(coords)
    line2 = LineString(resample_linestring(ls, 1000))  
    ##########################################

    fname = '../modelfiles/surface_contours.shp'
    gdf = gpd.read_file(fname)
    gdf.to_crs(epsg=project.crs, inplace=True)
    ls = gdf.geometry[0]
    line3 = LineString(resample_linestring(ls, 1000))   

    ######### CHEATING!!!! ###################
    coords = [(343000, 6767000), (360000, 6764000)]
    ls = LineString(coords)
    line4 = LineString(resample_linestring(ls, 1000))    
    ##########################################

    raw_lines = [line1, line2, line3, line4] 
    labels = ['line1', 'line2', 'line3', 'line4']
    return raw_lines, labels

def plot_linestrings(lines, labels):
    fig, ax = plt.subplots() 
    for line, label in zip(lines, labels):
        x, y = line.xy
        ax.plot(x, y, 'o', ms = 1, label = label)  # You can set color, linestyle, etc.

    ax.set_aspect('equal')
    ax.legend()
    plt.show()

def trim_boundary_linestrings(raw_lines):
    
    line1, line2, line3, line4 = raw_lines[0], raw_lines[1], raw_lines[2], raw_lines[3]

    trimmed = split(line3, line2) # line to be trimmer, cutter line
    line3 = trimmed.geoms[0]

    trimmed = split(line3, line4) # line to be trimmer, cutter line
    line3 = trimmed.geoms[1]

    trimmed = split(line1, line4) # line to be trimmer, cutter line
    line1 = trimmed.geoms[1]

    trimmed = split(line1, line2) # line to be trimmer, cutter line
    line1 = trimmed.geoms[0]

    trimmed = split(line4, line3) # line to be trimmer, cutter line
    line4 = trimmed.geoms[1]

    trimmed = split(line4, line1) # line to be trimmer, cutter line
    line4 = trimmed.geoms[0]

    trimmed = split(line2, line3) # line to be trimmer, cutter line
    line2 = trimmed.geoms[1]

    trimmed = split(line2, line1) # line to be trimmer, cutter line
    line2 = trimmed.geoms[0]

    trimmed_lines = [line1, line2, line3, line4]        
    return  trimmed_lines

def snap_linestring_endpoints(lines, tolerance=1e-6):
    # Collect all endpoints
    endpoints = []
    for line in lines:
        coords = list(line.coords)
        endpoints.append(coords[0])
        endpoints.append(coords[-1])
    endpoints = np.array(endpoints)

    # Snap endpoints that are within tolerance
    snapped_points = {}
    for i, pt1 in enumerate(endpoints):
        for j, pt2 in enumerate(endpoints):
            if i >= j:
                continue
            if np.linalg.norm(np.array(pt1) - np.array(pt2)) < tolerance:
                snapped_points[tuple(pt2)] = tuple(pt1)

    # Replace endpoints in lines
    new_lines = []
    for line in lines:
        coords = list(line.coords)
        start = tuple(coords[0])
        end = tuple(coords[-1])
        if start in snapped_points:
            coords[0] = snapped_points[start]
        if end in snapped_points:
            coords[-1] = snapped_points[end]
        new_lines.append(LineString(coords))
    return new_lines

def make_boundary_polygon(snapped_lines):
    
    polygons = list(polygonize(snapped_lines))
    if len(polygons) == 0:
        raise ValueError("No polygons could be created from the linestrings.")
    elif len(polygons) > 1:
        print(f"Warning: More than one polygon created, using the first one.")

    poly = polygons[0]
    gdf = gpd.GeoDataFrame(geometry=[poly], crs='EPSG:28350')
    gdf.to_file('../modelfiles/model_boundary_polygon.shp')

    return poly
