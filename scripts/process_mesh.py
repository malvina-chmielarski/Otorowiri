import numpy as np
import matplotlib.pyplot as plt
import flopy
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,MultiPoint,shape

def prepare_nodes(mesh, spatial):
    
    nodes = []  
    
    # DYKE REFINEMENT POINTS
    for point in spatial.faults_nodes: nodes.append(point)
    #for point in dyke1_nodes: nodes.append(point)
    #for point in dyke2_nodes: nodes.append(point)
    nodes = np.array(nodes)
    mesh.nodes = nodes

def prepare_polygons(mesh, spatial):

    polygons = [] # POLYGONS[(polygon, (x,y), maxtri)]
    
    # MODEL BOUNDARY - Use shapely object for boundary
    polygons.append((list(spatial.model_boundary_poly.exterior.coords), 
                     (spatial.model_boundary_poly.representative_point().x, 
                      spatial.model_boundary_poly.representative_point().y), 
                      mesh.modelmaxtri)) 
    polygons.append((list(spatial.inner_boundary_poly.exterior.coords), 
                     (spatial.inner_boundary_poly.representative_point().x, 
                      spatial.inner_boundary_poly.representative_point().y), 
                      mesh.modelmaxtri)) 
    
    # STREAMS
    #polygons.append((list(spatial.streams_poly.exterior.coords), 
    #                 (spatial.streams_poly.representative_point().x, 
    #                  spatial.streams_poly.representative_point().y), 
    #                  mesh.modelmaxtri)) # Refinement
    
    # DYKES - ADD THIS - JUSTTAKING OFF AS ADDS LOTS OF CELLS!
    #for poly in dykes_multipoly.geoms:
    #    polygons.append((list(poly.exterior.coords), 
    #                 (poly.representative_point().x, 
    #                  poly.representative_point().y), 
    #                  mesh.modelmaxtri)) # Refinement

    mesh.polygons = polygons

def locate_special_cells(mesh, spatial):

    #-------- OBS ------------
    obs_cells = []
    
    points = [Point(xy) for xy in spatial.xyobsbores]
    for point in points:
        cell = mesh.gi.intersect(point)["cellids"][0]
        mesh.ibd[cell] = 1
        obs_cells.append(cell)
    
    # ----------WEL ---------------
    wel_cells = []
    
    points = [Point(xy) for xy in spatial.xypumpbores]
    for point in points:
        cell = mesh.gi.intersect(point)["cellids"][0]
        mesh.ibd[cell] = 2
        wel_cells.append(cell)
    
    #-------- STREAM  ------------
    #stream_cells = []
    #for i in mesh.xcyc:
    #    point = Point((i[0], i[1]))
    #    cell = mesh.gi.intersect(point)["cellids"]
    #    if spatial.streams_poly.contains(point):
    #        mesh.ibd[cell[0]] = 3
    #        stream_cells.append(cell[0])
    
    #-------- CHD ------------
    
    chd_west_cells = []
    chd_west_cells = mesh.gi.intersects(spatial.chd_west_ls)["cellids"]
    print(chd_west_cells) 
    mesh.ibd[cell] = 3
    
    mesh.obs_cells = obs_cells
    mesh.wel_cells = wel_cells
    mesh.chd_west_cells = chd_west_cells

def plot_feature_cells(mesh, spatial):
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mesh.vgrid)
    p = pmv.plot_array(mesh.ibd, alpha = 0.6)
    mesh.tri.plot(ax=ax, edgecolor='black', lw = 0.1)
         
    #ax.set_xlim([700000, 707500]) 
    #ax.set_ylim([7475000, 7485000]) 

    # Add on polygons to plot
    #x, y = spatial.streams_poly.exterior.xy
    #ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='blue') 
    #for poly in dykes_multipoly.geoms:
    #    x, y = poly.exterior.xy
    #    ax.plot(x, y, '-o', ms = 0.5, lw = 0.5, color='red') 
    
    spatial.obsbore_gdf.plot(ax=ax, markersize = 7, color = 'darkblue', zorder=2)
    spatial.pumpbore_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    
    for cell in mesh.chd_west_cells:
        ax.plot(mesh.cell2d[cell][1], mesh.cell2d[cell][2], "o", color = 'red', ms = 1)
    for cell in mesh.obs_cells:
        ax.plot(mesh.cell2d[cell][1], mesh.cell2d[cell][2], "o", color = 'black', ms = 1)
    for cell in mesh.wel_cells:
        ax.plot(mesh.cell2d[cell][1], mesh.cell2d[cell][2], "o", color = 'blue', ms = 2)
    
