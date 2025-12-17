import flopy
import matplotlib.pyplot as plt

def get_watertable(self, geomodel, heads, hdry=-1e30):
    nlay, ncpl = geomodel.cellid_disv.shape
    
    head_disv = -999 * np.ones((geomodel.ncell_disv))  
    watertable = -999 * np.ones(ncpl)
    
    for disucell in range(geomodel.ncell_disu):
        disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
        head_disv[disvcell] = heads[0][disucell]   
    head_disv = head_disv.reshape((nlay, ncpl))
    
    for icpl in range(ncpl):
        for lay in range(nlay): 
            if geomodel.idomain[lay, icpl] == 1:    # if present
                if head_disv[lay, icpl] != hdry:    # if not dry
                    watertable[icpl] = head_disv[lay, icpl] 
                    break           
    
    return watertable

def plot_watertable(self, spatial, mesh, geomodel, flowmodel, watertable, extent = False, xsections = False, **kwargs):
    x0 = kwargs.get('x0', spatial.x0)
    y0 = kwargs.get('y0', spatial.y0)
    x1 = kwargs.get('x1', spatial.x1)
    y1 = kwargs.get('y1', spatial.y1)
        
    fig = plt.figure(figsize = (8,6))
    ax = plt.subplot(111)
    ax.set_title(flowmodel.scenario, size = 10)
    mapview = flopy.plot.PlotMapView(modelgrid=geomodel.vgrid)#, layer = layer)
    plan = mapview.plot_array(watertable, cmap='Spectral', alpha=0.8, **kwargs)#, vmin = vmin, vmax = vmax)
    #if vectors:
    #    mapview.plot_vector(flowmodel.spd["qx"], flowmodel.spd["qy"], alpha=0.5)
    ax.set_xlabel('x (m)', size = 10)
    ax.set_ylabel('y (m)', size = 10)
    plt.colorbar(plan, shrink = 0.4)

    if extent: 
        ax.set_xlim(extent[0][0], extent[0][1])
        ax.set_ylim(extent[1][0], extent[1][1])

    if xsections: 
        for i, xs in enumerate(spatial.xsections):
            x0, y0 = xs[0][0], xs[0][1]
            x1, y1 = xs[1][0], xs[1][1]
            ax.plot([x0,x1],[y0,y1], 'o-', ms = 2, lw = 1, color='black')
            name = spatial.xsection_names[i]
            ax.annotate(name, xy=(x0-1000, y0), xytext=(2, 2), size = 10, textcoords="offset points")
           
    for j in range(spatial.nobs):
        ax.plot(spatial.xyobsbores[j][0], spatial.xyobsbores[j][1],'o', ms = '4', c = 'black')
        #ax.annotate(spatial.idobsbores[j], (spatial.xyobsbores[j][0], spatial.xyobsbores[j][1]+100), c='black', size = 10) #, weight = 'bold')
        
    for j in range(spatial.npump):
        ax.plot(spatial.xypumpbores[j][0], spatial.xypumpbores[j][1],'o', ms = '4', c = 'red')
        #ax.annotate(spatial.idpumpbores[j], (spatial.xypumpbores[j][0], spatial.xypumpbores[j][1]+100), c='red', size = 10) #, weight = 'bold')
            
    #ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color = 'black', lw = 1)
    plt.tight_layout() 
    plt.savefig('../figures/watertable.png')

def plot_plan(self, spatial, mesh, array, layer, xsections = False, extent = None, vmin = None, vmax = None, vectors = None):
        
    fig = plt.figure(figsize = (8,6))
    ax = plt.subplot(111)
    ax.set_title(self.scenario, size = 10)
    mapview = flopy.plot.PlotMapView(model=self.gwf)#, layer = layer)
    plan = mapview.plot_array(getattr(self, array), cmap='Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
    if vectors:
        mapview.plot_vector(self.spd["qx"], self.spd["qy"], alpha=0.5)
    ax.set_xlabel('x (m)', size = 10)
    ax.set_ylabel('y (m)', size = 10)
    plt.colorbar(plan, shrink = 0.4)

    if xsections: 
        for i, xs in enumerate(spatial.xsections):
            x0, y0 = xs[0][0], xs[0][1]
            x1, y1 = xs[1][0], xs[1][1]
            ax.plot([x0,x1],[y0,y1], 'o-', ms = 2, lw = 1, color='black')
            name = spatial.xsection_names[i]
            ax.annotate(name, xy=(x0-1000, y0), xytext=(2, 2), size = 10, textcoords="offset points")
           
    for j in range(spatial.nobs):
        ax.plot(spatial.xyobsbores[j][0], spatial.xyobsbores[j][1],'o', ms = '4', c = 'black')
        #ax.annotate(spatial.idobsbores[j], (spatial.xyobsbores[j][0], spatial.xyobsbores[j][1]+100), c='black', size = 10) #, weight = 'bold')
        
    for j in range(spatial.npump):
        ax.plot(spatial.xypumpbores[j][0], spatial.xypumpbores[j][1],'o', ms = '4', c = 'red')
        #ax.annotate(spatial.idpumpbores[j], (spatial.xypumpbores[j][0], spatial.xypumpbores[j][1]+100), c='red', size = 10) #, weight = 'bold')
            
    if mesh.plangrid == 'car': mesh.sg.plot(ax = ax, color = 'black', lw = 0.2) 
    if mesh.plangrid == 'tri': mesh.tri.plot(ax = ax, edgecolor='black', lw = 0.2)
    if mesh.plangrid == 'vor': mesh.vor.plot(ax = ax, edgecolor='black', lw = 0.2)
    ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color = 'black', lw = 1)
    plt.tight_layout() 
    plt.savefig('../figures/plan_%s.png' % array)

def plot_transect(self, spatial, structuralmodel, array, title = False,
                    vmin = None, vmax = None, vectors = None,
                    **kwargs): # array needs to be a string of a property eg. 'k11', 'angle2'
    x0 = kwargs.get('x0', spatial.x0)
    y0 = kwargs.get('y0', spatial.y0)
    z0 = kwargs.get('z0', structuralmodel.z0)
    x1 = kwargs.get('x1', spatial.x1)
    y1 = kwargs.get('y1', spatial.y1)
    z1 = kwargs.get('z1', structuralmodel.z1)
    
    fig = plt.figure(figsize = (8,3))
    ax = plt.subplot(111)
    if title:
        ax.set_title(title, size = 10)
    else:   
        ax.set_title(self.scenario, size = 10)
      
    xsect = flopy.plot.PlotCrossSection(model=self.gwf, line={"line": [(x0, y0),(x1, y1)]}, 
                                        #extent = [P.x0,P.x1,P.z0,P.z1], 
                                        geographic_coords=True)
    xsect.plot_grid(lw = 0.5, color = 'black') 
    csa = xsect.plot_array(a = getattr(self, array), cmap = 'Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
    #if vectors:
    #    #import flopy
    #    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(self.spd, self.gwf)
    #  
    #    xsect.plot_vector(qx, qy, qz, scale=1., normalize=True, color="black")
        
    ax.set_xlabel('x (m)', size = 10)
    ax.set_ylabel('z (m)', size = 10)
    ax.set_ylim([z0,z1])
    plt.colorbar(csa, shrink = 0.4)

    plt.tight_layout()  
    plt.savefig('../figures/transect_%s.png' % array)
    plt.show()    