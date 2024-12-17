import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
     
def prepare_geodata(structuralmodel): 

    x0, y0, z0 = structuralmodel.x0, structuralmodel.y0, structuralmodel.z0
    x1, y1, z1 = structuralmodel.x1, structuralmodel.y1, structuralmodel.z1

    df = pd.read_excel(structuralmodel.geodata_fname, sheet_name = structuralmodel.data_sheetname)#sheet_name = 'bore_info', skiprows = 1)

    #origin = np.array([spatial.x0, spatial.y0, spatial.z0]).astype(float)
    #maximum = np.array([spatial.x1, spatial.y1, spatial.z1]).astype(float)
    
    sys.path.append('../../lab_tools/spatial')  
    
    from spatial import Ascii
    LSE_east = Ascii('../data/data_dem/Hydro_Enforced_1_Second_DEM.asc')
    LSE_west = Ascii('../data/data_dem/Hydro_Enforced_1_Second_DEM_1.asc')  

    diff = 703000.
    for i in range(len(df)):
        if np.isnan(df.loc[i, 'Z']):
            if df.loc[i, 'X'] > diff:
                df.loc[i, 'Z']= LSE_east.val_get(df.loc[i, 'X'], df.loc[i, 'Y']) - df.loc[i, 'Depth']
            else:
                df.loc[i, 'Z'] = LSE_west.val_get(df.loc[i, 'X'], df.loc[i, 'Y'])- df.loc[i, 'Depth']
    
    #add random points for land surface
    n = 500
    for i in range(n):
        xx = x0 + (x1 - x0) * np.random.rand()
        yy = y0 + (y1 - y0) * np.random.rand()
        if xx> diff:
            zz = LSE_east.val_get(xx,yy)
        else:
            zz = LSE_west.val_get(xx,yy)
        df_new_row = pd.DataFrame.from_records(
                {'X': xx,
                'Y': yy,
                'Z' : zz,
                'name' : ['Ground'],
                'val': [0.],
                'feature_name' : ['Ground'],
                'gx' : [0.],
                'gy' : [0.],
                'gz' : [1.]})
        df = pd.concat([df, df_new_row], ignore_index=True)
    df = df.loc[(df["Z"] >-9999.)] 
    structuralmodel.data = df

def prepare_strat_column(structuralmodel):
    strat = pd.read_excel(structuralmodel.geodata_fname, sheet_name = structuralmodel.strat_sheetname)
    strat_names = strat.unit.tolist()
    
    # Make bespoke colormap
    stratcolors = []
    for i in range(len(strat)):
        R = strat.R.loc[i].item() / 255
        G = strat.G.loc[i].item() / 255
        B = strat.B.loc[i].item() / 255
        stratcolors.append([round(R, 2), round(G, 2), round(B, 2)])
    
    import matplotlib.colors
    cvals  = [-1,0,1,2,3,4,5,6]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), stratcolors))
    print(norm)
    print(cvals)
    print(stratcolors)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    
    stratigraphic_column = {}
    stratigraphic_column["Ground"] = {}
    stratigraphic_column["Ground"]['Ground'] = {
                                                        'min': 0,
                                                        'max': np.inf,
                                                        'id': -1,
                                                        'color': stratcolors[0]}
    
    
    
    """stratigraphic_column["QT"] = {}
    stratigraphic_column["QT"]['sed'] = {
                                                        'min': -np.inf,
                                                        'max': np.inf,
                                                        'id': 0,
                                                        'color': [round(153./256.,2),
                                                                 round(204./256.,2),
                                                                 round(255./256.,2)]}
    UC2 = geomodel.add_unconformity(geomodel["QT"], -7)
    """
    stratigraphic_column["Sequence"] = {}
    stratigraphic_column["Sequence"]['Weeli_Wolli'] = {
                                                        'min': -200,
                                                        'max': np.inf,
                                                        'id': 0,
                                                        'color': stratcolors[1]}
    
    stratigraphic_column["Sequence"]['Whaleback'] =         {
                                                        'min': -240,
                                                        'max': -200,
                                                        'id': 1,
                                                        'color': stratcolors[2]}
    
    stratigraphic_column["Sequence"]['Dales'] =         {
                                                        'min': -350,
                                                        'max': -240,
                                                        'id': 2,
                                                        'color': stratcolors[3]}
    
    stratigraphic_column["Sequence"]['McRae'] =         {
                                                        'min': -470,
                                                        'max': -350,
                                                        'id': 3,
                                                        'color': stratcolors[4]}
    
    stratigraphic_column["Sequence"]['Wittenoom'] =    {
                                                        'min': -620,
                                                        'max': -470.,
                                                        'id': 4,
                                                        'color': stratcolors[5]}
    
    stratigraphic_column["Sequence"]['Marra_Mamba'] = {
                                                        'min': -770.,
                                                        'max': -620.,
                                                        'id': 5,
                                                        'color': stratcolors[6]}
    
    stratigraphic_column["Sequence"]['Fortescue'] =    {
                                                        'min': -np.inf,
                                                        'max': -770,
                                                        'id': 6,
                                                        'color': stratcolors[7]}  
    
    structuralmodel.strat = strat
    structuralmodel.strat_col = stratigraphic_column
    structuralmodel.strat_names = strat_names
    structuralmodel.cmap = cmap
    structuralmodel.norm = norm

def create_structuralmodel(structuralmodel):

    from LoopStructural import GeologicalModel
    model = GeologicalModel(structuralmodel.origin, structuralmodel.maximum)
    model.data = structuralmodel.data
    
    Ground = model.create_and_add_foliation("Ground", nelements=1e4, buffer=0.1)
    UC1 = model.add_unconformity(model["Ground"], 0)
    Top = model.create_and_add_foliation("QT", nelements=1e4, buffer=0.1)   
    Basement = model.create_and_add_foliation("Sequence", nelements=5e4, buffer=0.1    )
    
    model.set_stratigraphic_column(structuralmodel.strat_col)
    
    structuralmodel.model = model

