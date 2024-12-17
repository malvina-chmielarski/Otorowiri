import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers

def prepare_strat_column(structuralmodel):
    
    strat = pd.read_excel(structuralmodel.geodata_fname, sheet_name = structuralmodel.strat_sheetname)
    strat_names = strat.unit.tolist()
    lithids = strat.lithid.tolist()
    vals = strat.val.tolist()
    nlg = len(strat_names) - 1 # number of geological layers

    # Make bespoke colormap
    stratcolors = []
    for i in range(len(strat)):
        R = strat.R.loc[i].item() / 255
        G = strat.G.loc[i].item() / 255
        B = strat.B.loc[i].item() / 255
        stratcolors.append([round(R, 2), round(G, 2), round(B, 2)])
    
    import matplotlib.colors
    norm=plt.Normalize(min(lithids),max(lithids))
    tuples = list(zip(map(norm,lithids), stratcolors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    ##########################
    stratigraphic_column = {}
    stratigraphic_column["Ground"] = {}
    stratigraphic_column["Ground"]['Ground'] = {'min': 38, 'max': np.inf, 'id': -1, 'color': stratcolors[0]}
    stratigraphic_column["Quaternary"] = {}
    stratigraphic_column["Tertiary"] = {}
    stratigraphic_column["Coolyena_1a"] = {}
    stratigraphic_column["Coolyena_1b"] = {}
    stratigraphic_column["Coolyena_2"] = {}
    stratigraphic_column["Coolyena_3"] = {}
    stratigraphic_column["Leederville"] = {}
    stratigraphic_column["Warnbro"] = {}
    #stratigraphic_column["Carnac"] = {}
    stratigraphic_column["Parmelia"] = {}
    stratigraphic_column["Yarragadee"] = {}

    tops = [0,5,6,7,8,11,14,17,19,22]
    bots = [5,6,7,10,13,16,18,21,22]
    for i in range(0, len(strat) - 1, 1):
        if i in tops:
            maxval = np.inf
        else:
            maxval = vals[i - 1]
        if i in bots:
             minval = -np.inf
        else:
            minval = vals[i]       
        stratigraphic_column[strat.sequence[i]][strat.lithid[i]] = {
            "min": minval,
            "max": maxval,
            "id": i,
            "color": stratcolors[i],
        }
        
    structuralmodel.strat = strat
    structuralmodel.strat_col = stratigraphic_column
    structuralmodel.strat_names = strat_names
    structuralmodel.cmap = cmap
    structuralmodel.norm = norm
    
def prepare_geodata(structuralmodel, Lleyland = False, Brett = True):

    x0, y0, z0 = structuralmodel.x0, structuralmodel.y0, structuralmodel.z0
    x1, y1, z1 = structuralmodel.x1, structuralmodel.y1, structuralmodel.z1
    strat = structuralmodel.strat
    
    bore_info = pd.read_excel(structuralmodel.geodata_fname, sheet_name=structuralmodel.data_sheetname)
    df = bore_info.copy()

    df = df.loc[(df["Northing"] >= y0)]
    df = df.loc[(df["Northing"] <= y1)]
    df = df.drop(["Source"], axis=1)
    df = df.reset_index(drop=True)

    lithcodes = list(df.columns.values[3:])  # Make a list of formations
    df.Easting = pd.to_numeric(df.Easting)
    df.Northing = pd.to_numeric(df.Northing)
    df.Ground = pd.to_numeric(df.Ground)

    data_list = df.values.tolist()  # Turn data into a list of lists
    formatted_data = []
    for i in range(len(data_list)):  # iterate for each row
        end = False
        # okay, first we will establish the max value (i.e. the end of the hole)
        stuff = []
        for j in range(3, 26, 1):
            if isinstance(data_list[i][j], numbers.Number) == True:
                stuff.append(data_list[i][j])

        EOH = max(stuff)
        #print(EOH)

        boreid = data_list[i][2]
        easting, northing = data_list[i][0], data_list[i][1]
        groundlevel = data_list[i][3]
        # First channp.nan, np.nan, np.nange - we can get the norms from the geophys data...
        gx, gy, gz = 0.0, 0.0, 1.0  # np.nan, np.nan,np.nan

        # Add data for groundlevel
        val = strat.val[0]
        formatted_data.append(
            [
                boreid,
                easting,
                northing,
                groundlevel,
                val,
                "Ground",
                "Ground",
                gx,
                gy,
                gz,
                "raw_data"
            ]
        )  # eventually we cn get this from a dem...
        current_bottom = np.copy(groundlevel)

        if isinstance(data_list[i][4], numbers.Number) == True:
            bottom = groundlevel - float(data_list[i][4])  # Ground surface - TQ (mbgl)
            val = strat.val[1]  # designated isovalue
            lithid = strat.lithid[1]  # lithology id
            feat_name = strat.sequence[1]  # sequence name
            formatted_data.append([boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"])
            current_bottom = np.copy(bottom)
            
        non_conform = [0, #base Quaternary
                       4, #Base Tertiary
                       5, #Base lancelin
                       6, #Base Poison Hill
                       9,#Base Mirrabooka
                       12,#Base Henley
                       15,#Base Mariginiup
                       17,#Base Gage , where absent, make conformal bu adding a point in waneroo
                       18,#Base Carnac
                       20,#Base Ottorowirri
                       21,#Base Yarragadee
                       22]
        #This will just tuck the feature up above the unconformity if it is the first in the strat column and absent"
        top_feat = [1,5,6,7,10,13,16,18,19] 
        non_conform_name = ["NC0","NC1","NC2","NC3","NC4","NC5","NC6","NC7","NC8","NC9","NC10","NC11"]
        #gx, gy, gz = np.nan, np.nan,np.nan
        # I know, I know, there isn't any of the tertiary one here, but in being thorough...
        flags = []
        zdum = []
        for j in range(5, 26, 1):
            if isinstance(data_list[i][j], numbers.Number) == True:
                if data_list[i][j] < EOH:
                    flags.append(1)
                    bottom = groundlevel - float(
                        data_list[i][j])
                    zdum.append(bottom)
                    # Ground surface - TQ (mbgl)
                    val = strat.val[j - 3]  # designated isovalue
                    end = False
                else:
                    bottom = np.copy(current_bottom)  # Ground surface - TQ (mbgl)
                    zdum.append(groundlevel - float(
                        data_list[i][j]))
                    val = strat.val[j - 4]  # designated isovalue
                    end = True
                    flags.append(2)
                lithid = lithcodes[j - 3]  # lithology id
                feat_name = strat.sequence[j - 3]  # sequence name
                formatted_data.append(
                    [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                )
                current_bottom = np.copy(bottom)
            #unconfirmities
                """if strat.lithid[j - 3] in top_feat and EOH == False:
                    val = strat.val[j - 4]
                    bdum = bottom+ 1.
                    feat_name = strat.sequence[j - 3]
                    lithid = lithcodes[j - 3]
                    formatted_data.append(
                        [boreid, easting, northing, bdum, val, lithid, feat_name, gx, gy, gz]
                    )"""         
            else: 
                
                if not end:
                    zdum.append(current_bottom)
                    flags.append(0)
                else:
                    zdum.append(np.nan)
                    flags.append(-1)
                
            if strat.lithid[j - 3] in non_conform:
                if end == False:
                    idx = non_conform.index(strat.lithid[j - 3])
                    bottom = np.copy(current_bottom)
                    val = 0.0
                    lithid = non_conform_name[idx]
                    feat_name = non_conform_name[idx] + '_FEAT' 
                    formatted_data.append(
                        [boreid, easting, northing, bottom, val, lithid, feat_name, 0.0, 0.0, 1.0,"raw_data"]
                    ) 
        if flags[15] == 0 and flags[16] == 0:
            if not np.isnan(zdum[16]):
                val = -1109
                feat_name = strat.sequence[18]
                lithid = lithcodes[18] + '_con'
                #print(feat_name,lithid,zdum[16])
                formatted_data.append(
                        [boreid, easting, northing, zdum[16], val, lithid, feat_name, 0.0, 0.0, 1.0,"force_conform"]
                    )
        if flags[17] == 0 and flags[18] == 0  and flags[19] == 0:
            if not np.isnan(zdum[19]):
                val = 0.
                feat_name = "NC9_FEAT"
                lithid = "yar_dum"   
                formatted_data.append(
                        [boreid, easting, northing, zdum[19], val, lithid, feat_name, 0.0, 0.0, 1.0,"force_conform"]
                    )
                
    data = pd.DataFrame(formatted_data)
    data.columns = [
        "ID",
        "X",
        "Y",
        "Z",
        "val",
        "lithcode",
        "feature_name",
        "gx",
        "gy",
        "gz",
        "for_Kerry"
    ]
    
    if Lleyland:
        valdum = [38.,-10.,0.,0.,-732,-888,-947,-1029,0][::-1]
        feature = ['Ground','Quaternary','NC1_FEAT','NC5_FEAT', #['Quaternary','Quaternary','NC1_FEAT','NC5_FEAT',
                   'Leederville','Leederville','Leederville'
                   ,'Warnbro','NC7_FEAT'][::-1]
        
        ddum = []
        Lley = pd.read_excel("../data/data_dwer/geology.xls", sheet_name="Lleyland")

        for i in range(len(Lley)):
            BoreID = Lley["Name"][i]
            X = Lley["East"][i]
            Y = Lley["North"][i]
            LSE = Lley["Ground"][i]
            ddum.append([BoreID, X, Y, LSE, valdum[-1], 'Lley', feature[-1], 0.0, 0.0, 1.0,"Lucy"])
            ddum.append([BoreID, X, Y, LSE-Lley["Break up"][i], valdum[0], 'Lley', feature[0], 0.0, 0.0, 1.0,"Lucy"]) 
            if type(Lley["SPS"][i]) == type(1):
                ddum.append([BoreID, X, Y, LSE-Lley["SPS"][i], valdum[1], 'Lley', feature[1], 0.0, 0.0, 1.0,"Lucy"])
            if type(Lley["Maringiniup"][i]) == type(1):
                ddum.append([BoreID, X, Y, LSE-Lley["Maringiniup"][i], valdum[2], 'Lley', feature[2], 0.0, 0.0, 1.0,"Lucy"])            
            ddum.append([BoreID, X, Y, LSE-Lley["Waneroo"][i], valdum[3], 'Lley', feature[3], 0.0, 0.0, 1.0,"Lucy"])
            if type(Lley["Pinjar"][i]) == type(1):
                ddum.append([BoreID, X, Y, LSE-Lley["Pinjar"][i], valdum[4], 'Lley', feature[4], 0.0, 0.0, 1.0,"Lucy"])                    
            ddum.append([BoreID, X, Y, LSE-Lley["Aptian"][i], valdum[5], 'Lley', feature[5], 0.0, 0.0, 1.0,"Lucy"])   
            if type(Lley["Paleocene"][i]) == type(1):
                ddum.append([BoreID, X, Y, LSE-Lley["Paleocene"][i], valdum[5], 'Lley', feature[5], 0.0, 0.0, 1.0,"Lucy"]) 
            ddum.append([BoreID, X, Y, LSE-Lley["TQ"][i], valdum[6], 'Lley', feature[6], 0.0, 0.0, 1.0,"Lucy"]) 

        data = pd.concat([data,pd.DataFrame(ddum,columns = data.columns)])
  
    if Brett:
        gphys = pd.read_excel("../data/data_dwer/geology.xls", sheet_name="Other_constraints")
        ddum = []
        for i in range(len(gphys)):
            ddum.append(['GEO', gphys["Easting"][i], gphys["Northing"][i], gphys["z"][i], gphys["val"][i], 'GEO', gphys["Feature"][i], 0.0, 0.0, 1.0,"Brett"]) 

        data = pd.concat([data,pd.DataFrame(ddum,columns = data.columns)]) 
    
    structuralmodel.data = data

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

