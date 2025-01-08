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
    sequences = strat.sequence.tolist()
    sequence = set(sequences)
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
    for i in range(len(sequence)):
        stratigraphic_column[sequence[i]] = {}
    for i in range(len(sequences)):
        if i == 0 or sequences[i] != sequences[i-1]:
            mx = np.inf
        else:
            mx = vals[i-1]
        if i == (len(sequences) - 1) or sequences[i] != sequences[i+1]:
            mn = -np.inf
        else:
            mn = vals[i]
        if i == 0: mn = vals[i] #work around for the ground
        stratagraphic_column[sequences[i]][strat_names[i]] = {'min': mn, 'max': mx, 'id': lithids[i], 'color': stratcolors[i]}
        
           
    structuralmodel.strat = strat
    structuralmodel.strat_col = stratigraphic_column
    structuralmodel.strat_names = strat_names
    structuralmodel.cmap = cmap
    structuralmodel.norm = norm
    
def prepare_geodata(structuralmodel, Lleyland = False, Brett = True, Petroleum = True, Model = 1):

    x0, y0, z0 = structuralmodel.x0, structuralmodel.y0, structuralmodel.z0
    x1, y1, z1 = structuralmodel.x1, structuralmodel.y1, structuralmodel.z1
    strat = structuralmodel.strat
    
    bore_info = pd.read_excel(structuralmodel.geodata_fname, sheet_name=structuralmodel.data_sheetname)
    df = bore_info.copy()
    df = df.loc[(df["Northing"] >= y0)]
    df = df.loc[(df["Northing"] <= y1)]
    df = df.loc[(df["Easting"] >= x0)]
    df = df.loc[(df["Easting"] <= x1)]
    if Model == 1:
        df = df.loc[(df["A_model"] == 1)]
    else:
        df = df.loc[(df["B_model"] == 1)]
    df = df.reset_index(drop=True)

    df.Easting = pd.to_numeric(df.Easting)
    df.Northing = pd.to_numeric(df.Northing)
    df.Ground = pd.to_numeric(df.Ground)

    data_list = df.values.tolist()  # Turn data into a list of lists
    formatted_data = []
   
    for i in range(len(data_list)):  # iterate for each row
        end = False
        # okay, first we will establish the max value (i.e. the end of the hole)
        stuff = []
        for j in range(5,27):
            if isinstance(data_list[i][j], numbers.Number) == True:
                stuff.append(data_list[i][j])

        EOH = max(stuff)
        #print(EOH)

        boreid = data_list[i][3]
        easting, northing = data_list[i][0], data_list[i][1]
        groundlevel = data_list[i][4]
        # First channp.nan, np.nan, np.nange - we can get the norms from the geophys data...
        gx, gy, gz = 0.0, 0.0, 1.0  # np.nan, np.nan,np.nan

        # Add data for groundlevel
        val = strat.vals[0]
        formatted_data.append(
            [
                boreid,
                easting,
                northing,
                groundlevel,
                val,
                "Ground",
                "Quaternary",
                gx,
                gy,
                gz,
                "raw_data"
            ]
        )  # eventually we cn get this from a dem...
        current_bottom = np.copy(groundlevel)
        
        #Quaternary Unconformity
        if isinstance(data_list[i][5], numbers.Number) == True:
            bottom = groundlevel - float(data_list[i][5])  # Ground surface - TQ (mbgl)
            val = strat.vals[1]  # designated isovalue
            lithid = lithcodes[1]  # lithology id
            feat_name = strat.sequences[1]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )
            current_bottom = np.copy(bottom)
        #Kings park
        if isinstance(data_list[i][8], numbers.Number) == True:
            bottom = groundlevel - float(data_list[i][8])  # Ground surface - kinhs park base (mbgl)
            val = strat.vals[2]  # designated isovalue
            lithid = lithcodes[2]  # lithology id
            feat_name = strat.sequences[2]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )
            current_bottom = np.copy(bottom)  
        #Tertiaty_unconformity
        bottom = np.copy(current_bottom)  # current depth
        val = 0  # designated isovalue
        lithid = 'base_tert'  # lithology id
        feat_name = "Tert_U"  # sequence name
        formatted_data.append(
            [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
        )
        #Lancelin formation
        ldum = [3,4,5]
        cdum = [11,12,13]
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                val = strat.vals[ldum[j]]  # designated isovalue
                lithid = lithcodes[ldum[j]]  # lithology id
                feat_name = strat.sequences[ldum[j]]  # sequence name
                formatted_data.append(
                    [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                )
                current_bottom = np.copy(bottom)              

        #Lancelin _ unconformity
        bottom = np.copy(current_bottom)  # current depth
        val = 0  # designated isovalue
        lithid = 'base_lanc'  # lithology id
        feat_name = "Lanc_U"  # sequence name
        formatted_data.append(
            [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
        )

        #Osbourne_formation
        ldum = [6,7,8]
        cdum = [14,15,17]
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                val = strat.vals[ldum[j]]  # designated isovalue
                lithid = lithcodes[ldum[j]]  # lithology id
                feat_name = strat.sequences[ldum[j]]  # sequence name
                formatted_data.append(
                    [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                )
                current_bottom = np.copy(bottom)     

        #Osbourne _ unconformity
        bottom = np.copy(current_bottom)  # current depth
        val = 0  # designated isovalue
        lithid = 'base_Osb'  # lithology id
        feat_name = "Osb_U"  # sequence name
        formatted_data.append(
            [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
        )

        #Leedervilles
        ldum = [9,10,11]
        cdum = [18,19,20]
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                if float(data_list[i][cdum[j]]) < EOH:
                    bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                    val = strat.vals[ldum[j]]  # designated isovalue
                    lithid = lithcodes[ldum[j]]  # lithology id
                    feat_name = strat.sequences[ldum[j]]  # sequence name
                    formatted_data.append(
                        [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                    )
                    current_bottom = np.copy(bottom)         
        #Leederville _ unconformity
        bottom = np.copy(current_bottom)  # current depth
        val = 0  # designated isovalue
        lithid = 'base_Leed'  # lithology id
        feat_name = "Leed_U"  # sequence name
        formatted_data.append(
            [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
        )

        #SPS and Gage
        ldum = [12,13]
        cdum = [21,22]
        for j in range(2):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                if float(data_list[i][cdum[j]]) < EOH:
                    bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                    val = strat.vals[ldum[j]]  # designated isovalue
                    lithid = lithcodes[ldum[j]]  # lithology id
                    feat_name = strat.sequences[ldum[j]]  # sequence name
                    formatted_data.append(
                        [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                    )
                    current_bottom = np.copy(bottom)          
        #Warnbro _ unconformity
        bottom = np.copy(current_bottom)  # current depth
        val = 0  # designated isovalue
        lithid = 'base_Warn'  # lithology id
        feat_name = "Warn_U"  # sequence name
        formatted_data.append(
            [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
        )

        #Yaragadee, or parmelias, no real yaragadee bases
        ldum = [14,15,16]
        cdum = [23,24,25]
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                if float(data_list[i][cdum[j]]) < EOH:
                    bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                    val = strat.vals[ldum[j]]  # designated isovalue
                    lithid = lithcodes[ldum[j]]  # lithology id
                    feat_name = strat.sequences[ldum[j]]  # sequence name
                    formatted_data.append(
                        [boreid, easting, northing, bottom, val, lithid, feat_name, np.nan, np.nan, np.nan,"raw_data"]
                    )
                    current_bottom = np.copy(bottom)
    
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

    if Petroleum:
        Pet = pd.read_excel(
        "../data/data_dwer\geology.xls", sheet_name="Petroleum_readable")
        ddum = []
        for i in range(len(Pet)):
            ddum.append(['PET',Pet["X"][i], Pet["Y"][i], Pet["Z"][i], Pet["val"][i],Pet['lithcode'][i], Pet["feature_name"][i], np.nan, np.nan, np.nan,"WAPIMS"]) 

        data = pd.concat([data,pd.DataFrame(ddum,columns = data.columns)])  
    
    structuralmodel.data = data

def create_structuralmodel(structuralmodel):

    from LoopStructural import GeologicalModel
    model = GeologicalModel(structuralmodel.origin, structuralmodel.maximum)
    model.data = structuralmodel.data
    
        #add foliations

    Ground = model.create_and_add_foliation(
        "Ground", nelements=1e4, buffer=0.1
    )
    UC = model.add_onlap_unconformity(
    model["Ground"], 0.
    )
    #First add the quaternary formation
    Quat = model.create_and_add_foliation(
        "Quaternary", nelements=1e4, buffer=0.1
    )
    UC0 = model.add_onlap_unconformity(
    model["Quaternary"], -44.
    )
    Tert = model.create_and_add_foliation(
        "Tertiary", nelements=1e4, buffer=0.1
    )
    TertUC =model.create_and_add_foliation(
        "Tert_U", nelements=1e4, buffer=0.1
    )
    UC1 = model.add_onlap_unconformity(
        model["Tert_U"], 0.
    )
    Lanc = model.create_and_add_foliation(
        "Lancelin", nelements=1e4, buffer=0.1
    )
    Lancelin_UC = model.create_and_add_foliation(
        "Lanc_U", nelements=1e4, buffer=0.1
    )
    UC2 = smodel.add_onlap_unconformity(
        model["Lanc_U"], 0
    )
    Osb = model.create_and_add_foliation(
        "Osbourne", nelements=1e4, buffer=0.1
    )

    Osbourne_UC = model.create_and_add_foliation(
        "Osb_U", nelements=1e4, buffer=0.1
    )
    UC3 = model.add_onlap_unconformity(
        model["Osb_U"], 0
    )
   
    LEED = model.create_and_add_foliation(
        "Leederville", nelements=1e4, buffer=0.1
    )
    LEED_UC = model.create_and_add_foliation(
        "Leed_U", nelements=1e4, buffer=0.1
    )
    UC4 = model.add_onlap_unconformity(
        model["Leed_U"], 0
    )
    #SPS and/ or Gage can go missing 
    WARN = model.create_and_add_foliation(
        "Warnbro", nelements=1e4, buffer=0.1
    )
    
    WARN_UC = model.create_and_add_foliation(
        "Warn_U", nelements=1e4, buffer=0.1
    )
    UC5 = model.add_onlap_unconformity(
        model["Warn_U"], 0
    )

    YARR = model.create_and_add_foliation(
        "Yarragadee", nelements=1e4, buffer=0.1
    )   

    model.set_stratigraphic_column(structuralmodel.strat_col)
    
    structuralmodel.model = model

