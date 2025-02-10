import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
from LoopStructural.utils import strikedip2vector as strike_dip_vector

def prepare_strat_column(structuralmodel):
    
    strat = pd.read_excel(structuralmodel.geodata_fname, sheet_name = structuralmodel.strat_sheetname)
    strat_names = strat.unit.tolist()
    lithids = strat.lithid.tolist()
    vals = strat.val.tolist()
    nlg = len(strat_names) - 1 # number of geological layers
    sequences = strat.sequence.tolist()
    sequence = list(set(sequences))
    
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
        if i != (len(sequences) - 1) and sequences[i] != sequences[i+1] and i != 1:
            mn = -np.inf
        else:
            mn = vals[i]
        if i == 0: mn = vals[i] #work around for the ground
        stratigraphic_column[sequences[i]][strat_names[i]] = {'min': mn, 'max': mx, 'id': lithids[i], 'color': stratcolors[i]}
    ###########################    
           
    structuralmodel.strat = strat
    structuralmodel.strat_col = stratigraphic_column
    structuralmodel.strat_names = strat_names
    structuralmodel.cmap = cmap
    structuralmodel.norm = norm
    
def prepare_geodata(structuralmodel, 
                    Lleyland = True, 
                    Brett = True, 
                    Petroleum = True, 
                    Model = 1,
                    extent = None,
                    Fault = True):

    if type(extent) == type(None):
        x0, y0, z0 = structuralmodel.x0, structuralmodel.y0, structuralmodel.z0
        x1, y1, z1 = structuralmodel.x1, structuralmodel.y1, structuralmodel.z1
    else:
        x0, y0, z0 = extent[0], extent[1], extent[2]
        x1, y1, z1 = extent[3], extent[4], extent[5]      

    
    strat = structuralmodel.strat
    
    elev = np.loadtxt('../data/LSE.dat')

    
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
    gx, gy, gz = 0.0, 0.0, 1.0
    for i in range(len(elev)):
        val = strat.val[0]
        formatted_data.append(
            [
                "ground_cloud",
                elev[i,0],
                elev[i,1],
                elev[i,2],
                val,
                "Ground",
                "Ground",
                gx,
                gy,
                gz,
                "ELVIS_data"
            ]
        )
        
   
    for i in range(len(data_list)):  # iterate for each row
        end = False
        # okay, first we will establish the max value (i.e. the end of the hole)
        stuff = []
        for j in range(5,27):
            if isinstance(data_list[i][j], numbers.Number) == True:
                stuff.append(data_list[i][j])

        EOH = max(stuff)
        #print(EOH)
        EOH_flag = False
        boreid = data_list[i][3]
        easting, northing = data_list[i][0], data_list[i][1]
        groundlevel = data_list[i][4]
        bottom = np.copy(groundlevel)
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
        
        #Quaternary Unconformity
        if isinstance(data_list[i][5], numbers.Number) == True:
            bottom = groundlevel - float(data_list[i][5])  # Ground surface - TQ (mbgl)
            val = strat.val[1]  # designated isovalue
            lithid = strat.lithid[1]  # lithology id
            feat_name = strat.sequence[1]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )
            current_bottom = np.copy(bottom)
        bottom = np.copy(current_bottom) # current depth
        val = 0  # designated isovalue
        lithid = 'base_Quat'  # lithology id
        feat_name = "Quat_U"  # sequence name
        formatted_data.append(
            [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
        )    
        
        #Kings park
        TKP = np.copy(bottom)
        if isinstance(data_list[i][8], numbers.Number) == True:
            bottom = groundlevel - float(data_list[i][8])  # Ground surface - kinhs park base (mbgl)
            val = strat.val[2]  # designated isovalue
            lithid = strat.lithid[2]  # lithology id
            feat_name = strat.sequence[2]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )
            current_bottom = np.copy(bottom)  
        if TKP != bottom: #add top of kings park formation:
            val = strat.val[1]
            lithid = strat.lithid[2]  # lithology id
            feat_name = strat.sequence[2]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, TKP, val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )     
        else: #hide it    
            val = strat.val[1]
            lithid = strat.lithid[2]  # lithology id
            feat_name = strat.sequence[2]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom+2., val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )     
        #Tertiaty_unconformity
        gx, gy, gz = np.nan, np.nan,np.nan
        if not EOH_flag:
            bottom = np.copy(current_bottom) # current depth
            val = 0  # designated isovalue
            lithid = 'base_tert'  # lithology id
            feat_name = "Tert_U"  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
            )
        #Lancelin formation
        ldum = [3,4,5]
        cdum = [11,12,13]
        TLF = np.copy(bottom)
        first = None
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                val = strat.val[ldum[j]]  # designated isovalue
                lithid = strat.lithid[ldum[j]]  # lithology id
                feat_name = strat.sequence[ldum[j]]  # sequence name
                formatted_data.append(
                    [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                )
                current_bottom = np.copy(bottom)     
                if type(first) == type(None):
                    first = strat.val[ldum[j]-1]
        
        if bottom == TLF: #hide it
            val = strat.val[5]
            lithid = strat.lithid[5]  # lithology id
            feat_name = strat.sequence[5]  # sequence name
            if isinstance(data_list[i][10], numbers.Number) == True: #is it logged as undifferentiated?
                bottom = groundlevel - float(data_list[i][10])
                current_bottom = np.copy(bottom)               
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )                      
        else:    
            val = strat.val[2]
            lithid = strat.lithid[3]  # lithology id
            feat_name = strat.sequence[3]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, TLF, first, lithid, feat_name, gx, gy, gz,"raw_data"]
            )    

        #Lancelin _ unconformity
        bottom = np.copy(current_bottom)  # current depth
        if not EOH_flag:
            val = 0  # designated isovalue
            lithid = 'base_lanc'  # lithology id
            feat_name = "Lanc_U"  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
            )
            
        TOF = np.copy(bottom)
        #Osbourne_formation
        ldum = [6,7,8]
        cdum = [14,15,17]
        first = None
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                val = strat.val[ldum[j]]  # designated isovalue
                lithid = strat.lithid[ldum[j]]  # lithology id
                feat_name = strat.sequence[ldum[j]]  # sequence name
                formatted_data.append(
                    [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                )
                current_bottom = np.copy(bottom)   
                if type(first) == type(None):
                    first = strat.val[ldum[j]-1]  
                    lithdum = lithid
        if TOF == bottom:
            val = strat.val[8]
            lithid = strat.lithid[8]  # lithology id
            feat_name = strat.sequence[8]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom+2., val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )                      
        else:    
            val = strat.val[5]
            lithid = strat.lithid[6]  # lithology id
            feat_name = strat.sequence[6]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, TOF, first, lithid, feat_name, gx, gy, gz,"raw_data"]
            ) 
            
        #Osbourne _ unconformity
        bottom = np.copy(current_bottom)  # current depth

        if not EOH_flag:
            val = 0  # designated isovalue
            lithid = 'base_Osb'  # lithology id
            feat_name = "Osb_U"  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
            )
        TOL = np.copy(bottom)
        first = None
        #Leedervilles
        ldum = [9,10,11]
        cdum = [18,19,20]
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                if float(data_list[i][cdum[j]]) < EOH:
                    bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                    val = strat.val[ldum[j]]  # designated isovalue
                    lithid = strat.lithid[ldum[j]]  # lithology id
                    feat_name = strat.sequence[ldum[j]]  # sequence name
                    formatted_data.append(
                        [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                    )
                    current_bottom = np.copy(bottom)   
                else:
                    EOH_flag = True      
                if type(first) == type(None):
                    first = strat.val[ldum[j]-1]
        #Leederville _ unconformity
        if TOL == bottom and not EOH_flag:
            val = strat.val[11]
            lithid = strat.lithid[11]  # lithology id
            feat_name = strat.sequence[11]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom+2., val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )                      
        else:    
            val = strat.val[8]
            lithid = strat.lithid[9]  # lithology id
            feat_name = strat.sequence[9]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, TOL, first, lithid, feat_name, gx, gy, gz,"raw_data"]
            ) 
        bottom = np.copy(current_bottom)  # current depth
        if not EOH_flag:
            val = 0  # designated isovalue
            lithid = 'base_Leed'  # lithology id
            feat_name = "Leed_U"  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
            )

        TOW = np.copy(bottom)
        #SPS and Gage
        ldum = [12,13]
        cdum = [21,22]
        first = None
        for j in range(2):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                if float(data_list[i][cdum[j]]) < EOH:
                    bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                    val = strat.val[ldum[j]]  # designated isovalue
                    lithid = strat.lithid[ldum[j]]  # lithology id
                    feat_name = strat.sequence[ldum[j]]  # sequence name
                    formatted_data.append(
                        [boreid, easting, northing, bottom, val, lithid, feat_name, gx, gy, gz,"raw_data"]
                    )
                else:
                    EOH_flag = True   
                current_bottom = np.copy(bottom)
                if type(first) == type(None):
                    first = strat.val[ldum[j]-1]
                     
        #Warnbro _ unconformity
        if TOW == bottom and not EOH_flag:
            val = strat.val[13]
            lithid = strat.lithid[13]  # lithology id
            feat_name = strat.sequence[13]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom+2., val, lithid, feat_name, gx, gy, gz,"raw_data"]
            )                      
        else:    
            val = strat.val[11]
            lithid = strat.lithid[12]  # lithology id
            feat_name = strat.sequence[12]  # sequence name
            formatted_data.append(
                [boreid, easting, northing, TOW, first, lithid, feat_name, gx, gy, gz,"raw_data"]
            ) 
        bottom = np.copy(current_bottom)  # current depth
        if not EOH_flag:
            val = 0  # designated isovalue
            lithid = 'base_Warn'  # lithology id
            feat_name = "Warn_U"  # sequence name
            formatted_data.append(
                [boreid, easting, northing, bottom, val, lithid, feat_name, 0, 0, 1,"raw_data"]
            )
        TOY = np.copy(bottom)
        #Yaragadee, or parmelias, no real yaragadee bases
        ldum = [14,15,16,17]
        cdum = [23,24,25,26]
        for j in range(3):
            if isinstance(data_list[i][cdum[j]], numbers.Number) == True:
                if float(data_list[i][cdum[j]]) < EOH:
                    bottom = groundlevel - float(data_list[i][cdum[j]])  # Ground surface - kinhs park base (mbgl)
                    val = strat.val[ldum[j]]  # designated isovalue
                    lithid = strat.lithid[ldum[j]]  # lithology id
                    feat_name = strat.sequence[ldum[j]]  # sequence name
                    formatted_data.append(
                        [boreid, easting, northing, bottom, val, lithid, feat_name, np.nan, np.nan, np.nan,"raw_data"]
                    )
                    current_bottom = np.copy(bottom)
        if isinstance(data_list[i][23], numbers.Number) == True:    
            val = strat.val[13]
            lithid = strat.lithid[14]  # lithology id
            feat_name = strat.sequence[14]  # sequence name
            print('here', boreid,lithid)
            formatted_data.append(
                [boreid, easting, northing, TOY, val, lithid, feat_name, gx, gy, gz,"raw_data"]
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
        valdum = [0.,-44.,0.,0.,-440,-607,-660,-783,0][::-1]
        feature = ['Ground','Quaternary','NC1_FEAT','NC5_FEAT', #['Quaternary','Quaternary','NC1_FEAT','NC5_FEAT',
                   'Leederville','Leederville','Leederville'
                   ,'Warnbro','Warn_U'][::-1]
        
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
                ddum.append([BoreID, X, Y, LSE-Lley["SPS"][i], valdum[1], 'Lley', feature[1], np.nan, np.nan, np.nan,"Lucy"])
            if type(Lley["Maringiniup"][i]) == type(1):
                ddum.append([BoreID, X, Y, LSE-Lley["Maringiniup"][i], valdum[2], 'Lley', feature[2],  np.nan, np.nan, np.nan,"Lucy"])            
            ddum.append([BoreID, X, Y, LSE-Lley["Waneroo"][i], valdum[3], 'Lley', feature[3],  np.nan, np.nan, np.nan,"Lucy"])
            if type(Lley["Pinjar"][i]) == type(1):
                ddum.append([BoreID, X, Y, LSE-Lley["Pinjar"][i], valdum[4], 'Lley', feature[4],  np.nan, np.nan, np.nan,"Lucy"])                    
            ddum.append([BoreID, X, Y, LSE-Lley["Aptian"][i], valdum[5], 'Lley', feature[5], 0.0, 0.0, 1.0,"Lucy"])   
            if type(Lley["Paleocene"][i]) == type(1):
                ddum.append([BoreID, X, Y, LSE-Lley["Paleocene"][i], valdum[5], 'Lley', feature[5], 0.0, 0.0, 1.0,"Lucy"]) 
            ddum.append([BoreID, X, Y, LSE-Lley["TQ"][i], valdum[6], 'Lley', feature[6], 0.0, 0.0, 1.0,"Lucy"]) 

        data = pd.concat([data,pd.DataFrame(ddum,columns = data.columns)])
  
    if Brett:
        gphys = pd.read_excel("../data/data_dwer/geology.xls", sheet_name="Other_constraints")
        ddum = []
        for i in range(len(gphys)):
            ddum.append(['GEO', gphys["Easting"][i], gphys["Northing"][i], gphys["z"][i], gphys["val"][i], 'GEO', gphys["Feature"][i], np.nan, np.nan, np.nan,"Brett"]) 

        data = pd.concat([data,pd.DataFrame(ddum,columns = data.columns)]) 

    if Petroleum:
        Pet = pd.read_excel(
        "../data/data_dwer/geology.xls", sheet_name="Petroleum_readable")
        ddum = []
        for i in range(len(Pet)):
            ddum.append(['PET',Pet["X"][i], Pet["Y"][i], Pet["Z"][i], Pet["val"][i],Pet['lithcode'][i], Pet["feature_name"][i], np.nan, np.nan, np.nan,"WAPIMS"]) 

        data = pd.concat([data,pd.DataFrame(ddum,columns = data.columns)])  
        
    if Fault: 
        #Fault1
        Fault_1 = pd.read_excel(
        "../data/data_seismic/Fault_data.xlsx", 
        sheet_name="Fault_1")   
        rows = []
        for i in range(len(Fault_1)):
            nx1, ny1, nz1 = strike_dip_vector([Fault_1['strike'][i]],[Fault_1['dip'][i]])[0]
            df_new_row = pd.DataFrame.from_records(
                {
                    "X": [Fault_1['X'][i]],
                    "Y": [Fault_1['Y'][i]],
                    "Z": [Fault_1['Z'][i]],
                    "val": [0.0],
                    "feature_name": ["Fault_1"],
                    "nx": [nx1],
                    "ny": [ny1],
                    "nz": [nz1],
                    "for_Kerry" : ["Fault_Seis"]
                }
            )
            data = pd.concat([data,df_new_row], ignore_index=True) 
            
        Fault_2 = pd.read_excel(
        "../data/data_seismic/Fault_data.xlsx", 
        sheet_name="Fault_2")   
        rows = []
        for i in range(len(Fault_2)):
            nx1, ny1, nz1 = strike_dip_vector([Fault_2['strike'][i]],[Fault_2['dip'][i]])[0]
            df_new_row = pd.DataFrame.from_records(
                {
                    "X": [Fault_2['X'][i]],
                    "Y": [Fault_2['Y'][i]],
                    "Z": [Fault_2['Z'][i]],
                    "val": [0.0],
                    "feature_name": ["Fault_2"],
                    "nx": [nx1],
                    "ny": [ny1],
                    "nz": [nz1],
                    "for_Kerry" : ["Fault_Seis"]
                }
            )
            data = pd.concat([data,df_new_row], ignore_index=True) 

        Fault_3 = pd.read_excel(
        "../data/data_seismic/Fault_data.xlsx", 
        sheet_name="Fault_3")   
        rows = []
        for i in range(len(Fault_3)):
            nx1, ny1, nz1 = strike_dip_vector([Fault_3['strike'][i]],[Fault_3['dip'][i]])[0]
            df_new_row = pd.DataFrame.from_records(
                {
                    "X": [Fault_3['X'][i]],
                    "Y": [Fault_3['Y'][i]],
                    "Z": [Fault_3['Z'][i]],
                    "val": [0.0],
                    "feature_name": ["Fault_3"],
                    "nx": [nx1],
                    "ny": [ny1],
                    "nz": [nz1],
                    "for_Kerry" : ["Fault_Seis"]
                }
            )
            data = pd.concat([data,df_new_row], ignore_index=True)            

        Fault_4 = pd.read_excel(
        "../data/data_seismic/Fault_data.xlsx", 
        sheet_name="Fault_4")   
        rows = []
        for i in range(len(Fault_4)):
            nx1, ny1, nz1 = strike_dip_vector([Fault_4['strike'][i]],[Fault_4['dip'][i]])[0]
            df_new_row = pd.DataFrame.from_records(
                {
                    "X": [Fault_4['X'][i]],
                    "Y": [Fault_4['Y'][i]],
                    "Z": [Fault_4['Z'][i]],
                    "val": [0.0],
                    "feature_name": ["Fault_4"],
                    "nx": [nx1],
                    "ny": [ny1],
                    "nz": [nz1],
                    "for_Kerry" : ["Fault_Seis"]
                }
            )
            data = pd.concat([data,df_new_row], ignore_index=True)    
            
        Fault_5 = pd.read_excel(
        "../data/data_seismic/Fault_data.xlsx", 
        sheet_name="Fault_5")   
        rows = []
        for i in range(len(Fault_5)):
            nx1, ny1, nz1 = strike_dip_vector([Fault_5['strike'][i]],[Fault_5['dip'][i]])[0]
            df_new_row = pd.DataFrame.from_records(
                {
                    "X": [Fault_5['X'][i]],
                    "Y": [Fault_5['Y'][i]],
                    "Z": [Fault_5['Z'][i]],
                    "val": [0.0],
                    "feature_name": ["Fault_5"],
                    "nx": [nx1],
                    "ny": [ny1],
                    "nz": [nz1],
                    "for_Kerry" : ["Fault_Seis"]
                }
            )
            data = pd.concat([data,df_new_row], ignore_index=True)   
    
    structuralmodel.data = data

def create_structuralmodel(structuralmodel):

    from LoopStructural import GeologicalModel
    model = GeologicalModel(structuralmodel.origin, structuralmodel.maximum)
    model.data = structuralmodel.data
    
        #add foliations

    Ground = model.create_and_add_foliation(
        "Ground", nelements=1e4, buffer=0.1
    )

    UC = model.add_unconformity(
    model["Ground"], 0.
    )

    #First add the quaternary formation
    Quat = model.create_and_add_foliation(
        "Quaternary", nelements=1e4, buffer=0.1
    )
    
    Quat_UC = model.create_and_add_foliation(
        "Quat_U", nelements=1e4, buffer=0.1
    )
    
    UC0 = model.add_unconformity(
    model["Quaternary"], -44.
    )
    
    """UC0 = model.add_unconformity(
    model["Quaternary"], -44.
    )"""
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
    UC2 = model.add_onlap_unconformity(
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

    fault_slip_vector = [0., 0, 1.]
    Fault_1 = model.create_and_add_fault(
        "Fault_1",
        displacement=500,
        fault_slip_vector=fault_slip_vector,
        nelements=1e4,
        faultfunction = 'BaseFault'
    )
    
    Fault_4 = model.create_and_add_fault(
        "Fault_4",
        displacement=200,
        fault_slip_vector=fault_slip_vector,
        nelements=1e4,
        faultfunction = 'BaseFault'
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

    fault_slip_vector = [0, 0, -1.]
    Fault_2 = model.create_and_add_fault(
        "Fault_2",
        displacement=100,
        fault_slip_vector=fault_slip_vector,
        nelements=1e4,
        faultfunction = 'BaseFault'
    )
    
    fault_slip_vector = [0, 0, 1.]
    Fault_3 = model.create_and_add_fault(
        "Fault_3",
        displacement=100,
        fault_slip_vector=fault_slip_vector,
        nelements=1e4,
        faultfunction = 'BaseFault'
    )
    
    
    fault_slip_vector = [0, 0, 1.]
    Fault_5 = model.create_and_add_fault(
        "Fault_5",
        displacement=500,
        fault_slip_vector=fault_slip_vector,
        nelements=1e4,
        faultfunction = 'BaseFault'
    )

    YARR = model.create_and_add_foliation(
        "Yarragadee", nelements=1e4, buffer=0.1
    )   

    model.set_stratigraphic_column(structuralmodel.strat_col)
    
    structuralmodel.model = model

