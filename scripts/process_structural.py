import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
from LoopStructural.utils import strikedip2vector as strike_dip_vector
import geopandas as gpd
from loopflopy.mesh_routines import resample_linestring
import rasterio
from rasterio.transform import rowcol
from openpyxl import load_workbook
from shapely.geometry import LineString, Point

def prepare_strat_column(structuralmodel):
    
    strat = pd.read_excel(structuralmodel.geodata_fname, sheet_name = structuralmodel.strat_sheetname)
    strat_names = strat.unit.tolist()
    lithids = strat.lithid.tolist()
    vals = strat.val.tolist()
    nlg = len(strat_names) - 1 # number of geological layers
    sequences = strat.sequence.tolist()
    sequence = list(dict.fromkeys(sequences)) # Preserves order and removes duplicates
    
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
    print(strat)
    structuralmodel.strat = strat
    structuralmodel.strat_col = stratigraphic_column
    structuralmodel.strat_names = strat_names
    structuralmodel.cmap = cmap
    structuralmodel.norm = norm
    structuralmodel.sequence = sequence
    structuralmodel.lithids = lithids
    structuralmodel.sequences = sequences
    structuralmodel.vals = vals

def geo_boundaries(project, structuralmodel, geo_crop_poly, simplify_tolerance, node_spacing):
    df = pd.read_excel('../data/data_geology/Otorowiri_outcrop.xlsx', sheet_name = 'O-P contact simplified')
    df = df.dropna(subset=['Easting', 'Northing'])
    points = [Point(xy) for xy in zip(df['Easting'], df['Northing'])]
    line = LineString(points)
    ls_simple = line.simplify(tolerance=simplify_tolerance, preserve_topology=True)
    ls_resample = resample_linestring(ls_simple, node_spacing) # Resample linestring
    op_ls = LineString(ls_resample)
    op_gdf = gpd.GeoDataFrame(geometry = [op_ls], crs=project.crs)
    structuralmodel.op_gdf = gpd.clip(op_gdf, geo_crop_poly).reset_index(drop=True)
    structuralmodel.OP_nodes = op_ls.coords

    df = pd.read_excel('../data/data_geology/Otorowiri_outcrop.xlsx', sheet_name = 'Y-O contact simplified')
    df = df.dropna(subset=['Easting', 'Northing'])
    points = [Point(xy) for xy in zip(df['Easting'], df['Northing'])]
    line = LineString(points)
    ls_simple = line.simplify(tolerance=simplify_tolerance, preserve_topology=True)
    ls_resample = resample_linestring(ls_simple, node_spacing) # Resample linestring
    yo_ls = LineString(ls_resample)
    yo_gdf = gpd.GeoDataFrame(geometry = [yo_ls], crs=project.crs)
    structuralmodel.yo_gdf = gpd.clip(yo_gdf, geo_crop_poly).reset_index(drop=True)
    structuralmodel.YO_nodes = yo_ls.coords

# bring in the data from the outcrop in process spatial to add 'obs' points from the outcrop
def add_geo_boundaries(structuralmodel):
    df = pd.read_excel(structuralmodel.geodata_fname, sheet_name='geo') #Refer to the geology spreadsheet
    #import all the relative boundary information points
    OP_boundary = pd.DataFrame(structuralmodel.OP_nodes, columns=['Easting', 'Northing'])
    OP_boundary['ID'] = ['OP_boundary' + str(i) for i in range(len(OP_boundary))]
    OP_boundary['Data_type'] = 'Control'
    OP_boundary['Source'] = 'DMIRS geology shapefile'
    OP_boundary['Kp'] = 0
    OP_boundary['Kpo'] = '-'
    OP_boundary = OP_boundary[['ID', 'Easting', 'Northing', 'Data_type', 'Source', 'Kp', 'Kpo']]
    combined_df = pd.concat([df, OP_boundary], ignore_index=True)
    #print(combined_df)
    YO_boundary = pd.DataFrame(structuralmodel.YO_nodes, columns=['Easting', 'Northing'])
    YO_boundary['ID'] = ['YO_boundary' + str(i) for i in range(len(YO_boundary))]
    YO_boundary['Data_type'] = 'Control'
    YO_boundary['Source'] = 'DMIRS geology shapefile'
    YO_boundary['Kp'] = '-'
    YO_boundary['Kpo'] = 0
    YO_boundary = YO_boundary[['ID', 'Easting', 'Northing', 'Data_type', 'Source', 'Kp', 'Kpo']]
    combined_df = pd.concat([combined_df, YO_boundary], ignore_index=True)
    #write the combined data to a new excel sheet in the same file
    output_excel_path = structuralmodel.geodata_fname
    new_sheet_name = 'geo_boundaries'
    with pd.ExcelWriter(output_excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    print(f"\nUpdated DataFrame with geo_boundaries written to new sheet: '{new_sheet_name}' in file: {output_excel_path}")

#fill in any blank z values with an extract from the asc file
def elevation_fill_unknown(structuralmodel):
    #clipped_DEM = spatial.model_DEM #path to the clipped DEM file
    clipped_DEM = '../data/data_dem/Otorowiri_Model_DEM.tif'
    with rasterio.open(clipped_DEM) as src:
        ##check the DEM bounds
        bounds = src.bounds
        print("DEM bounds:")
        print(f"  xmin (left):   {bounds.left}")
        print(f"  ymin (bottom): {bounds.bottom}")
        print(f"  xmax (right):  {bounds.right}")
        print(f"  ymax (top):    {bounds.top}")
        print(f"  CRS:           {src.crs}")
    df = pd.read_excel(structuralmodel.geodata_fname, sheet_name='geo_boundaries') #Refer to the geology spreadsheet
    min_easting = df['Easting'].min()
    max_easting = df['Easting'].max()
    min_northing = df['Northing'].min()
    max_northing = df['Northing'].max()

    print("DataFrame coordinate extents:")
    print(f"  Easting:  min = {min_easting}, max = {max_easting}")
    print(f"  Northing: min = {min_northing}, max = {max_northing}")
    
    # Replace empty string with NaN in elevation
    df['Ground_mAHD'] = df['Ground_mAHD'].replace("", np.nan)
    missing_idx = df['Ground_mAHD'].isna()

    # If any elevations are missing, sample from the DEM
    if missing_idx.any():
        with rasterio.open(clipped_DEM) as src:
            nodata = src.nodata
            bounds = src.bounds

            # Extract coordinates for missing values
            coords = list(zip(df.loc[missing_idx, 'Easting'], df.loc[missing_idx, 'Northing']))

            # Filter coords to only those within DEM bounds
            valid_idx = [
                idx for idx, (x, y) in zip(df.loc[missing_idx].index, coords)
                if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
            ]
            valid_coords = [
                (df.at[idx, 'Easting'], df.at[idx, 'Northing']) for idx in valid_idx
            ]

            if not valid_coords:
                print("No valid coordinates inside the DEM bounds.")
                return df

            # Sample DEM values at those coordinates
            updated_elevations = []
            sampled_values = list(src.sample(valid_coords))

            # Assign values back to DataFrame
            for idx, val in zip(valid_idx, sampled_values):
                elevation = val[0]
                if elevation != nodata:
                    df.at[idx, 'Ground_mAHD'] = elevation
                    updated_elevations.append((idx, elevation))
                else:
                    # If the sampled value is nodata, set to NaN
                    df.at[idx, 'Ground_mAHD'] = np.nan
                    updated_elevations.append((idx, np.nan))
                #df.at[idx, 'Ground_mAHD'] = elevation if elevation != nodata else np.nan
                #updated_elevations.append((idx, np.nan))

            print(f"Filled {len(valid_coords)} missing 'Ground_mAHD' values from DEM.")
            # Print the list of updated values
            print("\nUpdated 'Ground_mAHD' values from DEM:")
            for idx, elev in updated_elevations:
                print(f"  Index {idx}: Elevation = {elev}")
    else:
        print("No missing 'Ground_mAHD' values to fill.")
    
    df = df.dropna(subset=['Ground_mAHD'])  # Keep only rows with valid elevation
    df = df[df['Ground_mAHD'] != 0]  # Drop rows where elevation is 0 - this is invalid for this area

    output_excel_path = structuralmodel.geodata_fname
    new_sheet_name = 'geodata_elevation'
    with pd.ExcelWriter(output_excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    print(f"\nUpdated DataFrame written to new sheet: '{new_sheet_name}' in file: {output_excel_path}")

    return df

# elevation fill ALL data - this will match all borehole tops to the DEM
def elevation_fill_all(structuralmodel):
    clipped_DEM = '../data/data_dem/Otorowiri_Model_DEM.tif'
    df = pd.read_excel(structuralmodel.geodata_fname, sheet_name='geo_boundaries') #Refer to the geology spreadsheet
    #see QAQC code above in elevation_fill_unknown if needed
    
    # Open the DEM file and get its bounds and nodata value
    with rasterio.open(clipped_DEM) as src:
        nodata = src.nodata
        bounds = src.bounds

        # list all coordinates within the geological spreadsheet
        coords = list(zip(df['Easting'], df['Northing']))

        # Filter coordinates within DEM bounds
        valid_idx = [
            idx for idx, (x, y) in zip(df.index, coords)
            if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top]
        valid_coords = [
            (df.at[idx, 'Easting'], df.at[idx, 'Northing']) for idx in valid_idx]

        if not valid_coords:
            print("No valid coordinates inside the DEM bounds.")
            return df

        sampled_values = list(src.sample(valid_coords))

        # Replace ALL elevation values with DEM elevation
        for idx, val in zip(valid_idx, sampled_values):
            elevation = val[0]
            if elevation != nodata:
                df.at[idx, 'Ground_mAHD'] = elevation
            else:
                df.at[idx, 'Ground_mAHD'] = np.nan

        print(f"Replaced elevation values for {len(valid_coords)} points from DEM.")

    # Remove invalid values
    df = df.dropna(subset=['Ground_mAHD'])  # Drop NaNs
    df = df[df['Ground_mAHD'] != 0]         # Drop 0 elevations if invalid for your region

    # Save to a new sheet in the same Excel file
    output_excel_path = structuralmodel.geodata_fname
    new_sheet_name = 'geodata_elevation'
    with pd.ExcelWriter(output_excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=new_sheet_name, index=False)

    print(f"Updated elevations written to sheet: '{new_sheet_name}' in {output_excel_path}")

    return df

# ---------- Prepare borehole data ----------------
def prepare_geodata(structuralmodel, extent = None, Fault = True):
    df= pd.read_excel(structuralmodel.geodata_fname, sheet_name='geodata_elevation') #this should be the corrected data with elevations
    df['Ground_mAHD'] = pd.to_numeric(df['Ground_mAHD'], errors='coerce')  # Ensure values are numeric, convert invalid to NaN
    print(f"{len(df)} valid elevation points retained for further processing.")
    strat = structuralmodel.strat
    data_list = df.values.tolist()  # Turn data into a list of lists
    formatted_data = []
    for i in range(len(data_list)): #iterate for each row
        data_type = data_list[i][3]  
        
        #-----------RAW DATA-----------------------
        if data_type == 'Raw': ### Z VALUES: Ground (mAHD), Else: (mBGL)
            
            boreid = data_list[i][0]
            easting, northing = data_list[i][1], data_list[i][2]
            groundlevel = data_list[i][5]  

            # Add ground level to dataframe
            formatted_data.append([boreid, easting, northing, groundlevel, 0, 'Ground', 'Ground', 0, 0, 1, data_type]) 

            #print(df.shape[1])
            count = 1  # Add data row for each lithology
            for j in range(6,8): #iterate through each formation 
                if isinstance(data_list[i][j], numbers.Number) == True:  # Add lithology
                    #print(count)  
                    bottom    = groundlevel - float(data_list[i][j])  # Ground surface - formation bottom (mbgl)
                    val       = strat.val[count]                   # designated isovalue
                    unit      = strat.unit[count]                  # unit 
                    feature   = strat.sequence[count]              # sequence
                    gx, gy, gz = np.nan, np.nan, np.nan                              # normal vector to surface (flat) 
                    formatted_data.append([boreid, easting, northing, bottom, val, unit, feature, gx, gy, gz, data_type])    
                    current_bottom = np.copy(bottom)    
                count+=1
        
        #-----------CONTROL POINT-----------------------
        if data_type == 'Control': ### Z VALUES: mAHD

            boreid = data_list[i][0]
            easting, northing = data_list[i][1], data_list[i][2]
            groundlevel = data_list[i][5] ### this directly refers to the 'Ground_mAHD' column in the data sheet

            count = 1  # Add data row for each lithology
            for j in range(6,8): #iterate through each formation of interest
                if isinstance(data_list[i][j], numbers.Number) == True:  # Add lithology  
                    Z         = groundlevel - float(data_list[i][j])             # This finds the UNIT elevation in mAHD, 
                    val       = strat.val[count]                   # designated isovalue
                    unit      = strat.unit[count]                  # unit 
                    feature   = strat.sequence[count]              # sequence
                    gx, gy, gz = np.nan, np.nan, np.nan                            # normal vector to surface (flat) 
                    formatted_data.append([boreid, easting, northing, Z, val, unit, feature, gx, gy, gz, data_type])      
                count+=1
                
    data = pd.DataFrame(formatted_data)
    data.columns =['ID','X','Y','Z','val','lithcode','feature_name', 'gx', 'gy', 'gz','data_type']

    structuralmodel.data = data

    output_excel_path = structuralmodel.geodata_fname
    new_sheet_name = 'structuralmodel_data'
    with pd.ExcelWriter(output_excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        data.to_excel(writer, sheet_name=new_sheet_name, index=False)
    print(f"\nUpdated DataFrame written to new sheet: '{new_sheet_name}' in file: {output_excel_path}")

def create_structuralmodel(structuralmodel):
    
    origin  = (structuralmodel.x0, structuralmodel.y0, structuralmodel.z0)
    maximum = (structuralmodel.x1, structuralmodel.y1, structuralmodel.z1)

    import LoopStructural
    from LoopStructural import GeologicalModel
    print(LoopStructural.__version__)

    model = GeologicalModel(origin, maximum)
    model.set_model_data(structuralmodel.data)  
    
    Ground     = model.create_and_add_foliation("Ground", nelements=1e4, interpolatortype = "FDI")
    Ground_UC  = model.add_unconformity(Ground, structuralmodel.strat[structuralmodel.strat.unit == 'Ground'].val.iloc[0]) 
    Yarragadee         = model.create_and_add_foliation("Yarragadee", nelements=1e4 , interpolatortype = "FDI")

    #Kp        = model.create_and_add_foliation("Kp", nelements=1e4 , interpolatortype = "FDI", buffer = 0.1)
    #Kp_UC     = model.add_unconformity(Kp, structuralmodel.strat[structuralmodel.strat.unit == 'Kp'].val.iloc[0])
    #Kpo        = model.create_and_add_foliation("Kpo", nelements=1e4 , interpolatortype = "FDI", buffer = 0.1)
    model.set_stratigraphic_column(structuralmodel.strat_col)
    structuralmodel.model = model
    

    
