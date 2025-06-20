import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import loopflopy.utils as utils
import flopy
import math
import os
import rasterio
from scipy.interpolate import griddata
#from adjustText import adjust_text

#####STEP 0: Extract a csv file of borehole IDs from the DWER extract that can be fed into the WIR data request#####
def create_bore_list():
    DWER_extract = gpd.read_file("../data/data_waterlevels/WIN_Sites_ArrowsmithExtract.shp")
    valid_bores = DWER_extract[DWER_extract['SITE_ID'].astype(str).str.match(r'^\d{8}$')]  # Filter for 8-digit borehole IDs
    bore_IDs = valid_bores['SITE_ID'].astype(int).reset_index(drop=True) #change to astype(str) to set back to strings
    # WIR can only accept bore IDs of less than 1000, so the total list needs to be split into chunks of 1000
    output_dir = "../data/data_waterlevels/obs/bore_id_chunks"
    os.makedirs(output_dir, exist_ok=True)
    chunk_size = 999
    num_chunks = math.ceil(len(bore_IDs) / chunk_size)
    for i in range(num_chunks):
        chunk = bore_IDs[i * chunk_size : (i + 1) * chunk_size]
        output_path = os.path.join(output_dir, f"bore_id_list_{i+1:02d}.txt")
        chunk.to_csv(output_path, index=False, header=False, float_format='%.0f') # take off float_format to get strings
        print(f"Saved chunk {i+1} with {len(chunk)} IDs to: {output_path}")
    # Save a full csv with no header and no index for easy WIR data request
    output_path = "../data/data_waterlevels/obs/bore_id_list.txt"
    bore_IDs.to_csv(output_path, index=False, header=False)
    print(f"{len(bore_IDs)} BORE_IDs written to: {output_path}")

#####STEP 1: Trim ALL the observation points extracted from DWER to model boundary shape#####
    # This will likely still contain a lot of information from bores which are NOT screened in the Parmelia aquifer

def trim_obs_to_shape(geomodel_shapefile):
    all_extracted_obs = gpd.read_file("../data/data_waterlevels/WIN_Sites_ArrowsmithExtract.shp")
    model_boundary = gpd.read_file(geomodel_shapefile)
    if all_extracted_obs.crs != model_boundary.crs:
        print("CRS mismatch: reprojecting observation points to match model boundary.")
        all_extracted_obs = all_extracted_obs.to_crs(model_boundary.crs)

    # keep only bore points that fall within the model polygon
    bores_in_model = gpd.sjoin(all_extracted_obs, model_boundary, how="inner", predicate="within")

    # drop extra columns from the spatial join
    bores_in_model = bores_in_model.drop(columns=[col for col in bores_in_model.columns if 'index_right' in col])

    # Filter for SITE_IDs that are exactly 8 digits - these are the only borehole IDs (the rest are surface or weather stations)
    bores_in_model = bores_in_model[
        bores_in_model['SITE_ID'].astype(str).str.match(r'^\d{8}$')]

    print(f"{len(bores_in_model)} bore points retained within model boundary.")

    bore_ids = bores_in_model['SITE_ID'].astype(int).reset_index(drop=True)
    output_path = "../data/data_waterlevels/obs/filtered_bore_ids.txt"
    bore_ids.to_csv(output_path, index=False, header=False)
    print(f"Filtered bore ID list written to: {output_path}")

    #plot all the initial points within the polygon
    fig, ax = plt.subplots(figsize=(8, 8))
    model_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    bores_in_model.plot(ax=ax, color='red', markersize=25, label='Filtered Bores')
    ax.set_title("Filtered Bore Locations Within Model Boundary")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return bores_in_model

#####STEP 2: For all points within geomodel boundary start merging all essential details from all the different Excel sheets#####
        #this step collates all the data fields of interest but does NOT filter out any unusable/incomplete bores

def create_merged_obs(bores_in_model):
    #filter the full bore list to be only those spatially within the model
    site_details = pd.read_excel('../data/data_waterlevels/model_area_raw/174416/Site - All Site Details (Excel).xlsx', sheet_name='Site Details')
    bore_in_model_filter = bores_in_model['SITE_ID'].astype(str).unique()
    site_details['Site Ref'] = site_details['Site Ref'].astype(str)
    site_in_model_details = site_details[site_details['Site Ref'].isin(bore_in_model_filter)]
    print(f"{len(site_in_model_details)} site records matched to model bore IDs.")

    # Extract the fields of interest from site details and create the preliminary dataframe to add to
    columns_from_site_details = ['Site Ref', 'Site Name', 'Site Short Name', 'Easting', 'Northing', 'Latitude', 'Longitude']
    df = site_in_model_details[columns_from_site_details]
    df = df.sort_values('Site Ref').reset_index(drop=True)

    # Add the Elevations where they exist from the 'Depth Measurement Points' sheet
    elevations = pd.read_excel('../data/data_waterlevels/model_area_raw/174416/Site - All Site Details (Excel).xlsx', sheet_name='Depth Measurement Points')
    elevations = elevations[elevations['Measurement Point Type'] == 'Ground level'].drop_duplicates(subset=['Site Ref'])
    elevations['Site Ref'] = elevations['Site Ref'].astype(str)
    df = df.merge(
        elevations[['Site Ref', 'Elevation (m as per Datum Plane)']],
        on='Site Ref',
        how='left')
    df = df.rename(columns={'Elevation (m as per Datum Plane)': 'Elevation_DWER'})

    # Add the drilled depths and owder names from the 'Borehole Information' sheet
    drilled_depths = pd.read_excel('../data/data_waterlevels/model_area_raw/174416/Site - All Site Details (Excel).xlsx', sheet_name='Borehole Information')
    drilled_depths['Site Ref'] = drilled_depths['Site Ref'].astype(str)
    df = df.merge(
        drilled_depths[['Site Ref', 'Depth Drilled (mbGL)', 'Owner Name']],
        on='Site Ref',
        how='left')
    df = df.rename(columns={'Depth Drilled (mbGL)': 'Drilled Depth'})

    # Add the screened intervals from the 'Casing' sheet
    casing = pd.read_excel('../data/data_waterlevels/model_area_raw/174416/Site - All Site Details (Excel).xlsx', sheet_name='Casing')
    casing = casing[casing['Element'] == 'Inlet (screen)']
    casing['Site Ref'] = casing['Site Ref'].astype(str)
    df = df.merge(
        casing[['Site Ref', 'From (mbGL)', 'To (mbGL)', 'Inside Dia. (mm)']],
        on='Site Ref',
        how='left')
    df = df.rename(columns={
        'From (mbGL)': 'Screen From (mbGL)',
        'To (mbGL)': 'Screen To (mbGL)',
        'Inside Dia. (mm)': 'Screen Diameter (mm)'})

    output_path = "../data/data_waterlevels/obs/01_All_Groundwater_bores.xlsx"
    df.to_excel(output_path, index=False)

#####STEP 3: Assign elevations from the project DEM file to all the boreholes#####
        #this step allows us to include bores that were not surveyed, and makes sure that even those that DO contain elevation data are
        #compatible with the geomodel (no issues later with depth/geomodel mismatches)

def fill_bore_elevations():
    # Fill in the elevations from the DEM data for all bores
    df = pd.read_excel("../data/data_waterlevels/obs/01_All_Groundwater_bores.xlsx")
    geomodel_DEM = '../data/data_dem/Otorowiri_Geomodel_DEM.tif' #DEM of the geological model, which should be the same as the boundary
    with rasterio.open(geomodel_DEM) as src:
        coords = list(zip(df['Easting'], df['Northing']))
        sampled = list(src.sample(coords))
        elevations = [val[0] if val[0] != src.nodata else np.nan for val in sampled]
    df['Elevation_DEM'] = elevations

    # QAQC differences between the DEM elevations and the DWER elevations
    if 'Elevation_DWER' in df.columns:
        df['Elevation_diff'] = df['Elevation_DWER'] - df['Elevation_DEM']
        def check_qaqc(row):
            if pd.notna(row['Elevation_DWER']) and pd.notna(row['Elevation_DEM']):
                return 'PASS' if abs(row['Elevation_diff']) <= 5 else 'FAIL'
            return np.nan
        df['Elevation_QAQC'] = df.apply(check_qaqc, axis=1)
        insert_after = 'Elevation_DWER'
        col_order = df.columns.tolist()
        for col in ['Elevation_DEM', 'Elevation_diff', 'Elevation_QAQC']:
            if col in col_order:
                col_order.remove(col)
        insert_index = col_order.index(insert_after) + 1
        for col in ['Elevation_DEM', 'Elevation_diff', 'Elevation_QAQC']:
            col_order.insert(insert_index, col)
            insert_index += 1
        df = df[col_order]
    else:
        print("Warning: 'Elevation_DWER' column not found; QAQC not performed.")

    output_path = "../data/data_waterlevels/obs/02_Model_bores_elevations.xlsx"
    df.to_excel(output_path, index=False)

#####STEP 3: Assign aquifer by evaluating the bore information, attached to new DEM elevations, against the geomodel #####
        #this step will convert all bore depth and screened interval data to mbGL according to DEM-extracted elevations
        #aquifer is assigned by checking either the screened interval data or depth data against geomodel aquifer bottom

def assign_aquifer(mesh, geomodel):
    df = pd.read_excel("../data/data_waterlevels/obs/02_Model_bores_elevations.xlsx")
    parmelia_bottom = geomodel.botm_geo[0] # Bottom of the Parmelia aquifer
    cell_coords = np.column_stack([mesh.xc, mesh.yc])
    coords = np.column_stack([df['Easting'], df['Northing']])
    df['Kp_botm'] = griddata(points=cell_coords, values=parmelia_bottom, xi=coords, method='linear') # use method = 'nearest' or 'cubic' to play with the interpolation

    # Putting all the drills depths and screen intervals into mAHD from the DEM data
    df['Bore_bottom_elevation'] = df['Elevation_DEM'] - df['Drilled Depth']
    zero_screen_mask = (df['Screen From (mbGL)'] == 0) & (df['Screen To (mbGL)'] == 0)
    df.loc[zero_screen_mask, ['Screen From (mbGL)', 'Screen To (mbGL)']] = np.nan
    if {'Screen From (mbGL)', 'Screen To (mbGL)'}.issubset(df.columns):
        df['Screen_top_elevation'] = df['Elevation_DEM'] - df['Screen From (mbGL)']
        df['Screen_bottom_elevation'] = df['Elevation_DEM'] - df['Screen To (mbGL)']
        df['zobs'] = df['Screen_bottom_elevation'] + (df['Screen_top_elevation'] - df['Screen_bottom_elevation'])/2

    # Create a df to track unused bores - these will be added to tracker spreadsheet
    unused_bores = pd.DataFrame(columns=['Site Ref', 'Site Name', 'Site Short Name', 'Owner Name', 'Reason'])

    # Filter out bores that have both a depth of 0 and no screen information
    zero_depth = df['Drilled Depth'].fillna(0) == 0
    no_screen_info = df[['Screen From (mbGL)', 'Screen To (mbGL)']].isna().all(axis=1)
    insufficient_info_mask = zero_depth & no_screen_info
    if insufficient_info_mask.any():
        filtered_df = df.loc[insufficient_info_mask, ['Site Ref', 'Site Name', 'Site Short Name', 'Owner Name']].copy()
        filtered_df['Reason'] = 'Insufficient bore information'
        unused_bores = pd.concat([unused_bores, filtered_df], ignore_index=True)   
    df = df[~insufficient_info_mask].copy()
    print(f"{len(df)} bores retained after filtering out zero-depth with no screen info.")

    # If bores have a depth of 0 but some screen information, assume the screen bottom is the drilled depth
    #(we're going to assume that even if the bore is drilled deeper, the screens represent the sampled aquifer)
    depth_is_zero = df['Drilled Depth'].fillna(0) == 0
    has_screen_info = df[['Screen From (mbGL)', 'Screen To (mbGL)']].notna().any(axis=1)
    assumed_depth_mask = depth_is_zero & has_screen_info
    df.loc[assumed_depth_mask, 'Drilled Depth'] = df.loc[assumed_depth_mask, 'Screen To (mbGL)']
    assumed_depth_bores = df.loc[assumed_depth_mask, 'Site Short Name'].tolist()
    if assumed_depth_bores:
        print(f"The bores {assumed_depth_bores} have assumed drill depths based on screen information.")

    # Assign aquifer based on the screen bottom elevation first, where available
    # If screen bottom elevations aren't available, then the bore bottom will be used
    def choose_aquifer(row):
        kp_botm = row['Kp_botm']
        if pd.isna(kp_botm):
            return np.nan
        if pd.notna(row.get('Screen_bottom_elevation')):
            if row['Screen_bottom_elevation'] >= kp_botm:
                return 'Parmelia'
            else:
                return 'Deeper'
        elif pd.notna(row.get('Bore_bottom_elevation')):
            if row['Bore_bottom_elevation'] >= kp_botm:
                return 'Parmelia'
            else:
                return 'Deeper'
        return 'Unknown'
    df['Aquifer'] = df.apply(choose_aquifer, axis=1)

    # Filter out the bores that are not screened in the Parmelia aquifer
    not_parmelia_mask = df['Aquifer'] != 'Parmelia'
    if not_parmelia_mask.any():
        filtered_df = df.loc[not_parmelia_mask, ['Site Ref', 'Site Name', 'Site Short Name', 'Owner Name']].copy()
        filtered_df['Reason'] = 'Not in Parmelia'
        unused_bores = pd.concat([unused_bores, filtered_df], ignore_index=True)
    df = df[~not_parmelia_mask].copy()
    print(f"{len(df)} bores retained after filtering for Parmelia aquifer.")

    # Once all the bores that are not screened in Parmelia are filtered out, a second pass at filling in the zobs using the bottom of the bore
    zobs_missing_before = df['zobs'].isna().sum()
    df['zobs'] = df['zobs'].fillna(df['Elevation_DEM'] - df['Drilled Depth'] / 2)
    zobs_missing_after = df['zobs'].isna().sum()
    zobs_filled_by_depth = zobs_missing_before - zobs_missing_after
    if zobs_filled_by_depth > 0:
        print(f"{zobs_filled_by_depth} bores had 'zobs' filled in using (Elevation_DEM - Drilled Depth / 2).")
    df['DEM-zobs'] = df['Elevation_DEM'] - df['zobs']
    df['zobs-Kp_botm'] = df['zobs'] - df['Kp_botm']
    
    # Also once only bores screened in Parmelia are left, any boreholes that have two screened intervals should be merged into one
    parmelia_df = df.copy()
    screen_intervals = ['Screen From (mbGL)', 'Screen To (mbGL)']
    grouped = df.groupby('Site Ref')
    merged_screens = grouped[screen_intervals].agg({
        'Screen From (mbGL)': 'min',  # shallowest entry
        'Screen To (mbGL)': 'max'     # deepest entry
        }).reset_index()
    first_rows = grouped.first().reset_index() #this will keep only the first line of the duplicate entry
    first_rows = first_rows.drop(columns=screen_intervals)
    df = pd.merge(first_rows, merged_screens, on='Site Ref', how='left')
    original_cols = parmelia_df.columns.tolist() #re-order columns to be the same as they were in the previous spreadsheet
    final_cols = [] 
    for col in original_cols:
        if col in df.columns:
            final_cols.append(col)
    for col in df.columns: 
        if col not in final_cols:
            final_cols.append(col)
    df = df[final_cols]
    print(f"{len(parmelia_df) - len(df)} duplicate screened intervals merged into single entries.")

    output_path = "../data/data_waterlevels/obs/03_Bores_within_aquifer.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, index=False, sheet_name='Parmelia bores')
        unused_bores.to_excel(writer, index=False, sheet_name='Unused bore tracker')

#####STEP 4: Bring in the water level information from the other WIR extract for the narrowed down bores#####
        #this step filters out all the points that are either missing any water level information, or have no dates available

def filtered_groundwater_obs(geomodel_shapefile):
    raw_WL_df = pd.read_excel('../data/data_waterlevels/model_area_raw/174415/WaterLevelsDiscreteForSiteCrossTab.xlsx')
    #raw_continuous_WL_df = pd.read_excel('../data/data_waterlevels/model_area_raw/174415/WaterLevelsContinuousForSiteCrossTab.xlsx')
    borehole_df = pd.read_excel('../data/data_waterlevels/obs/03_Bores_within_aquifer.xlsx', sheet_name='Parmelia bores')
    unused_df = pd.read_excel('../data/data_waterlevels/obs/03_Bores_within_aquifer.xlsx', sheet_name='Unused bore tracker')

    # Continue tracking boreholes that can't be used
    unused_bores = pd.DataFrame(columns=['Site Ref', 'Site Name', 'Site Short Name', 'Owner Name', 'Reason'])

    # Filter out any boreholes that do not contain water level data
    raw_WL_df['Site Ref'] = raw_WL_df['Site Ref'].astype(str)
    borehole_df['Site Ref'] = borehole_df['Site Ref'].astype(str)
    bores_with_data = raw_WL_df['Site Ref'].unique()
    has_no_data_mask = ~borehole_df['Site Ref'].isin(bores_with_data)
    if has_no_data_mask.any():
        filtered_df = borehole_df.loc[has_no_data_mask, ['Site Ref', 'Site Name', 'Site Short Name', 'Owner Name']].copy()
        filtered_df['Reason'] = 'No water level data'
        unused_bores = pd.concat([unused_bores, filtered_df], ignore_index=True)
    borehole_df = borehole_df[~has_no_data_mask].copy()
    print(f"{len(borehole_df)} bores retained that contain some water level information.")
    unused_df = pd.concat([unused_df, unused_bores], ignore_index=True)

    # Filter out any boreholes that have unknown timeframes (i.e. collect year is 1900)
    raw_WL_df = raw_WL_df[raw_WL_df['Collect Year'] != 1900].copy()
    bores_with_valid_years = raw_WL_df['Site Ref'].unique()
    lost_due_to_1900_mask = ~borehole_df['Site Ref'].isin(bores_with_valid_years)
    if lost_due_to_1900_mask.any():
        removed_bores = borehole_df.loc[lost_due_to_1900_mask, ['Site Ref', 'Site Name', 'Site Short Name', 'Owner Name']].copy()
        removed_bores['Reason'] = 'No sample data available'
        unused_bores = pd.concat([unused_bores, removed_bores], ignore_index=True)
    borehole_df = borehole_df[~lost_due_to_1900_mask].copy()
    print (f"{len(borehole_df)} bores retained that have valid sample years.")
    unused_df = pd.concat([unused_df, unused_bores], ignore_index=True)

    # write the Excel file to QAQC what bores are being discarded for not having water level data
    output_path = '../data/data_waterlevels/obs/04_Bores_with_water_levels.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        borehole_df.to_excel(writer, index=False, sheet_name='Parmelia bores')
        unused_df.to_excel(writer, index=False, sheet_name='Unused bore tracker')
    
    # Continue working with the narrowed down borehole data
    columns_to_keep = ['Site Ref', 'Collect Date', 'Collect Month', 'Collect Year', 'Depth Measurement Point', 'Static water level (m)', 'Water level (AHD) (m)', 'Water level (SLE) (m)']
    WL_df = raw_WL_df[columns_to_keep].copy()
    bore_check = WL_df['Site Ref'].unique()
    print(f"Unique bore IDs in water level data from Water level file: {len(bore_check)}")
    
    # Keep only the boreholes that have been highlighted as in the Parmelia aquifer
    analyse_df = pd.read_excel('../data/data_waterlevels/obs/04_Bores_with_water_levels.xlsx', sheet_name='Parmelia bores')
    WL_df['Site Ref'] = WL_df['Site Ref'].astype(str)
    analyse_df['Site Ref'] = analyse_df['Site Ref'].astype(str)
    WL_df = WL_df[WL_df['Site Ref'].isin(analyse_df['Site Ref'])].copy()
    clean_up_columns = ['Static water level (m)', 'Water level (AHD) (m)', 'Water level (SLE) (m)']
    for col in clean_up_columns:
        WL_df[col] = WL_df[col].astype(str).str.replace(r'[<>~]', '', regex=True).replace('', pd.NA)
        WL_df[col] = pd.to_numeric(WL_df[col], errors='coerce')
    #delete rows that have no inputs in any of the columns 'Static water level (m)', 'Water level (AHD) (m)', 'Water level (SLE) (m)'
    WL_df = WL_df.dropna(subset=['Static water level (m)', 'Water level (AHD) (m)', 'Water level (SLE) (m)'], how='all')
    print(f"Unique bore IDs in water level data after filtering to Parmelia and dropping where there's no WL data: {len(WL_df['Site Ref'].unique())}")

    # Merge with the borehole details to get the Site Short Name and Easting/Northing
    WL_df = pd.merge(WL_df, borehole_df[['Site Ref', 'Site Short Name', 'Easting', 'Northing', 'Elevation_DEM', 'zobs']], on='Site Ref', how='left')

    # Figure out whether the 'static water level' is the depth to water
    # Add m_AHD where there is SLE data and static water level data
    conflict_mask = WL_df['Static water level (m)'].notna() & WL_df['Water level (SLE) (m)'].notna()
    if conflict_mask.any():
        print(f"Warning: {conflict_mask.sum()} rows have both Static WL and SLE WL — dropping these rows.")
        WL_df = WL_df[~conflict_mask].copy()
    def derive_wl(row):
        if pd.notna(row['Water level (SLE) (m)']):
            return row['Elevation_DEM'] - row['Water level (SLE) (m)']
        elif pd.notna(row['Static water level (m)']) and pd.notna(row['Elevation_DEM']):
            return row['Elevation_DEM'] - row['Static water level (m)']
        else:
            return pd.NA
    WL_df['Derived WL (mAHD)'] = WL_df.apply(derive_wl, axis=1)

    #plot and label all the bores on a map
    fig, ax = plt.subplots(figsize=(8, 8))
    model_boundary = gpd.read_file(geomodel_shapefile)
    model_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    unique_bores = WL_df[['Site Ref', 'Easting', 'Northing']].drop_duplicates()
    geometry = [Point(xy) for xy in zip(unique_bores['Easting'], unique_bores['Northing'])]
    bores_with_water_levels = gpd.GeoDataFrame(unique_bores, geometry=geometry, crs=model_boundary.crs)
    bores_with_water_levels.plot(ax=ax, color='red', markersize=25, label='Filtered Bores')
    ax.set_title("Filtered Bore Locations Within Model Boundary")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    output_path = '../data/data_waterlevels/obs/05_Water_levels_clean.xlsx'
    WL_df.to_excel(output_path, index=False)

def seasonal_flows(geomodel_shapefile):
    WL_df = pd.read_excel('../data/data_waterlevels/obs/05_Water_levels_clean.xlsx')
    model_boundary = gpd.read_file(geomodel_shapefile)

    # Create bins for the seasons (dry versus wet season)
    WL_df['Season'] = WL_df['Collect Month'].apply(lambda x: 'Wet' if x in [5, 6, 7, 8, 9, 10] else 'Dry')
    WL_df['Sample timeframe'] = WL_df['Collect Year'].astype(str) + '_' + WL_df['Season'] #this is now Year_Season

    water_level_min = WL_df['Derived WL (mAHD)'].min()
    water_level_max = WL_df['Derived WL (mAHD)'].max()

    # For each Year_Season that has water levels from at least 10 bores, create a groundwater contour map
    # Loop through each Year_Season group
    for timeframe, group in WL_df.groupby('Sample timeframe'):
        group = group.dropna(subset=['Easting', 'Northing', 'Derived WL (mAHD)', 'Collect Date']) #get rid of any rows that are missing important data 
        group = group.sort_values('Collect Date').groupby('Site Ref', as_index=False).last() #take the most recent entry in that timeframe for each bore
        if group['Site Ref'].nunique() < 10:
            continue  # Skip if fewer than 10 unique bores

        print(f"Plotting contours for: {timeframe} with {group['Site Ref'].nunique()} bores")

        # Prepare grid for interpolation
        x = group['Easting'].values
        y = group['Northing'].values
        z = group['Derived WL (mAHD)'].values

        # Create a meshgrid over model extent
        xmin, ymin, xmax, ymax = model_boundary.total_bounds
        xi = np.linspace(xmin, xmax, 200)
        yi = np.linspace(ymin, ymax, 200)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate WLs over the grid
        zi = griddata((x, y), z, (xi, yi), method='linear')
        #zi[np.isnan(zi)] = griddata((x, y), z, (xi, yi), method='nearest')[np.isnan(zi)] # Fill NaNs with nearest values, use if above is 'cubic'

        # Plot contours
        fig, ax = plt.subplots(figsize=(10, 8))
        model_boundary.boundary.plot(ax=ax, edgecolor='black')
        zi_masked = np.ma.masked_invalid(zi)
        cs = ax.contourf(xi, yi, zi_masked, levels=np.linspace(water_level_min, water_level_max, 20), cmap='viridis')
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label('Derived WL (mAHD)')

        # Overlay bore locations
        ax.scatter(x, y, color='red', s=10, label='Bores')
        texts = []
        for _, row in group.iterrows():
            texts.append(ax.text(row['Easting'] + 5, row['Northing'] + 5, row['Site Short Name'],
                         fontsize=6, color='black', alpha=0.7))
        adjust_text(texts, ax=ax, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle='->', color='gray'))
        for _, row in group.iterrows():
            ax.text(row['Easting'] + 5, row['Northing'] + 5, row['Site Short Name'],
                fontsize=6, color='black', alpha=0.7)
        ax.set_title(f"Groundwater Contours – {timeframe}")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()

        output_dir = '../data/data_waterlevels/obs/seasonal_flows'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"Groundwater_Contours_{timeframe}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
    
    output_path = '../data/data_waterlevels/obs/06_Water_levels_over_time.xlsx'
    WL_df.to_excel(output_path, index=False)
  
def make_steady_state_gdf(df, geomodel, mesh, spatial):

    # as above, group by the sample timeframe (Year_Season) and where there are more than 10 bores of data available
    timeframe_counts = df.groupby('Sample timeframe')['Site Ref'].nunique()
    eligible_timeframes = timeframe_counts[timeframe_counts >= 10].sort_index()
    if eligible_timeframes.empty:
        print("No timeframe with at least 10 bores found.")
        return None
    
    #find the earliest timeframe where the condition of >10 bores sampled within the season is met
    earliest_timeframe = eligible_timeframes.index[0]
    print(f"Using earliest timeframe with ≥10 bores: {earliest_timeframe}")
    steady_state_df = df[df['Sample timeframe'] == earliest_timeframe].copy() #save this earliest timeframe as the steady state dataframe
    steady_state_df = steady_state_df.sort_values('Collect Date').groupby('Site Ref', as_index=False).last() #if there are multiple entries for a timeframe, this will take the latest entry

    gdf = gpd.GeoDataFrame(steady_state_df, geometry=gpd.points_from_xy(steady_state_df.Easting, steady_state_df.Northing), crs=spatial.epsg)

    mask = gdf.geometry.within(spatial.model_boundary_poly)
    if not mask.all():
        print("The following geometries are NOT within the polygon:")
        print(gdf[~mask])
    else:
        print("All geometries are within the polygon.")
    gdf = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points outside model

    gdf = gdf[gdf['zobs'] != np.nan] # Don't include obs with no zobs
    gdf = gdf[gdf['zobs'] > geomodel.z0] # Don't include obs deeper than flow model bottom
    gdf = gdf.reset_index(drop=True)

    # Perform the intersection
    gdf['icpl'] = gdf.apply(lambda row: mesh.vgrid.intersect(row.Easting,row.Northing), axis=1)
    gdf['ground'] = gdf.apply(lambda row: geomodel.top_geo[row.icpl], axis=1)
    gdf['model_bottom'] = gdf.apply(lambda row: geomodel.botm[-1, row.icpl], axis=1)
    gdf['zobs-bot'] = gdf.apply(lambda row: row['zobs'] - row['model_bottom'], axis=1)

    for idx, row in gdf.iterrows():
        result = row['zobs'] - row['model_bottom']
        if result < 0:
            print(f"Bore {row['id']} has a zobs elevation below model bottom by: {result} m, removing from obs list")
    print(gdf)

    gdf = gdf[gdf['zobs-bot'] > 0] # filters out observations that are below the model bottom

    gdf['cell_disv'] = gdf.apply(lambda row: utils.xyz_to_disvcell(geomodel, row.Easting, row.Northing, row.zobs), axis=1)
    gdf['cell_disu'] = gdf.apply(lambda row: utils.disvcell_to_disucell(geomodel, row['cell_disv']), axis=1)  

    gdf['(lay,icpl)'] = gdf.apply(lambda row: utils.disvcell_to_layicpl(geomodel, row['cell_disv']), axis = 1)
    gdf['lay']        = gdf.apply(lambda row: row['(lay,icpl)'][0], axis = 1)
    gdf['icpl']       = gdf.apply(lambda row: row['(lay,icpl)'][1], axis = 1)
    gdf['obscell_xy'] = gdf['icpl'].apply(lambda icpl: (mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]))
    gdf['obscell_z']  = gdf.apply(lambda row: geomodel.zc[row['lay'], row['icpl']], axis=1)
    gdf['obs_zpillar']  = gdf.apply(lambda row: geomodel.zc[:, row['icpl']], axis=1)
    gdf['geolay']       = gdf.apply(lambda row: math.floor(row['lay']/geomodel.nls), axis = 1) # model layer to geolayer

    gdf.rename(columns={'Easting': 'x', 'Northing': 'y', 'zobs': 'z', 'ID' : 'id'}, inplace=True) # to be consistent when creating obs_rec array

    # Make sure no pinched out observations
    if -1 in gdf['cell_disu'].values:
        print('Warning: some observations are pinched out. Check the model and data.')
        print('Number of pinched out observations: ', len(gdf[gdf['cell_disu'] == -1]))
        gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf 

def plot_hydrograph(df_boredetails, spatial):
    # Import all water level observations from WIR
    df_WL = pd.read_excel('../data/data_waterlevels/Otorowiri_water_levels.xlsx')
    #append the df to have the 'Site Short Name' column, which matches the Site Ref in df_boreids   
    df_WL = pd.merge(df_WL, df_boredetails[['Site Ref', 'Site Short Name', 'Easting', 'Northing', 'GL mAHD']], on='Site Ref', how='left')

    # Plot water levels from the filtered bores
    parmelia_boreids = df_boredetails['Site Short Name'].unique()
    parmelia_WL_df = df_WL[df_WL['Site Short Name'].isin(parmelia_boreids)]
    parmelia_WL_df['geometry'] =  parmelia_WL_df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
    gdf = gpd.GeoDataFrame(parmelia_WL_df, geometry='geometry', crs=spatial.epsg)
    if gdf.empty:
        print("GeoDataFrame is empty.")
    else:
        print("GeoDataFrame contains data.")

    # Filter out points outside model boundary
    #parmelia_WL_df = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points within the polygon if any snuck in
    parmelia_bore_refs = parmelia_WL_df['Site Ref'].unique()
    parmelia_bores = parmelia_WL_df['Site Short Name'].unique()
    print(f"Unique Parmelia bore IDs: {parmelia_bores}")
    print(f"Unique Parmelia bore refs: {parmelia_bore_refs}")

    #Cleaning up the data pre-hydrograph
    #column_names = parmelia_WL_df.columns.unique()
    parmelia_WL_df = parmelia_WL_df[parmelia_WL_df['Variable Type'] == 'Water level (discrete)'] # Filter for Water Level variable type (eliminates pump test data)
    parmelia_WL_df = parmelia_WL_df[(parmelia_WL_df['Variable Name'] == 'Water level (AHD) (m)' )]
                                    #| (parmelia_WL_df['Variable Name'] == 'Static water level (m)')] # Filter for Water Level variable name (eliminates water quality data)
    parmelia_WL_df['Collect Date'] = pd.to_datetime(parmelia_WL_df['Collect Date'], errors='coerce') # Ensure Collect Date is datetime
    parmelia_WL_df = parmelia_WL_df.dropna(subset=['Collect Date', 'Reading Value']) # Remove rows with NaNs in the Reading Value or Collect Date columns
    parmelia_WL_df['Reading Value'] = ( # Remove the ~ character from the Reading Value column and convert to numeric
        parmelia_WL_df['Reading Value']
        .astype(str)                     # ensure it's all strings for replacement
        .str.replace('~', '', regex=False)  # remove ~
        .str.strip())                     # remove any leading/trailing whitespace
    parmelia_WL_df['Reading Value'] = pd.to_numeric(parmelia_WL_df['Reading Value'], errors='coerce')
    parmelia_WL_df = parmelia_WL_df.rename(columns={'Reading Value': 'Water level (m AHD)'}) # All values should now be in AHD so column renamed

    ##Plot all hydrographs on different pages
    bores_per_page = 12
    total_bores = len(parmelia_bores)
    pages = int(np.ceil(total_bores / bores_per_page))

    for page in range(pages):
        fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=False, sharey=False)
        axes = axes.flatten()

        start = page * bores_per_page
        end = min(start + bores_per_page, total_bores)
        selected_bores = parmelia_bores[start:end]

        for i, bore in enumerate(selected_bores):
            ax = axes[i]
            df = parmelia_WL_df[parmelia_WL_df['Site Short Name'] == bore].dropna(subset=['Collect Date', 'Water level (m AHD)'])
            if not df.empty:
                ax.plot(df['Collect Date'], df['Water level (m AHD)'], '-o', label=bore)
                ax.set_title(bore, fontsize=10)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_title(f"{bore} (No Data)", fontsize=10)
        for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle(f"Hydrographs Page {page + 1}", fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.92)
        plt.show()

def make_obs_gdf(df, geomodel, mesh, spatial):
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=spatial.epsg)

    mask = gdf.geometry.within(spatial.model_boundary_poly)
    if not mask.all():
        print("The following geometries are NOT within the polygon:")
        print(gdf[~mask])
    else:
        print("All geometries are within the polygon.")
    gdf = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points outside model

    gdf = gdf[gdf['zobs'] != np.nan] # Don't include obs with no zobs
    gdf = gdf[gdf['zobs'] > geomodel.z0] # Don't include obs deeper than flow model bottom
    gdf = gdf.reset_index(drop=True)

    # Perform the intersection
    gdf['icpl'] = gdf.apply(lambda row: mesh.vgrid.intersect(row.Easting,row.Northing), axis=1)
    gdf['ground'] = gdf.apply(lambda row: geomodel.top_geo[row.icpl], axis=1)
    gdf['model_bottom'] = gdf.apply(lambda row: geomodel.botm[-1, row.icpl], axis=1)
    gdf['zobs-bot'] = gdf.apply(lambda row: row['zobs'] - row['model_bottom'], axis=1)

    for idx, row in gdf.iterrows():
        result = row['zobs'] - row['model_bottom']
        if result < 0:
            print(f"Bore {row['id']} has a zobs elevation below model bottom by: {result} m, removing from obs list")

    gdf = gdf[gdf['zobs-bot'] > 0] # filters out observations that are below the model bottom

    gdf['cell_disv'] = gdf.apply(lambda row: utils.xyz_to_disvcell(geomodel, row.Easting, row.Northing, row.zobs), axis=1)
    gdf['cell_disu'] = gdf.apply(lambda row: utils.disvcell_to_disucell(geomodel, row['cell_disv']), axis=1)  

    gdf['(lay,icpl)'] = gdf.apply(lambda row: utils.disvcell_to_layicpl(geomodel, row['cell_disv']), axis = 1)
    gdf['lay']        = gdf.apply(lambda row: row['(lay,icpl)'][0], axis = 1)
    gdf['icpl']       = gdf.apply(lambda row: row['(lay,icpl)'][1], axis = 1)
    gdf['obscell_xy'] = gdf['icpl'].apply(lambda icpl: (mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]))
    gdf['obscell_z']  = gdf.apply(lambda row: geomodel.zc[row['lay'], row['icpl']], axis=1)
    gdf['obs_zpillar']  = gdf.apply(lambda row: geomodel.zc[:, row['icpl']], axis=1)
    gdf['geolay']       = gdf.apply(lambda row: math.floor(row['lay']/geomodel.nls), axis = 1) # model layer to geolayer

    gdf.rename(columns={'Easting': 'x', 'Northing': 'y', 'zobs': 'z', 'ID' : 'id'}, inplace=True) # to be consistent when creating obs_rec array

    # Make sure no pinched out observations
    if -1 in gdf['cell_disu'].values:
        print('Warning: some observations are pinched out. Check the model and data.')
        print('Number of pinched out observations: ', len(gdf[gdf['cell_disu'] == -1]))
        gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf



'''
#filter all the raw WIR data to narrow down the bores screened in the lithology of interest
def prefilter_data():
    # Import all bore data available, and filter to include only Parmelia
    aquifers = pd.read_excel('../data/data_waterlevels/Otorowiri_site_details.xlsx', sheet_name='Aquifers')
    df = aquifers[
        (aquifers['Aquifer Name'] == 'Perth-Parmelia') | 
        (aquifers['Aquifer Name'] == 'Perth-Leederville-Parmelia')]
    # Add Site Short Name, Easting, Northing to df
    details = pd.read_excel('../data/data_waterlevels/Otorowiri_site_details.xlsx', sheet_name='Site Details')
    df = pd.merge(df, details[['Site Ref', 'Site Short Name', 'Easting', 'Northing']], on='Site Ref', how='left')
        
    # Import screen details
    casing = pd.read_excel('../data/data_waterlevels/Otorowiri_site_details.xlsx', sheet_name='Casing')
    casing = casing[casing['Element'] == 'Inlet (screen)']

    # Identify wells with multiple screen intervals and remove from df
    duplicates = casing[casing.duplicated('Site Ref', keep=False)]
    duplicates = pd.merge(duplicates, details[['Site Ref', 'Site Short Name']], on='Site Ref', how='left')
    duplicate_site_refs = duplicates['Site Ref'].unique()
    df_boreids = df[~df['Site Ref'].isin(duplicate_site_refs)]

    # Add screened interval to df
    df_boreids = pd.merge(df_boreids, casing[['Site Ref', 'From (mbGL)', 'To (mbGL)', 'Inside Dia. (mm)']], on='Site Ref', how='left')

    # Export desired obs bores as csv
    df_boreids.to_csv('../data/data_waterlevels/obs/cleaned_obs_bores.csv', index=False)
    parmelia_boreids = df_boreids['Site Short Name'].unique()
    print(f"Unique Parmelia bore IDs: {parmelia_boreids}")

    # Use df_cleaned for data request on Water Information Reporting
    return df_boreids
    #print (f"Unique Parmelia bore IDs: {df_boreids.columns}")

def assemble_clean_data(df_boreids): #df_boreids is used here as df_filtered
    # Now we have get extra bore details from WIR, we can add groundlevel and well screens to our dataframe

    # Add GL to main df
    measurements = pd.read_excel('../data/data_waterlevels/Otorowiri_site_details.xlsx', sheet_name='Depth Measurement Points')
    ground_level = measurements[measurements['Measurement Point Type'] == 'Ground level'].drop_duplicates(subset=['Site Ref'])
    toc = measurements[measurements['Measurement Point Type'] == 'Top of casing'].drop_duplicates(subset=['Site Ref'])
    mp = measurements[measurements['Measurement Point Type'] == 'Measurement Point'].drop_duplicates(subset=['Site Ref'])

    # First use groundlevel measurement
    df = pd.merge(df_boreids, ground_level[['Site Ref', 'Measurement Point Type', 'Elevation (m as per Datum Plane)']], on='Site Ref', how='left')
    df = df.rename(columns={'Elevation (m as per Datum Plane)': 'GL mAHD'})
    df = df.rename(columns={'Measurement Point Type': 'GL source'})

    # If no groundlevel, then Top of Casing - 700mm for ground level
    df = pd.merge(df, toc[['Site Ref', 'Measurement Point Type', 'Elevation (m as per Datum Plane)']], on='Site Ref', how='left')
    df.loc[df['GL mAHD'].isna(), 'GL source'] = 'Top of casing'
    df.loc[df['GL mAHD'].isna(), 'GL mAHD'] = df['Elevation (m as per Datum Plane)'] - 0.7
    df = df.drop(columns=['Elevation (m as per Datum Plane)'])
    df = df.drop(columns=['Measurement Point Type'])

    # If no groundlevel or Top of Casing, use measurement point
    df = pd.merge(df, mp[['Site Ref', 'Measurement Point Type', 'Elevation (m as per Datum Plane)']], on='Site Ref', how='left')
    df.loc[df['GL mAHD'].isna(), 'GL source'] = 'Measurement Point'
    df.loc[df['GL mAHD'].isna(), 'GL mAHD'] = df['Elevation (m as per Datum Plane)'] - 0.7
    df = df.drop(columns=['Elevation (m as per Datum Plane)'])
    #df = df.drop(columns=['Measurement Point Type'])
    df = df.drop(columns=['Comments'])
    df = df.drop(columns=['Depth From/To (mbGL)'])

    # Get top and bottom of screen in mAHD
    df['Screen top'] = df['GL mAHD'] - df['From (mbGL)']
    df['Screen bot'] = df['GL mAHD'] - df['To (mbGL)']
    df['zobs'] = df['Screen bot'] + (df['Screen top'] - df['Screen bot'])/2
        
    #df = df.rename(columns={'Site Short Name': 'ID'})
    df_boredetails = df
    df_boredetails.to_csv('../data/data_waterlevels/obs/cleaned_obs_bores.csv', index=False) #add information to the csv
    print_bore_details = df_boredetails.columns
    print(f"Bore detail columns are: {print_bore_details}")

    return df_boredetails'''