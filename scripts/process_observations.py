import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import loopflopy.utils as utils
import flopy
import math
import os

def convert_static_to_ahd(row):
    key = (row['Site Short Name'], row['Collect Date'])
    if (
        row['Variable Name'] == 'Static water level (m)' and
        key not in ahd_index and
        pd.notna(row['GL mAHD']) and
        pd.notna(row['Reading Value'])
    ):
        return row['GL mAHD'] - row['Reading Value']
    else:
        return pd.NA

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

    '''fig, ax = plt.subplots(figsize=(8, 8))
    model_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    bores_in_model.plot(ax=ax, color='red', markersize=25, label='Filtered Bores')
    ax.set_title("Filtered Bore Locations Within Model Boundary")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''
    
    return bores_in_model

def assign_aquifer(bores_in_model):
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

    # Add the Elevations where they exist
    elevations = pd.read_excel('../data/data_waterlevels/model_area_raw/174416/Site - All Site Details (Excel).xlsx', sheet_name='Depth Measurement Points')
    elevations = elevations[elevations['Measurement Point Type'] == 'Ground level'].drop_duplicates(subset=['Site Ref'])
    elevations['Site Ref'] = elevations['Site Ref'].astype(str)
    df = df.merge(
        elevations[['Site Ref', 'Elevation (m as per Datum Plane)']],
        on='Site Ref',
        how='left')
    df = df.rename(columns={'Elevation (m as per Datum Plane)': 'Elevation_DWER'})

    # Add the drilled depths
    drilled_depths = pd.read_excel('../data/data_waterlevels/model_area_raw/174416/Site - All Site Details (Excel).xlsx', sheet_name='Borehole Information')
    drilled_depths['Site Ref'] = drilled_depths['Site Ref'].astype(str)
    df = df.merge(
        drilled_depths[['Site Ref', 'Drilled Depth (m)']],
        on='Site Ref',
        how='left')
    df = df.rename(columns={'Drilled Depth (m)': 'Drilled Depth'})

    output_path = "../data/data_waterlevels/obs/01_All_Groundwater_bores.xlsx"
    df.to_excel(output_path, index=False)


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

    return df_boredetails

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

def transient_timestamps(parmelia_WL_df):
    # Use the data that has been filtered for the hydrographs, which should now have all AHD values
    #parmelia_WL_df

    # Create bins for the seasons (dry versus wet season)
    parmelia_WL_df['Season'] = parmelia_WL_df['Collect Month'].apply(lambda x: 'Wet' if x in [5, 6, 7, 8, 9, 10] else 'Dry')
    parmelia_WL_df['Sample timeframe'] = parmelia_WL_df['Collect Year'].astype(str) + '_' + parmelia_WL_df['Season'] #this is now Year_Season

    # Finding the number of bores available for each sample timeframe
    bore_counts = (
        parmelia_WL_df
        .dropna(subset=['Water level (m AHD)'])  # ensure data exists
        .groupby('Sample timeframe')['Site Short Name']
        .nunique()
        .reset_index(name='Num Unique Bores'))
    #print(bore_counts.sort_values('Sample timeframe'))


    '''
    sample_timeframe_counts = parmelia_WL_df['Sample timeframe'].value_counts() #counting the sample timeframe
    print(sample_timeframe_counts)
    parmelia_WL_df['Sample timeframe years'] = parmelia_WL_df['Sample timeframe'].str.extract(r'(\d{4})') #extracting the year from the Sample timeframe
    selected_years = []
    last_year = None
    for _, row in df.iterrows():
        year = row['Collect Year']
    if last_year is None or (year - last_year) >= 5:
        selected_years.append(row['Filter date'])
        last_year = year
    if len(selected) == 5:
        break

    print("Selected filter dates (at least 5 years apart):")
    print(selected)'''

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
    gdf['obscell_z']  = gdf.apply(lambda row: geomodel.zcenters[row['lay'], row['icpl']], axis=1)
    gdf['obs_zpillar']  = gdf.apply(lambda row: geomodel.zcenters[:, row['icpl']], axis=1)
    gdf['geolay']       = gdf.apply(lambda row: math.floor(row['lay']/geomodel.nls), axis = 1) # model layer to geolayer

    gdf.rename(columns={'Easting': 'x', 'Northing': 'y', 'zobs': 'z', 'ID' : 'id'}, inplace=True) # to be consistent when creating obs_rec array

    # Make sure no pinched out observations
    if -1 in gdf['cell_disu'].values:
        print('Warning: some observations are pinched out. Check the model and data.')
        print('Number of pinched out observations: ', len(gdf[gdf['cell_disu'] == -1]))
        gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf