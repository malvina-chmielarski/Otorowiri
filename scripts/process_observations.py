import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
import geopandas as gpd

#filter all the raw WIR data to narrow down the bores screened in the lithology of interest
def prefilter_data():
    # Import observation bores screened in Parmelia, and filter to include only Parmelia
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

# Plot hydrographs of all the bores from the pre-filtering selection
def plot_hydrograph(df_boreids, spatial): #df_boreids is used here as the data filtered to the Parmelia aquifer

    # Import all water level observations from WIR
    df_WL = pd.read_excel('../data/data_waterlevels/Otorowiri_water_levels.xlsx')
    #append the df to have the 'Site Short Name' column, which matches the Site Ref in df_boreids   
    df_WL = pd.merge(df_WL, df_boreids[['Site Ref', 'Site Short Name']], on='Site Ref', how='left')

    # Plot water levels from the filtered bores
    parmelia_boreids = df_boreids['Site Short Name'].unique()
    parmelia_WL_df = df_WL[df_WL['Site Short Name'].isin(parmelia_boreids)]
    parmelia_WL_df['geometry'] =  parmelia_WL_df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
    gdf = gpd.GeoDataFrame(parmelia_WL_df, geometry='geometry', crs=spatial.epsg)

    # Filter out points outside model boundary
    parmelia_WL_df = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points within the polygon if any snuck in
    parmelia_bores = parmelia_WL_df['Site Short Name'].unique()

    for bore in parmelia_bores:
        df = parmelia_WL_df[parmelia_WL_df['Site Ref'] == bore]
        plt.plot(df['Collect Date'], df['Reading Value'], label = df['ID'].iloc[0])
    plt.legend(loc = 'upper left',fontsize = 'small', markerscale=0.5)
    plt.show()

# Now we have get extra bore details from WIR, we can add groundlevel and well screens to our dataframe
def assemble_clean_data(df_filtered): #df_boreids is used here as df_filtered

    # Add GL to main df
    measurements = pd.read_excel('../data/data_waterlevels/Otorowiri_site_details.xlsx', sheet_name='Depth Measurement Points')
    ground_level = measurements[measurements['Measurement Point Type'] == 'Ground level'].drop_duplicates(subset=['Site Ref'])
    toc = measurements[measurements['Measurement Point Type'] == 'Top of casing'].drop_duplicates(subset=['Site Ref'])
    mp = measurements[measurements['Measurement Point Type'] == 'Measurement Point'].drop_duplicates(subset=['Site Ref'])

    # First use groundlevel measurement
    df = pd.merge(df_filtered, ground_level[['Site Ref', 'Measurement Point Type', 'Elevation (m as per Datum Plane)']], on='Site Ref', how='left')
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
    df = df.drop(columns=['Measurement Point Type'])
    df = df.drop(columns=['Comments'])
    df = df.drop(columns=['Depth From/To (mbGL)'])

    # Get top and bottom of screen in mAHD
    df['Screen top'] = df['GL mAHD'] - df['From (mbGL)']
    df['Screen bot'] = df['GL mAHD'] - df['To (mbGL)']
    df['zobs'] = df['Screen bot'] + (df['Screen top'] - df['Screen bot'])/2
    
    df = df.rename(columns={'Site Short Name': 'ID'})
    df_boredetails = df
    print_bore_details = df_boredetails.columns
    print(f"Bore detail columns are: {print_bore_details}")

    return df_boredetails

# Now we have bore details, we can add Water Level observations to our dataframe
'''def add_WL_obs(df_boredetails):
    # Import water level data from WIR
    WL = pd.read_excel('../data/data_waterlevels/Otorowiri_water_levels.xlsx')

    # Create bins for the seasons (dry versus wet season)
    WL['Season'] = WL['Collect Month'].apply(lambda x: 'Wet' if x in [5, 6, 7, 8, 9, 10] else 'Dry')
    WL['Sample timeframe'] = df['Collect year'].astype(str) + '_' + df['Season'] #this is now Year_Season
    
    # Select the Sample timeframes which have the most data
    sample_timeframe_counts = WL['Sample timeframe'].value_counts() #counting the sample timeframe
    print(sample_timeframe_counts)
    WL['Sample timeframe years'] = WL['Sample timeframe'].str.extract(r'(\d{4})') #extracting the year from the Sample timeframe
    selected_years = []
    last_year = None
    for _, row in df.iterrows():
        year = row['Year']
    if last_year is None or (year - last_year) >= 5:
        selected.append(row['Filter date'])
        last_year = year
    if len(selected) == 5:
        break

    print("Selected filter dates (at least 5 years apart):")
    print(selected)

    print (dskhfkjsdhf)'''

'''    # Filter based on Sample timeframe counts
    WL = WL[WL['Collect Date'] > '2005-01-01']
    start_date = '2005-01-01'
    df_filtered = WL[
                    (WL['Collect Date'] >= start_date) & 
                    (WL['Variable Name'] == 'Water level (AHD) (m)')
                    ]

    # Add ID, screened aquifer to main df
    df_obs = pd.merge(df_filtered, df_boredetails, on='Site Ref', how='left')
    df_obs = df_obs[['ID', 'Site Ref', 'Easting', 'Northing', 'Collect Date', 'Aquifer Name', 'Reading Value', 'GL mAHD', 'Screen top', 'Screen bot']]

    # Filter out points outside model boundary for df_obs only'''



'''df_obs['geometry'] = df_obs.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
    gdf = gpd.GeoDataFrame(df_obs, geometry='geometry', crs=spatial.epsg)    
    df_obs= gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points within the polygon
    count = len(df_obs) # Count the number of points within the polygon
    print(f"Number of bores within model boundary: {count}")'''

'''
    df_boredetails['min_WL'] = None
    df_boredetails['max_WL'] = None
    df_boredetails['mean_WL'] = None

    bores = df_obs['Site Ref'].unique()
    for bore in bores:
        df = df_obs[df_obs['Site Ref'] == bore]
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'min_WL'] = df['Reading Value'].min()
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'max_WL'] = df['Reading Value'].max()
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'mean_WL'] = df['Reading Value'].mean()
    
    # Filter out known duplicates

    df_obs = df_obs[df_obs['ID'] != 'GG11 (I)'] # This is at the same location as AM4A so remove
    df_obs = df_obs[df_obs['ID'] != 'GG11 (O)'] # This is at the same location as AM4A so remove
    
    return (df_boredetails, df_obs)'''