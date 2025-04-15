import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
<<<<<<< HEAD
import geopandas as gpd
=======
>>>>>>> c4df708f44e71f7bdc31e4e1f853915c2183824c

def prefilter_data():
    # Import observation bores screened in Leederville or Yarragadee, and filter ti unclude only Leederville and Yarragadee
    aquifers = pd.read_excel('../data/data_dwer/obs/obs_bores.xlsx', sheet_name='Aquifers')
    df = aquifers[(aquifers['Aquifer Name'] == 'Perth-Yarragadee North') | (aquifers['Aquifer Name'] == 'Perth-Leederville')]

    # Add Site Short Name, Easting, Northing to df
    details = pd.read_excel('../data/data_dwer/obs/obs_bores.xlsx', sheet_name='Site Details')
    df = pd.merge(df, details[['Site Ref', 'Site Short Name', 'Easting', 'Northing']], on='Site Ref', how='left')
    
    # Import screen details
    casing = pd.read_excel('../data/data_dwer/obs/obs_bores.xlsx', sheet_name='Casing')
    casing = casing[casing['Element'] == 'Inlet (screen)']

    # Identify wells with multiple screen intervals and remove from df
    duplicates = casing[casing.duplicated('Site Ref', keep=False)]
    duplicates = pd.merge(duplicates, details[['Site Ref', 'Site Short Name']], on='Site Ref', how='left')
    duplicate_site_refs = duplicates['Site Ref'].unique()
    df_boreids = df[~df['Site Ref'].isin(duplicate_site_refs)]

    # Add screened interval to df
    df_boreids = pd.merge(df_boreids, casing[['Site Ref', 'From (mbGL)', 'To (mbGL)', 'Inside Dia. (mm)']], on='Site Ref', how='left')

    # Export desired obs bores as csv
    df_boreids.to_csv('../data/data_dwer/obs/cleaned_obs_bores.csv', index=False)

    # Use df_cleaned for data request on Water Information Reporting
    return df_boreids

# Now we have git extra bore details from WIR, we can groundlevel and well screens to our dataframe
def assemble_clean_data(df_filtered):
<<<<<<< HEAD

=======
>>>>>>> c4df708f44e71f7bdc31e4e1f853915c2183824c
    # Add GL to main df
    measurements = pd.read_excel('../data/data_dwer/obs/obs_bores.xlsx', sheet_name='Depth Measurement Points')
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

    return df_boredetails

# Now we have bore details, we can add Water Level observations to our dataframe
def add_WL_obs(df_boredetails):
    # Import water level data from WIR
    WL = pd.read_excel('../data/data_dwer/obs/170999/WaterLevelsDiscreteForSiteFlatFile.xlsx')

    # Filter based on date and variable name
    WL = WL[WL['Collect Date'] > '2005-01-01']
    start_date = '2005-01-01'
    df_filtered = WL[
                    (WL['Collect Date'] >= start_date) & 
                    (WL['Variable Name'] == 'Water level (AHD) (m)')
                    ]

    # Add ID, screened aquifer to main df
    df_obs = pd.merge(df_filtered, df_boredetails, on='Site Ref', how='left')
<<<<<<< HEAD
    df_obs = df_obs[['ID', 'Site Ref', 'Easting', 'Northing', 'Collect Date', 'Aquifer Name', 'Reading Value', 'GL mAHD', 'Screen top', 'Screen bot']]

    # Filter out points outside model boundary for df_obs only
    '''df_obs['geometry'] = df_obs.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
    gdf = gpd.GeoDataFrame(df_obs, geometry='geometry', crs=spatial.epsg)    
    df_obs= gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points within the polygon
    count = len(df_obs) # Count the number of points within the polygon
    print(f"Number of bores within model boundary: {count}")'''
=======
    df_obs = df_obs[['ID', 'Site Ref', 'Collect Date', 'Aquifer Name', 'Reading Value', 'GL mAHD', 'Screen top', 'Screen bot']]
>>>>>>> c4df708f44e71f7bdc31e4e1f853915c2183824c

    df_boredetails['min_WL'] = None
    df_boredetails['max_WL'] = None
    df_boredetails['mean_WL'] = None

    bores = df_obs['Site Ref'].unique()
    for bore in bores:
        df = df_obs[df_obs['Site Ref'] == bore]
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'min_WL'] = df['Reading Value'].min()
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'max_WL'] = df['Reading Value'].max()
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'mean_WL'] = df['Reading Value'].mean()
<<<<<<< HEAD
    
    # Filter out known duplicates

    df_obs = df_obs[df_obs['ID'] != 'GG11 (I)'] # This is at the same location as AM4A so remove
    df_obs = df_obs[df_obs['ID'] != 'GG11 (O)'] # This is at the same location as AM4A so remove
    
    return (df_boredetails, df_obs)

def plot_leederville_hydrographs(df_obs, spatial):
 
    # Plot water levels - Leederville
    leed_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Leederville']
    leed_df['geometry'] = leed_df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
    gdf = gpd.GeoDataFrame(leed_df, geometry='geometry', crs=spatial.epsg)

    # Filter out points outside model boundary
    leed_df = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points within the polygon
    count = len(set(leed_df.ID.tolist())) # Count the number of points within the polygon
    print(f"Number of Leerville bores within model boundary: {count}")

=======

    return (df_boredetails, df_obs)

def plot_leederville_hydrographs(df_obs):
    # Plot water levels - Leederville
    leed_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Leederville']
>>>>>>> c4df708f44e71f7bdc31e4e1f853915c2183824c
    leed_bores = leed_df['Site Ref'].unique()

    for bore in leed_bores:
        df = leed_df[leed_df['Site Ref'] == bore]
        plt.plot(df['Collect Date'], df['Reading Value'], label = df['ID'].iloc[0])
    plt.legend(loc = 'upper left',fontsize = 'small', markerscale=0.5)
<<<<<<< HEAD
    plt.show()

def plot_yarragadee_hydrographs(df_obs, spatial):
    # Plot water levels - Yarragadee
    yarr_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Yarragadee North']
    yarr_df['geometry'] = yarr_df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
    gdf = gpd.GeoDataFrame(yarr_df, geometry='geometry', crs=spatial.epsg)

    # Filter out points outside model boundary
    yarr_df = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points within the polygon
    count = len(set(yarr_df.ID.tolist())) # Count the number of points within the polygon
    print(f"Number of Yarragadee bores within model boundary: {count}")

=======

def plot_yarragadee_hydrographs(df_obs):
    # Plot water levels - Yarragadee
    yarr_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Yarragadee North']
>>>>>>> c4df708f44e71f7bdc31e4e1f853915c2183824c
    yarr_bores = yarr_df['Site Ref'].unique()

    for bore in yarr_bores:
        df = yarr_df[yarr_df['Site Ref'] == bore]
        plt.plot(df['Collect Date'], df['Reading Value'], label = df['ID'].iloc[0])
    plt.legend(loc = 'upper left',fontsize = 'small', markerscale=0.5)
