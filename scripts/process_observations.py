import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
import geopandas as gpd

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
    df = df.drop(columns=['Measurement Point Type'])
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
    column_names = parmelia_WL_df.columns.unique()
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
                ax.plot(df['Collect Date'], df['Water level (m AHD)'], label=bore)
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