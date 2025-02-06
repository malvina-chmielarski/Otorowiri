import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt

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
    filtered_df = WL[
                    (WL['Collect Date'] >= start_date) & 
                    (WL['Variable Name'] == 'Water level (AHD) (m)')
                    ]

    # Add ID, screened aquifer to main df
    filtered_df = pd.merge(filtered_df, df_boredetails, on='Site Ref', how='left')
    filtered_df = filtered_df[['ID', 'Site Ref', 'Collect Date', 'Aquifer Name', 'Reading Value', 'GL mAHD', 'Screen top', 'Screen bot']]

    return filtered_df

def plot_leederville_hydrographs(df_obs):
    # Plot water levels - Leederville
    leed_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Leederville']
    leed_bores = leed_df['Site Ref'].unique()

    for bore in leed_bores:
        df = leed_df[leed_df['Site Ref'] == bore]
        plt.plot(df['Collect Date'], df['Reading Value'], label = df['ID'].iloc[0])
    plt.legend(loc = 'upper left',fontsize = 'small', markerscale=0.5)

def plot_yarragadee_hydrographs(df_obs):
    # Plot water levels - Yarragadee
    yarr_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Yarragadee North']
    yarr_bores = yarr_df['Site Ref'].unique()

    for bore in yarr_bores:
        df = yarr_df[yarr_df['Site Ref'] == bore]
        plt.plot(df['Collect Date'], df['Reading Value'], label = df['ID'].iloc[0])
    plt.legend(loc = 'upper left',fontsize = 'small', markerscale=0.5)

class Observations:
    def __init__(self, fname, sheetname):

        self.observations_label = "ObservationsBaseClass"
        
    def obs_bores(spatial):   
        obsbore_df = pd.read_excel('../data/data_dwer/Formation picks.xls', sheet_name = 'bore_info')
        obsbore_gdf = gpd.GeoDataFrame(obsbore_df, geometry=gpd.points_from_xy(obsbore_df.Easting, obsbore_df.Northing), crs="epsg:28350")
        obsbore_gdf = gpd.clip(obsbore_gdf, spatial.inner_boundary_poly).reset_index(drop=True)
        spatial.idobsbores = list(obsbore_gdf.ID)
        spatial.xyobsbores = list(zip(obsbore_gdf.Easting, obsbore_gdf.Northing))
        spatial.nobs = len(spatial.xyobsbores)
        spatial.obsbore_gdf = obsbore_gdf
    
    
    def process_obs(self, spatial, geomodel, mesh):

        # Get observation elevation (z) from dataframe
        depth = spatial.obsbore_gdf.zobs_mbgl.tolist()
        zobs = []
        for n in range(spatial.nobs):
            icpl = mesh.obs_cells[n]
            zobs.append(geomodel.top_geo[icpl] - depth[n])
        
        xobs, yobs = spatial.obsbore_gdf.Easting.tolist(), spatial.obsbore_gdf.Northing.tolist(), 
        obslist = list(zip(xobs, yobs, zobs))
    
        # Cretae input arrays
        obs_rec = []
        for i, cell in enumerate(mesh.obs_cells):
            x,y,z = obslist[i][0], obslist[i][1], obslist[i][2]
            point = Point(x,y,z)
            lay, icpl = geomodel.vgrid.intersect(x,y,z)
            cell_disv = icpl + lay*mesh.ncpl
            cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
            obs_rec.append([spatial.idobsbores[i], 'head', (cell_disu+1)])   
    
        self.obs_rec = obs_rec