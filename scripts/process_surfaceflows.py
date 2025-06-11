import pandas as pd
import matplotlib.pyplot as plt

def plot_surfaceflows():
    # Load the continuous water level data from the CSV file
    continuous_df = pd.read_excel('../data/data_surfacewater/174345/WaterLevelsContinuousForSiteCrossTab.xlsx')
    discrete_df = pd.read_excel('../data/data_surfacewater/174344/WaterLevelsDiscreteForSiteCrossTab.xlsx')
    
    # Standardise the critical column names -------------------------------
    rename_map = {
        # Continuous file
        'Collected Date' : 'DateTime',
        'Collect Date'   : 'DateTime',     # if this typo shows up
        # Discrete file
        'Collected Date' : 'DateTime',
        'Collect Date'   : 'DateTime',
        # Stage columns (make sure the capitalisation is identical)
        'STAGE (m) MEAN' : 'Stage (m) MEAN',
        'STAGE (m) MIN'  : 'Stage (m) MIN',
        'STAGE (m) MAX'  : 'Stage (m) MAX',}
    
    continuous_df.rename(columns=rename_map, inplace=True, errors='ignore')
    discrete_df.rename(columns=rename_map, inplace=True, errors='ignore')

    # Convert the 'DateTime' column to datetime format
    continuous_df['DateTime'] = pd.to_datetime(continuous_df['DateTime'])
    discrete_df['DateTime'] = pd.to_datetime(discrete_df['DateTime'])

    print("Continuous df columns:", continuous_df.columns.tolist())

    for site_id in continuous_df['Site ID'].unique():
        site_data = continuous_df[continuous_df['Site ID'] == site_id]
        site_discrete = discrete_df[discrete_df['Site ID'] == site_id]

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Site ID: {site_id}', fontsize=16)

        # Plot 1: Discharge
        axs[0].plot(site_data['DateTime'], site_data['Discharge (m3/s) MEAN'], label='Mean', color='blue')
        axs[0].fill_between(site_data['DateTime'], site_data['Discharge (m3/s) MIN'], site_data['Discharge (m3/s) MAX'],
                            color='blue', alpha=0.2, label='Min-Max')
        axs[0].set_ylabel('Discharge (m³/s)')
        axs[0].set_title('Discharge (m³/s)')

        # Plot 2: Stage - CTF
        axs[1].plot(site_data['DateTime'], site_data['Stage - CTF (m) MEAN'], label='Mean', color='green')
        axs[1].fill_between(site_data['DateTime'], site_data['Stage - CTF (m) MIN'], site_data['Stage - CTF (m) MAX'],
                            color='green', alpha=0.2, label='Min-Max')
        axs[1].set_ylabel('Stage - CTF (m)')
        axs[1].set_title('Stage - CTF (m)')

        # Plot 3: Stage
        axs[2].plot(site_data['DateTime'], site_data['Stage (m) MEAN'], label='Mean', color='orange')
        axs[2].fill_between(site_data['DateTime'], site_data['Stage (m) MIN'], site_data['Stage (m) MAX'],
                            color='orange', alpha=0.2, label='Min-Max')
        if not site_discrete.empty:
            axs[2].scatter(site_discrete['DateTime'], site_discrete['Water level (SLE) (m)'],
                           color='black', label='Discrete', marker='o', zorder=5)
        axs[2].set_ylabel('Stage (m)')
        axs[2].set_title('Stage (m)')
        axs[2].set_xlabel('Collected Date')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()