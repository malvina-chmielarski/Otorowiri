import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from pandas.tseries.offsets import MonthEnd
import os

def plot_evaporation_data():
    carmamah_df = pd.read_excel('../data/data_evaporation/8025 - Carnamah.xlsx')
    three_springs_df = pd.read_excel('../data/data_evaporation/8121 - Three Springs.xlsx')
    eneabba_df = pd.read_excel('../data/data_evaporation/8225 - Eneabba.xlsx')
    badgingarra_df = pd.read_excel('../data/data_evaporation/9037 - Badgingarra.xlsx')

    df_list = [
        ("Carnamah", carmamah_df),
        ("Three Springs", three_springs_df),
        ("Eneabba", eneabba_df),
        ("Badgingarra", badgingarra_df)
    ]

    for name, df in df_list:
        # Ensure 'date' is datetime
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['Evap'], label='Evaporation')
        plt.title(f'Evaporation Over Time - {name}')
        plt.xlabel('Date')
        plt.ylabel('Evaporation (mm/day)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def create_average_evaporation_df():
    # Create a DataFrame with average evaporation data
    carmamah_df = pd.read_excel('../data/data_evaporation/8025 - Carnamah.xlsx')
    three_springs_df = pd.read_excel('../data/data_evaporation/8121 - Three Springs.xlsx')
    eneabba_df = pd.read_excel('../data/data_evaporation/8225 - Eneabba.xlsx')
    badgingarra_df = pd.read_excel('../data/data_evaporation/9037 - Badgingarra.xlsx')

    df_list = [
        ("Carnamah", carmamah_df),
        ("Three Springs", three_springs_df),
        ("Eneabba", eneabba_df),
        ("Badgingarra", badgingarra_df)
    ]

    # Convert dates (assuming column is 'date' or similar)
    for name, df in df_list:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.rename(columns={'Evap': f'Evap_{name}'}, inplace=True)
    # Merge on date
    df_merged = carmamah_df[['Date', 'Evap_Carnamah']] \
        .merge(three_springs_df[['Date', 'Evap_Three Springs']], on='Date', how='outer') \
        .merge(eneabba_df[['Date', 'Evap_Eneabba']], on='Date', how='outer') \
        .merge(badgingarra_df[['Date', 'Evap_Badgingarra']], on='Date', how='outer')
    df_merged.sort_values('Date', inplace=True)

    # Compute row-wise average evaporation, skipping NaNs
    df_merged['Average_Evap'] = df_merged[
        ['Evap_Carnamah', 'Evap_Three Springs', 'Evap_Eneabba', 'Evap_Badgingarra']
    ].mean(axis=1, skipna=True)

    df_merged.to_excel('../data/data_evaporation/average_evaporation.xlsx', index=False)

def plot_average_evaporation():
    # Load the average evaporation data
    df = pd.read_excel('../data/data_evaporation/average_evaporation.xlsx')

    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Average_Evap'], label='Average Evaporation', color='orange')
    plt.title('Average Evaporation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Evaporation (mm/day)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_seasonal_evaporation_summary(seasonal_df):
    evaporation_df = pd.read_excel('../data/data_evaporation/average_evaporation.xlsx')
    