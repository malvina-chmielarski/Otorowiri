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

def create_seasonal_evaporation_summary(
    seasonal_csv='../data/data_precipitation/periods.csv',
    evap_xlsx='../data/data_evaporation/average_evaporation.xlsx',
    output_xlsx='../data/data_evaporation/seasonal_average_evaporation.xlsx'
):
    """
    Summarise seasonal evaporation per period (Wet/Dry) from average evaporation data.
    Computes Evaporation in m/day using period lengths.
    Fills missing periods (Evaporation_m_per_day == 0) with mean Wet/Dry values.
    """

    # --- Read evaporation data ---
    evaporation_df = pd.read_excel(evap_xlsx)
    evaporation_df['Date'] = pd.to_datetime(evaporation_df['Date']).dt.normalize()  # yyyy-mm-dd

    # --- Read seasonal periods ---
    seasonal_df = pd.read_csv(seasonal_csv)

    # Parse Start/End dates correctly (dd/mm/yyyy in CSV)
    seasonal_df['Start'] = pd.to_datetime(seasonal_df['Start'], dayfirst=True, errors='raise')
    seasonal_df['End']   = pd.to_datetime(seasonal_df['End'],   dayfirst=True, errors='raise')

    # --- Extract year and period type ---
    seasonal_df['Period_Year'] = seasonal_df['Class'].str.extract(r'(\d{4})').astype(int)
    seasonal_df['Period_Type'] = seasonal_df['Class'].str.extract(r'_(Wet|Dry)$')[0]
    seasonal_df['Period_Type'] = pd.Categorical(
        seasonal_df['Period_Type'], categories=['Wet', 'Dry'], ordered=True
    )

    # --- Sort chronologically Wet → Dry ---
    seasonal_df = seasonal_df.sort_values(['Period_Year', 'Period_Type']).reset_index(drop=True)

    # --- Compute total evaporation per period ---
    results = []
    for _, row in seasonal_df.iterrows():
        mask = (evaporation_df['Date'] >= row['Start']) & (evaporation_df['Date'] <= row['End'])
        subset = evaporation_df.loc[mask]
        total_evap_mm = subset['Average_Evap'].sum() if not subset.empty else 0

        # Number of days in the period
        period_days = row['Days']

        # Convert to mm/day and then to m/day
        evap_mm_per_day = total_evap_mm / period_days if period_days > 0 else 0
        evap_m_per_day = evap_mm_per_day / 1000

        results.append({
            'Class': row['Class'],
            'Period_Year': row['Period_Year'],
            'Period_Type': row['Period_Type'],
            'Start': row['Start'],
            'End': row['End'],
            'Days': period_days,
            'Total_Evaporation_mm': total_evap_mm,
            'Evaporation_m_per_day': evap_m_per_day
        })

    # --- Create DataFrame ---
    seasonal_evap_df = pd.DataFrame(results)

    # --- Fill missing Wet/Dry periods with mean of existing periods ---
    if not seasonal_evap_df.empty:
        for period_type in ['Wet', 'Dry']:
            # Compute mean of existing non-zero periods
            mean_val = seasonal_evap_df.loc[
                (seasonal_evap_df['Period_Type'] == period_type) &
                (seasonal_evap_df['Evaporation_m_per_day'] > 0),
                'Evaporation_m_per_day'
            ].mean()

            # Fill missing (0) values
            seasonal_evap_df.loc[
                (seasonal_evap_df['Period_Type'] == period_type) &
                (seasonal_evap_df['Evaporation_m_per_day'] == 0),
                'Evaporation_m_per_day'
            ] = mean_val

    # --- Export to Excel ---
    seasonal_evap_df.to_excel(output_xlsx, index=False)
    print(f"Seasonal evaporation summary saved to {output_xlsx}")

    return seasonal_evap_df

'''def create_seasonal_evaporation_summary(
    seasonal_csv='../data/data_precipitation/periods.csv',
    evap_xlsx='../data/data_evaporation/average_evaporation.xlsx',
    output_xlsx='../data/data_evaporation/seasonal_average_evaporation.xlsx'
):
    """
    Summarise seasonal evaporation per period (Wet/Dry) from average evaporation data.
    Computes Evaporation in m/day using period lengths.
    """

    # --- Read evaporation data ---
    evaporation_df = pd.read_excel(evap_xlsx)
    evaporation_df['Date'] = pd.to_datetime(evaporation_df['Date']).dt.normalize()  # yyyy-mm-dd

    # --- Read seasonal periods ---
    seasonal_df = pd.read_csv(seasonal_csv)

    # Parse Start/End dates correctly (dd/mm/yyyy in CSV)
    seasonal_df['Start'] = pd.to_datetime(seasonal_df['Start'], dayfirst=True, errors='raise')
    seasonal_df['End']   = pd.to_datetime(seasonal_df['End'],   dayfirst=True, errors='raise')

    # --- Extract year and period type ---
    seasonal_df['Period_Year'] = seasonal_df['Class'].str.extract(r'(\d{4})').astype(int)
    seasonal_df['Period_Type'] = seasonal_df['Class'].str.extract(r'_(Wet|Dry)$')[0]
    seasonal_df['Period_Type'] = pd.Categorical(
        seasonal_df['Period_Type'], categories=['Wet', 'Dry'], ordered=True
    )

    # --- Sort chronologically Wet → Dry ---
    seasonal_df = seasonal_df.sort_values(['Period_Year', 'Period_Type']).reset_index(drop=True)

    # --- Compute total evaporation per period ---
    results = []
    for _, row in seasonal_df.iterrows():
        mask = (evaporation_df['Date'] >= row['Start']) & (evaporation_df['Date'] <= row['End'])
        subset = evaporation_df.loc[mask]
        total_evap_mm = subset['Average_Evap'].sum()

        # Get number of days in the period from periods.csv
        period_days = row['Days']

        # Convert to mm/day
        evap_mm_per_day = total_evap_mm / period_days if period_days > 0 else None

        # Convert to m/day for the model
        evap_m_per_day = evap_mm_per_day / 1000 if evap_mm_per_day is not None else None

        results.append({
            'Class': row['Class'],
            'Period_Year': row['Period_Year'],
            'Period_Type': row['Period_Type'],
            'Start': row['Start'],
            'End': row['End'],
            'Days': period_days,
            'Total_Evaporation_mm': total_evap_mm,
            'Evaporation_m_per_day': evap_m_per_day
        })
    # --- Fill missing Wet/Dry periods with mean ---
    for period_type in ['Wet', 'Dry']:
        mean_val = seasonal_evap_df.loc[
            (seasonal_evap_df['Period_Type'] == period_type) & 
            (seasonal_evap_df['Evaporation_m_per_day'] > 0),
            'Evaporation_m_per_day'
        ].mean()

        seasonal_evap_df.loc[
            (seasonal_evap_df['Period_Type'] == period_type) & 
            (seasonal_evap_df['Evaporation_m_per_day'] == 0),
            'Evaporation_m_per_day'
        ] = mean_val

    # --- Create DataFrame and export ---
    seasonal_evap_df = pd.DataFrame(results)
    seasonal_evap_df.to_excel(output_xlsx, index=False)
    print(f"Seasonal evaporation summary saved to {output_xlsx}")

    return seasonal_evap_df'''
'''
def create_seasonal_evaporation_summary(seasonal_csv='../data/data_precipitation/start_end_seasons.csv',
                                        evap_xlsx='../data/data_evaporation/average_evaporation.xlsx',
                                        output_xlsx='../data/data_evaporation/seasonal_average_evaporation.xlsx'):
    # --- Read evaporation data ---
    evaporation_df = pd.read_excel(evap_xlsx)
    evaporation_df['Date'] = pd.to_datetime(evaporation_df['Date'])

    # --- Read seasonal periods ---
    seasonal_df = pd.read_csv(seasonal_csv)
    seasonal_df['Start'] = pd.to_datetime(seasonal_df['Start'], format='%d/%m/%Y', errors='raise')
    seasonal_df['End'] = pd.to_datetime(seasonal_df['End'], format='%d/%m/%Y' errors='raise')

    seasonal_df['Start'] = seasonal_df['Start'].dt.normalize()
    seasonal_df['End']   = seasonal_df['End'].dt.normalize()

    # --- Extract year and period type from Class ---
    seasonal_df['Period_Year'] = seasonal_df['Class'].str.extract(r'(\d{4})').astype(int)
    seasonal_df['Period_Type'] = seasonal_df['Class'].str.extract(r'_(Wet|Dry)$')[0]
    seasonal_df['Period_Type'] = pd.Categorical(seasonal_df['Period_Type'], categories=['Wet', 'Dry'], ordered=True)

    # --- Sort correctly ---
    seasonal_df = seasonal_df.sort_values(['Period_Year', 'Period_Type']).reset_index(drop=True)

    # --- Compute total evaporation per period ---
    results = []
    for _, row in seasonal_df.iterrows():
        mask = (evaporation_df['Date'] >= row['Start']) & (evaporation_df['Date'] <= row['End'])
        subset = evaporation_df.loc[mask]
        seasonal_evap = subset['Average_Evap'].sum()

        results.append({
            'Class': row['Class'],
            'Period_Year': row['Period_Year'],
            'Period_Type': row['Period_Type'],
            'Start': row['Start'],
            'End': row['End'],
            'Total_Evaporation_mm': seasonal_evap
        })

    seasonal_evap_df = pd.DataFrame(results)

    # --- Export to Excel ---
    seasonal_evap_df.to_excel(output_xlsx, index=False)
    print(f"Seasonal evaporation summary saved to {output_xlsx}")

    return seasonal_evap_df'''