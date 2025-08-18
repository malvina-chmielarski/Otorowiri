import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from pandas.tseries.offsets import MonthEnd
import os

def clean_precipitation():
    three_springs_df = pd.read_csv('../data/data_precipitation/IDCJAC0009_008121_1800/IDCJAC0009_008121_1800_Data.csv')
    arena_df = pd.read_csv('../data/data_precipitation/IDCJAC0009_008273_1800/IDCJAC0009_008273_1800_Data.csv')
    mingenew_df = pd.read_csv('../data/data_precipitation/IDCJAC0009_008088_1800/IDCJAC0009_008088_1800_Data.csv')

    df_list = [
        ('three_springs_008121.csv', three_springs_df),
        ('arena_008273.csv', arena_df),
        ('mingenew_008088.csv', mingenew_df)
        ]

    # Column names in all precipitation dataframes = ['Product code', 'Bureau of Meteorology station number', 'Year', 'Month', 'Day', 'Rainfall amount (millimetres)', 'Period over which rainfall was measured (days)', 'Quality']
    rename_map = {
        'Product code': 'Product Code',
        'Bureau of Meteorology station number': 'Station Number',
        'Year': 'Year',
        'Month': 'Month',
        'Day': 'Day',
        'Rainfall amount (millimetres)': 'Rainfall (mm)',
        'Period over which rainfall was measured (days)': 'Measurement Period (days)',
        'Quality': 'Quality'
    }

    output_folder = '../data/data_precipitation/'
    os.makedirs(output_folder, exist_ok=True)

    for filename, df in df_list:
        df.rename(columns=rename_map, inplace=True, errors='ignore') # standardise the column names
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']]) # Create datetime column
        df.drop(columns=['Year', 'Month', 'Day'], inplace=True, errors='ignore') # drop the original Year, Month, Day columns
        df = df.dropna(subset=['Rainfall (mm)']) # Drop rows with NaN in 'Rainfall (mm)' column
        #check df measurement periods and fix anything where the measurement period is not 1 day
        print("original  measurement periods:", df['Measurement Period (days)'].unique())
        mask = df['Rainfall (mm)'].notna() & df['Measurement Period (days)'].isna()
        df.loc[mask, 'Measurement Period (days)'] = 1

        #normalise the measurement periods
        new_rows = []
        for idx, row in df[df['Measurement Period (days)'] > 1].iterrows():
            period = int(row['Measurement Period (days)'])
            total_rainfall = row['Rainfall (mm)']
            date = row['Date']
            split_rainfall = total_rainfall / period if pd.notnull(total_rainfall) else 0
            
            for i in range(period):
                new_date = date - timedelta(period -1 - i)
                new_row = row.copy()
                new_row['Date'] = new_date
                new_row['Rainfall (mm)'] = split_rainfall
                new_row['Measurement Period (days)'] = 1
                new_rows.append(new_row)
        
        df = df[df['Measurement Period (days)'] == 1].copy() # Remove original rows with period > 1
        
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True) # Append the normalized rows
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("cleaned  measurement periods:", df['Measurement Period (days)'].unique())

        # check if there are any duplicate dates
        if df.duplicated(subset=['Date']).any():
            print(f"Warning: Duplicate dates are present in {filename}")
            #find the duplicates
            duplicates = df[df.duplicated(subset=['Date'], keep=False)]
            print("Duplicate dates found: " + str(duplicates['Date'].unique().tolist()))
            # Remove duplicates, keep where 'Rainfall (mm)' is not 0
            df = df[~df.duplicated(subset=['Date'], keep='first') | (df['Rainfall (mm)'] != 0)]
            print(f"Removed duplicates from {filename}, keeping first occurrence where 'Rainfall (mm)' is not 0")
        else:
            print(f"No duplicate dates found in {filename}")

        # Save to CSV
        df.to_csv(os.path.join(output_folder, filename), index=False)
        print(f"Saved cleaned data to {filename}")

def average_precipitation():
    # Load the cleaned CSV files
    base_path = '../data/data_precipitation/'
    files = {
        'Three Springs': 'three_springs_008121.csv',
        'Arena': 'arena_008273.csv',
        'Mingenew': 'mingenew_008088.csv'
    }
    dfs = {}
    for name, file in files.items():
        df = pd.read_csv(os.path.join(base_path, file), parse_dates=['Date'])
        dfs[name] = df[['Date', 'Rainfall (mm)']].copy()
        dfs[name].rename(columns={'Rainfall (mm)': name}, inplace=True)

    merged_df = dfs['Three Springs']
    for name in ['Arena', 'Mingenew']:
        merged_df = pd.merge(merged_df, dfs[name], on='Date', how='outer')
    merged_df.sort_values('Date', inplace=True)
    merged_df['Mean Rainfall'] = merged_df[['Three Springs', 'Arena', 'Mingenew']].mean(axis=1, skipna=True) #takes the average of one, two, or three stations where available

    output_df = merged_df[['Date', 'Mean Rainfall']].copy()
    output_file = os.path.join(base_path, 'average_rainfall.csv')
    output_df.to_csv(output_file, index=False)

    print(f"Saved average rainfall data to: {output_file}")

def plot_full_precipitation():

    # Load the cleaned CSV files
    base_path = '../data/data_precipitation/'
    files = {
        'Three Springs': 'three_springs_008121.csv',
        'Arena': 'arena_008273.csv',
        'Mingenew': 'mingenew_008088.csv',
        'Average': 'average_rainfall.csv'
    }
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.tight_layout(pad=3)

    for i, (name, filename) in enumerate(files.items()):
        df = pd.read_csv(os.path.join(base_path, filename), parse_dates=['Date'])
        
        # Column name may differ for average file
        y_col = 'Rainfall (mm)' if name != 'Average' else 'Mean Rainfall'
        if y_col not in df.columns:
            # fallback for renamed average column
            y_col = df.columns[1]

        axes[i].plot(df['Date'], df[y_col], label=name, color='tab:blue' if name != 'Average' else 'tab:green')
        axes[i].set_ylabel('Rainfall (mm)')
        axes[i].set_title(f'{name} Rainfall')
        axes[i].grid(True)

    axes[-1].set_xlabel('Date')
    plt.show()

def create_precip_summary():
    # Load the average rainfall data
    base_path = '../data/data_precipitation/'
    avg_df = pd.read_csv(os.path.join(base_path, 'average_rainfall.csv'), parse_dates=['Date'])

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Date': avg_df['Date'],
        'Year': avg_df['Date'].dt.year,
        'Month': avg_df['Date'].dt.month,
        'Total Rainfall (mm)': avg_df['Mean Rainfall']
    })

    # Group by Year and Month to get total rainfall for each month
    summary_df = summary_df.groupby(['Year', 'Month'])[['Total Rainfall (mm)']].sum().reset_index()
    #summary_df = summary_df.groupby(['Year', 'Month']).sum().reset_index()
    # Recreate full date column for ordering
    summary_df['Date'] = pd.to_datetime(summary_df[['Year', 'Month']].assign(DAY=1))

    # Classify wet and dry months
    summary_df['Month Type'] = summary_df['Total Rainfall (mm)'].apply(lambda x: 'Wet' if x > 40 else 'Dry')
    summary_df['Year Type'] = summary_df.groupby('Year')['Total Rainfall (mm)'].transform(
        lambda x: 'Wet' if x.sum() > 600 else 'Dry')
    
    #Create class code
    class_labels = []
    prev_class = None

    months = summary_df.sort_values('Date').reset_index(drop=True)
    dry_buffer = []

    for i, row in months.iterrows():
        year, month, month_type = row['Year'], row['Month'], row['Month Type']
        date = row['Date']

        is_wet_season_month = 5 <= month <= 10  # May to September
        
        if is_wet_season_month and month_type == 'Wet':
            # Assign as wet season in current year
            current_class = f"{year}_Wet"
            dry_buffer = []  # reset dry buffer
        else:
            # Dry season handling
            dry_buffer.append((i, year, month))
            # Determine if we are at the end of a dry run
            if len(dry_buffer) > 4:
                # If >4 dry months in a row, confirm dry period
                for idx, y, m in dry_buffer:
                    assign_year = y if m > 4 else y - 1
                    months.loc[idx, 'Class'] = f"{assign_year}_Dry"
                dry_buffer = []
            current_class = None

        if current_class:
            months.loc[i, 'Class'] = current_class

    # For any leftover dry months in buffer
    for idx, y, m in dry_buffer:
        assign_year = y if m > 4 else y - 1
        months.loc[idx, 'Class'] = f"{assign_year}_Dry"

    summary_df['Class'] = months['Class']

    # Save the summary DataFrame to CSV
    summary_file = os.path.join(base_path, 'precipitation_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved precipitation summary to: {summary_file}")

def start_end_seasons():
    # Load the summary DataFrame
    base_path = '../data/data_precipitation/'
    summary_df = pd.read_csv(os.path.join(base_path, 'precipitation_summary_classifications.csv'))
    summary_df['Date'] = pd.to_datetime(summary_df['Date'], dayfirst =True)

    # Sort by date for correct plotting
    summary_df = summary_df.sort_values('Date').reset_index(drop=True)

    # Identify start and end of each unique contiguous 'Class' group
    summary_df['Class_Change'] = (summary_df['Class'] != summary_df['Class'].shift()).cumsum()

    period_ranges = summary_df.groupby(['Class_Change', 'Class']).agg(
        Start=('Date', 'min'),
        End=('Date', 'max')
    ).reset_index()

    period_ranges['End'] = period_ranges['End'] + MonthEnd(0)
    print(period_ranges[['Class', 'Start', 'End']].head())

    # Check if there are any overlapping periods
    overlaps = period_ranges[period_ranges.duplicated(subset=['Start', 'End'], keep=False)]
    if not overlaps.empty:
        print("Warning: Overlapping periods found in the classification data:")
        print(overlaps)

    # Check if there are any gaps in the periods
    period_ranges['Next_Start'] = period_ranges['Start'].shift(-1)
    period_ranges['Gap'] = (period_ranges['Next_Start'] - period_ranges['End']).dt.days
    gaps = period_ranges[period_ranges['Gap'] > 1]
    if not gaps.empty:
        print("Warning: Gaps found in the classification periods:")
        print(gaps[['Class', 'End', 'Next_Start', 'Gap']])

    #save the period ranges to a CSV file
    period_ranges_file = os.path.join(base_path, 'start_end_seasons.csv')
    period_ranges.to_csv(period_ranges_file, index=False)
    print(f"Saved precipitation period ranges to: {period_ranges_file}")

'''def plot_precip_summary():
    # Load the summary DataFrame
    base_path = '../data/data_precipitation/'
    summary_df = pd.read_csv(os.path.join(base_path, 'precipitation_summary_classifications.csv'), parse_dates=['Date'])

    # Sort by date for correct plotting
    summary_df = summary_df.sort_values('Date').reset_index(drop=True)

    # Identify start and end of each unique contiguous 'Class' group
    summary_df['Class_Change'] = (summary_df['Class'] != summary_df['Class'].shift()).cumsum()

    period_ranges = summary_df.groupby(['Class_Change', 'Class']).agg(
        Start=('Date', 'min'),
        End=('Date', 'max')
    ).reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))

    # Background colored rectangles
    for _, row in period_ranges.iterrows():
        color = 'lightblue' if 'Wet' in row['Class'] else 'khaki'
        ax.axvspan(row['Start'], row['End'] + pd.offsets.MonthEnd(1), color=color, alpha=0.3)

    # Plot precipitation as dots
    ax.plot(summary_df['Date'], summary_df['Total Rainfall (mm)'], 'o', color='black', markersize=4)

    # Format plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Rainfall (mm)')
    ax.set_title('Monthly Precipitation with Wet/Dry Periods Highlighted')
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()'''