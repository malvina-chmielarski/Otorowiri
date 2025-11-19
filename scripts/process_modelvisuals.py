import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math

def plot_steady_state(observed_df, modelled_df):

    observed_subset = observed_df[['Site Ref', 'Derived WL (mAHD)', 'Elevation_DEM']].copy()
    observed_subset['Site Ref'] = observed_subset['Site Ref'].astype(str)

    modelled_long = modelled_df.T.reset_index()
    modelled_long = modelled_long[1:]
    modelled_long.columns = ['Site Ref', 'Modelled WL (mAHD)']
    modelled_long['Site Ref'] = modelled_long['Site Ref'].astype(str)
    #print(modelled_long)

    merged_df = pd.merge(observed_subset, modelled_long, on='Site Ref', how='inner')

    #print(merged_df)

    #sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(
        data=merged_df,
        x='Derived WL (mAHD)',
        y='Modelled WL (mAHD)',
        hue='Site Ref',            # Color by Site Ref
        palette='tab20',           # Optional: Use a color palette with more unique hues
        s=100                      # Optional: Size of points
    )

    # Add 1:1 line for reference
    max_val = max(merged_df['Derived WL (mAHD)'].max(), merged_df['Modelled WL (mAHD)'].max())
    min_val = min(merged_df['Derived WL (mAHD)'].min(), merged_df['Modelled WL (mAHD)'].min())
    plt.plot([min_val, max_val], [min_val, max_val], ls='--', color='black', label='1:1 Line')

    # Add ground elevation line (x = y = Elevation_DEM)
    plt.scatter(
        merged_df['Derived WL (mAHD)'],
        merged_df['Elevation_DEM'],
        color='black',
        label='Ground Elevation',
        marker='o',
        s=20)

    # Labels and legend
    plt.xlabel('Observed WL (mAHD)')
    plt.ylabel('Modelled WL (mAHD)')
    plt.title('Modelled vs Observed Water Levels by Site')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Site Ref')
    plt.tight_layout()
    plt.show()

def plot_transient_state(period_df, observed_df, modelled_df):
    # Step 1 — match structure and make long-format modelled data
    modelled_df.insert(1, 'Class', period_df['Class'].values)
    modelled_long = modelled_df.melt(
        id_vars=['time', 'Class'],
        var_name='Bore ID',
        value_name='Modelled Head'
    )

    # Clean bore IDs for consistency
    modelled_long['Bore ID'] = modelled_long['Bore ID'].astype(str)
    observed_df['Site Ref'] = observed_df['Site Ref'].astype(str)

    # Step 2 — subset observed data
    observed_subset = observed_df[['Site Ref', 'Site Short Name',
                                   'Sample timeframe', 'Derived WL (mAHD)']].copy()
    observed_subset.rename(columns={
        'Site Ref': 'Bore ID',
        'Sample timeframe': 'Class',
        'Derived WL (mAHD)': 'Observed Head'
    }, inplace=True)

    # Step 3 — merge observed and modelled data on Bore ID + Class
    merged = pd.merge(
        observed_subset,
        modelled_long,
        on=['Bore ID', 'Class'],
        how='left'
    )

    print(merged.head())
    #return merged

        # Step 4 — keep only bores with at least 4 observations
    bore_counts = merged['Bore ID'].value_counts()
    valid_bores = bore_counts[bore_counts >= 4].index
    filtered = merged[merged['Bore ID'].isin(valid_bores)]

    # ---- Step 5 — Plot 4 bores per page ----
    bores = sorted(filtered['Bore ID'].unique())
    n_bores = len(bores)
    n_pages = math.ceil(n_bores / 4)

    for page in range(n_pages):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i in range(4):
            idx = page * 4 + i
            if idx >= n_bores:
                axes[i].axis('off')
                continue

            bore = bores[idx]
            bore_data = filtered[filtered['Bore ID'] == bore].sort_values('time')

            short_name = bore_data['Site Short Name'].iloc[0] if 'Site Short Name' in bore_data else bore
            axes[i].plot(bore_data['time'], bore_data['Modelled Head'], '-o', label='Modelled')
            axes[i].scatter(bore_data['time'], bore_data['Observed Head'], color='red', label='Observed')

            axes[i].set_title(f"{short_name} ({bore})")
            axes[i].set_xlabel("Model Time (days)")
            axes[i].set_ylabel("Head (mAHD)")
            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()
        plt.show()
