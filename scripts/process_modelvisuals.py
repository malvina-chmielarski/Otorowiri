import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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