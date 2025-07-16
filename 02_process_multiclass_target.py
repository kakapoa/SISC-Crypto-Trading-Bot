# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ta as ta_lib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("Starting Pipeline V4: Creating Multiclass Target...")

# --- Load and Process Data ---
print("\n--- STEP 1: Loading and processing base data ---")
try:
    df = pd.read_csv('crypto_market_data_binance_10y.csv', parse_dates=['timestamp'])
    df_filtered = df.groupby('coin_id').filter(lambda x: len(x) >= 35)
    print(f"Processing {df_filtered['coin_id'].nunique()} coins with sufficient data.")
    
    def calculate_features(group):
        group = group.set_index('timestamp')
        group = ta_lib.add_all_ta_features(group, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        return group

    features_df = df_filtered.groupby('coin_id', group_keys=False).apply(calculate_features)
    features_df.reset_index(inplace=True)
    features_df.set_index(['coin_id', 'timestamp'], inplace=True)
    print("Feature engineering complete.")
except FileNotFoundError:
    print("Error: 'crypto_market_data_binance_10y.csv' not found.")
    exit()

# --- STEP 2: Create Multiclass Target ---
print("\n--- STEP 2: Creating multiclass target variable ---")
PREDICTION_WINDOW = 7
features_df['future_return'] = features_df.groupby('coin_id')['close'].transform(lambda x: x.shift(-PREDICTION_WINDOW) / x - 1)
features_df.dropna(subset=['future_return'], inplace=True)

bins = [-np.inf, -0.10, 0.0, 0.15, np.inf]
labels = [0, 1, 2, 3]
features_df['target'] = pd.cut(features_df['future_return'], bins=bins, labels=labels, right=False)

features_df.dropna(subset=['target'], inplace=True)
features_df['target'] = features_df['target'].astype(int)
features_df.drop(columns=['future_return'], inplace=True)
print("Multiclass target variable created.")

# --- STEP 3: Final Cleanup and Save ---
print("\n--- STEP 3: Cleaning and saving final dataset ---")
numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
if 'target' not in numeric_cols:
    numeric_cols.append('target')
final_df = features_df[numeric_cols]
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(inplace=True)

output_filename = 'dataset_para_entrenamiento_multiclass.csv'
final_df.to_csv(output_filename)

print("\nMULTICLASS PIPELINE COMPLETE!")
print(f"Final dataset for training saved as '{output_filename}'.")
print(f"Final dataset shape: {final_df.shape}")
print("\nClass balance of the new Target:")
print(final_df['target'].value_counts(normalize=True).sort_index())

