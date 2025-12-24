import pandas as pd
import os

# Define file paths
FINAL_FEATURES_PATH = 'final_features.csv'
SPLIT_PATH = '../TwiBot-22/split.csv'
LABEL_PATH = '../TwiBot-22/label.csv'

OUTPUT_FULL = 'final_features_with_split_and_label.csv'
OUTPUT_TRAIN = 'train_features.csv'
OUTPUT_VAL = 'val_features.csv'
OUTPUT_TEST = 'test_features.csv'

def normalize_id(df, col_name='id'):
    """
    Normalizes user IDs by stripping 'u' prefix and converting to integer.
    """
    # Ensure column is string first to handle potential mixed types safely, though they should be strings
    df[col_name] = df[col_name].astype(str).str.replace('u', '', regex=False)
    # Convert to numeric, coercing errors to NaN (though we expect valid ints)
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int64')
    return df

def main():
    print("Loading files...")
    if not os.path.exists(FINAL_FEATURES_PATH):
        print(f"Error: {FINAL_FEATURES_PATH} not found.")
        return

    df_features = pd.read_csv(FINAL_FEATURES_PATH)
    print(f"Loaded features: {len(df_features)} rows")

    if os.path.exists(SPLIT_PATH):
        df_split = pd.read_csv(SPLIT_PATH)
        print(f"Loaded split: {len(df_split)} rows")
        df_split = normalize_id(df_split, 'id')
        df_split.rename(columns={'id': 'user_id'}, inplace=True)
    else:
        print(f"Warning: {SPLIT_PATH} not found. Splits will be unknown.")
        df_split = pd.DataFrame(columns=['user_id', 'split'])

    if os.path.exists(LABEL_PATH):
        df_label = pd.read_csv(LABEL_PATH)
        print(f"Loaded labels: {len(df_label)} rows")
        df_label = normalize_id(df_label, 'id')
        df_label.rename(columns={'id': 'user_id'}, inplace=True)
    else:
        print(f"Warning: {LABEL_PATH} not found. Labels will be unknown.")
        df_label = pd.DataFrame(columns=['user_id', 'label'])

    # Merge
    print("Merging data...")
    # Left join features with split
    df_merged = df_features.merge(df_split, on='user_id', how='left')
    
    # Left join result with labels
    df_merged = df_merged.merge(df_label, on='user_id', how='left')
    
    # Fill missing values
    df_merged['split'] = df_merged['split'].fillna('unknown')
    df_merged['label'] = df_merged['label'].fillna('unknown')
    
    print("Sample of merged data:")
    print(df_merged.head())
    
    # Save full file
    print(f"Saving merged data to {OUTPUT_FULL}...")
    df_merged.to_csv(OUTPUT_FULL, index=False)
    
    # Create and save subsets
    splits = ['train', 'val', 'test']
    files = [OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST]
    
    for split_name, file_name in zip(splits, files):
        subset = df_merged[df_merged['split'] == split_name]
        print(f"Saving {split_name} subset ({len(subset)} rows) to {file_name}...")
        subset.to_csv(file_name, index=False)
        
    print("Done.")

if __name__ == "__main__":
    main()
