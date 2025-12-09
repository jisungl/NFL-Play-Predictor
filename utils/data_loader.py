import nfl_data_py as nfl
import pandas as pd

def load_nfl_data(years=[2022, 2023]):
    """Load NFL play-by-play data"""
    print(f"Loading data for years: {years}")
    df = nfl.import_pbp_data(years)
    print(f"Loaded {len(df)} plays")
    return df

def quick_explore(df):
    """Quick data exploration"""
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()[:20]}...")
    print(f"\nPlay types:\n{df['play_type'].value_counts()}")
    print(f"\nNull values:\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
    
if __name__ == "__main__":
    df = load_nfl_data([2023])
    quick_explore(df)