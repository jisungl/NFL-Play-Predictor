import nfl_data_py as nfl
import pandas as pd

def load_nfl_data(years=[2022, 2023]):
    df = nfl.import_pbp_data(years)
    return df

if __name__ == "__main__":
    df = load_nfl_data([2023])