import pandas as pd
import numpy as np

def create_play_labels(df):
    """Create more granular play type labels"""
    
    df['play_label'] = 'other'
    
    run_mask = df['play_type'] == 'run'
    df.loc[run_mask & (df['run_location'] == 'left'), 'play_label'] = 'run_left'
    df.loc[run_mask & (df['run_location'] == 'middle'), 'play_label'] = 'run_middle'
    df.loc[run_mask & (df['run_location'] == 'right'), 'play_label'] = 'run_right'
    
    pass_mask = df['play_type'] == 'pass'
    df.loc[pass_mask & (df['air_yards'] < 10), 'play_label'] = 'pass_short'
    df.loc[pass_mask & (df['air_yards'].between(10, 20)), 'play_label'] = 'pass_medium'
    df.loc[pass_mask & (df['air_yards'] > 20), 'play_label'] = 'pass_deep'
    
    return df

def engineer_features(df):
    """Create comprehensive features for model training"""
    
    print("Creating play labels...")
    df = create_play_labels(df)
    
    valid_labels = ['run_left', 'run_middle', 'run_right', 
                    'pass_short', 'pass_medium', 'pass_deep']
    df = df[df['play_label'].isin(valid_labels)].copy()
    
    print(f"Play distribution:\n{df['play_label'].value_counts()}\n")
    
    features = pd.DataFrame()
    
    # basic situation features
    features['down'] = df['down']
    features['ydstogo'] = df['ydstogo']
    features['yardline_100'] = df['yardline_100']
    features['quarter'] = df['qtr']
    features['half_seconds_remaining'] = df['half_seconds_remaining']
    features['game_seconds_remaining'] = df['game_seconds_remaining']
    features['score_differential'] = df['score_differential']
    features['posteam_timeouts_remaining'] = df['posteam_timeouts_remaining'].fillna(3)
    features['defteam_timeouts_remaining'] = df['defteam_timeouts_remaining'].fillna(3)
    
    # personnel
    if 'offense_personnel' in df.columns:
        features['num_rbs'] = df['offense_personnel'].str.extract('(\d+) RB')[0].fillna(1).astype(float)
        features['num_tes'] = df['offense_personnel'].str.extract('(\d+) TE')[0].fillna(1).astype(float)
        features['num_wrs'] = df['offense_personnel'].str.extract('(\d+) WR')[0].fillna(3).astype(float)
    
    if 'defense_personnel' in df.columns:
        features['def_dl'] = df['defense_personnel'].str.extract('(\d+) DL')[0].fillna(4).astype(float)
        features['def_lb'] = df['defense_personnel'].str.extract('(\d+) LB')[0].fillna(4).astype(float)
        features['def_db'] = df['defense_personnel'].str.extract('(\d+) DB')[0].fillna(3).astype(float)
    
    # formation
    if 'offense_formation' in df.columns:
        formation_dummies = pd.get_dummies(df['offense_formation'], prefix='form')
        top_formations = formation_dummies.sum().nlargest(5).index
        features = pd.concat([features, formation_dummies[top_formations]], axis=1)
    
    # situational flags
    features['is_redzone'] = (df['yardline_100'] <= 20).astype(int)
    features['is_goalline'] = (df['yardline_100'] <= 5).astype(int)
    features['is_third_down'] = (df['down'] == 3).astype(int)
    features['is_fourth_down'] = (df['down'] == 4).astype(int)
    features['is_two_minute_drill'] = (df['half_seconds_remaining'] <= 120).astype(int)
    features['is_shotgun'] = (df['shotgun'] == 1).astype(int) if 'shotgun' in df.columns else 0
    features['is_no_huddle'] = (df['no_huddle'] == 1).astype(int) if 'no_huddle' in df.columns else 0
    
    if 'wp' in df.columns:
        features['win_probability'] = df['wp'].fillna(0.5)
    
    if 'ep' in df.columns:
        features['expected_points'] = df['ep'].fillna(0)
    
    # team features
    if 'posteam' in df.columns:
        team_dummies = pd.get_dummies(df['posteam'], prefix='team')
        features = pd.concat([features, team_dummies], axis=1)
    
    features['play_label'] = df['play_label']
    
    features = features.dropna()
    
    print(f"Engineered {len(features)} plays with {len(features.columns)-1} features")
    print(f"Final play distribution:\n{features['play_label'].value_counts()}\n")
    
    return features

if __name__ == "__main__":
    from data_loader import load_nfl_data
    df = load_nfl_data([2022, 2023])
    features = engineer_features(df)
    print(features.head())
    print(f"\nFeature columns: {features.columns.tolist()}")