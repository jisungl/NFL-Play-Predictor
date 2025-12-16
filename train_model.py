import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import load_nfl_data
from utils.feature_engineering import engineer_features

def calculate_sample_weights(features):
    weights = np.ones(len(features))
    
    half_sec = features['half_seconds_remaining']
    game_sec = features['game_seconds_remaining']
    score_diff = features['score_differential']
    timeouts = features['posteam_timeouts_remaining']
    
    # 2-min off
    two_min_drill = (half_sec < 120) & (abs(score_diff) <= 8)
    weights[two_min_drill] *= 2.5
    
    # must move
    desperation = (game_sec < 120) & (score_diff < -3) & (timeouts <= 1)
    weights[desperation] *= 4.0
    
    # must-score
    must_score = (game_sec < 300) & (score_diff < 0)
    weights[must_score] *= 2.0
    
    # 4-min off
    run_clock = (game_sec < 300) & (score_diff > 10)
    weights[run_clock] *= 1.5
    
    # 4th down
    fourth_down = features['is_fourth_down'] == 1
    weights[fourth_down] *= 2.0
    
    print(f"\nSample Weight Statistics:")
    print(f"  Default weight: {(weights == 1.0).sum()} samples")
    print(f"  Two-minute drill (2.5x): {two_min_drill.sum()} samples")
    print(f"  Desperation (4.0x): {desperation.sum()} samples")
    print(f"  Must-score (2.0x): {must_score.sum()} samples")
    print(f"  Run clock (1.5x): {run_clock.sum()} samples")
    print(f"  Fourth down (2.0x): {fourth_down.sum()} samples")
    
    return weights

def train_model():
    df = load_nfl_data([2018, 2019, 2020, 2021, 2022, 2023])
    
    features = engineer_features(df)
    
    le = LabelEncoder()
    features['play_label_encoded'] = le.fit_transform(features['play_label'])
    
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    X = features.drop(['play_label', 'play_label_encoded'], axis=1)
    y = features['play_label_encoded']
    
    sample_weights = calculate_sample_weights(features)
    
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=20,
        gamma=2.0,
        subsample=0.8,
        colsample_bytree=0.6,
        colsample_bylevel=0.6,
        max_delta_step=1,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    
    model.fit(
        X_train, y_train, 
        sample_weight=weights_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[weights_test],
        verbose=50
    )
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    weighted_acc = accuracy_score(y_test, y_pred, sample_weight=weights_test)
    print(f"Test Accuracy (unweighted): {accuracy:.2%}")
    print(f"Test Accuracy (weighted): {weighted_acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
    
    joblib.dump(model, 'models/play_predictor.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    joblib.dump(importance_df, 'models/feature_importance.pkl')
    
    
    return model, le, X.columns.tolist()

if __name__ == "__main__":
    train_model()