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

def train_model():
    print("Loading data for 2018-2023...")
    df = load_nfl_data([2018, 2019, 2020, 2021, 2022, 2023])
    
    print("Engineering features...")
    features = engineer_features(df)
    
    le = LabelEncoder()
    features['play_label_encoded'] = le.fit_transform(features['play_label'])
    
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    X = features.drop(['play_label', 'play_label_encoded'], axis=1)
    y = features['play_label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} plays")
    print(f"Test set: {len(X_test)} plays")
    print(f"Number of features: {X.shape[1]}")
    
    # train XGBoost
    print("\nTraining XGBoost multi-class classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # eval
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n✅ Confusion matrix saved to models/confusion_matrix.png")
    
    # feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15))
    
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
    print("✅ Feature importance plot saved to models/feature_importance.png")
    
    joblib.dump(model, 'models/play_predictor.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    joblib.dump(importance_df, 'models/feature_importance.pkl')
    
    print("\n✅ Model and metadata saved!")
    
    return model, le, X.columns.tolist()

if __name__ == "__main__":
    train_model()