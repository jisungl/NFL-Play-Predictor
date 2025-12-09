import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(page_title="NFL Play Predictor Pro", page_icon="🏈", layout="wide")

# load model
@st.cache_resource
def load_model():
    model = joblib.load('models/play_predictor.pkl')
    le = joblib.load('models/label_encoder.pkl')
    features = joblib.load('models/feature_names.pkl')
    importance_df = joblib.load('models/feature_importance.pkl')
    return model, le, features, importance_df

st.title("🏈 NFL Play-Calling Predictor Pro")
st.markdown("Predict **6 play types**: Short/Medium/Deep Pass + Run Left/Middle/Right")

try:
    model, label_encoder, feature_names, importance_df = load_model()
    st.success("✅ Multi-class model loaded!")
except Exception as e:
    st.error(f"⚠️ Model not found. Run `python train_model.py` first!\n{e}")
    st.stop()

# sidebar
with st.sidebar:
    st.header("📊 Model Insights")
    
    viz_option = st.radio("Select Visualization:", 
                          ["Feature Importance", "Confusion Matrix"])
    
    if viz_option == "Feature Importance":
        try:
            img = Image.open('models/feature_importance.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Train model to see feature importance")
    
    elif viz_option == "Confusion Matrix":
        try:
            img = Image.open('models/confusion_matrix.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Train model to see confusion matrix")

# prediction interface
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Game Situation")
    down = st.selectbox("Down", [1, 2, 3, 4])
    ydstogo = st.slider("Yards to Go", 1, 30, 10)
    yardline_100 = st.slider("Yards to Endzone", 1, 99, 50)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])

with col2:
    st.subheader("⏱️ Game Clock")
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    
    # current qtr time remaining
    col_min, col_sec = st.columns(2)
    with col_min:
        minutes = st.number_input("Minutes", min_value=0, max_value=15, value=7, step=1)
    with col_sec:
        seconds = st.number_input("Seconds", min_value=0, max_value=59, value=30, step=1)
    
    # current qtr seconds
    quarter_seconds = (minutes * 60) + seconds
    st.caption(f"⏱️ Time in Quarter: {minutes:02d}:{seconds:02d}")
    
    if quarter == 1:
        game_seconds = 2700 + quarter_seconds
    elif quarter == 2:
        game_seconds = 1800 + quarter_seconds
    elif quarter == 3:
        game_seconds = 900 + quarter_seconds
    else:  # Quarter 4
        game_seconds = quarter_seconds
    
    if quarter <= 2:
        half_seconds_remaining = 900 + quarter_seconds if quarter == 1 else quarter_seconds
    else:
        half_seconds_remaining = 900 + quarter_seconds if quarter == 3 else quarter_seconds
    
    total_mins = game_seconds // 60
    total_secs = game_seconds % 60
    st.caption(f"Total game time: {total_mins}:{total_secs:02d} remaining")
    
    score_diff = st.slider("Score Differential", -28, 28, 0)
    posteam_to = st.slider("Offense Timeouts", 0, 3, 3)
    defteam_to = st.slider("Defense Timeouts", 0, 3, 3)

with col3:
    st.subheader("Personnel")
    num_rbs = st.selectbox("RBs", [0, 1, 2, 3], index=1)
    num_tes = st.selectbox("TEs", [0, 1, 2, 3], index=1)
    num_wrs = st.selectbox("WRs", [0, 1, 2, 3, 4, 5], index=3)
    is_shotgun = st.checkbox("Shotgun Formation", value=False)
    is_no_huddle = st.checkbox("No Huddle", value=False)

def create_input_features(feature_names):
    """Create feature vector matching model training"""
    input_dict = {
        'down': down,
        'ydstogo': ydstogo,
        'yardline_100': yardline_100,
        'quarter': quarter,
        'half_seconds_remaining': game_seconds % 1800,
        'game_seconds_remaining': game_seconds,
        'score_differential': score_diff,
        'posteam_timeouts_remaining': posteam_to,
        'defteam_timeouts_remaining': defteam_to,
        'num_rbs': num_rbs,
        'num_tes': num_tes,
        'num_wrs': num_wrs,
        'def_dl': 4, 
        'def_lb': 3,
        'def_db': 4,
        'is_redzone': int(yardline_100 <= 20),
        'is_goalline': int(yardline_100 <= 5),
        'is_third_down': int(down == 3),
        'is_fourth_down': int(down == 4),
        'is_two_minute_drill': int(game_seconds % 1800 <= 120),
        'is_shotgun': int(is_shotgun),
        'is_no_huddle': int(is_no_huddle),
        'win_probability': 0.5,
        'expected_points': 0.0, 
    }
    
    input_df = pd.DataFrame([input_dict])
    
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_names]
    
    return input_df

# predict button
if st.button("Predict Play Type", type="primary", use_container_width=True):
    input_data = create_input_features(feature_names)
    
    probs = model.predict_proba(input_data)[0]
    predicted_class = model.predict(input_data)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    prob_df = pd.DataFrame({
        'Play Type': label_encoder.classes_,
        'Probability': probs
    }).sort_values('Probability', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if play == predicted_label else '#3498db' 
                  for play in prob_df['Play Type']]
        ax.barh(prob_df['Play Type'], prob_df['Probability'], color=colors)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title('Play Type Probabilities', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        for i, (play, prob) in enumerate(zip(prob_df['Play Type'], prob_df['Probability'])):
            ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.metric("🎯 Predicted Play", predicted_label.replace('_', ' ').title())
        st.metric("✨ Confidence", f"{probs.max():.1%}")
        
        st.markdown("**Top 3 Predictions:**")
        for i, row in prob_df.head(3).iterrows():
            st.write(f"{i+1}. {row['Play Type'].replace('_', ' ').title()}: {row['Probability']:.1%}")
    
    st.markdown("---")
    st.subheader("📈 Situational Context")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ydstogo <= 2:
            st.info("🏃 **Short yardage** - Higher run probability expected")
        elif ydstogo >= 10:
            st.info("✈️ **Long yardage** - Higher pass probability expected")
        else:
            st.info("⚖️ **Balanced situation**")
    
    with col2:
        if yardline_100 <= 5:
            st.warning("**Goal line** - Run-heavy situation")
        elif yardline_100 <= 20:
            st.warning("**Red zone** - Mixed playcalling")
    
    with col3:
        if score_diff >= 14:
            st.success("**Leading big** - Run clock expected")
        elif score_diff <= -14:
            st.error("**Trailing big** - Pass-heavy expected")

st.markdown("---")
st.caption("Built with XGBoost on 6 seasons of NFL play-by-play data (2018-2023)")