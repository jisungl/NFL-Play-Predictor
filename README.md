# 🏈 NFL Play-Calling Predictor

An AI-powered system that predicts offensive play types (run direction and pass depth) based on game situation using machine learning.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-orange?style=for-the-badge)

## Overview

This project uses machine learning to predict NFL play-calling decisions in real-time. The model analyzes game situations (down, distance, field position, personnel, etc.) and predicts one of 6 play types:
- **Run Left/Middle/Right**
- **Pass Short/Medium/Deep**

## Features

- **Multi-class XGBoost Classifier** trained on 200K+ plays from 2018-2023 NFL seasons
- **50+ engineered features** including down/distance, personnel groupings, formations, and advanced metrics
- **Interactive Streamlit Dashboard** for real-time predictions
- **Model Explainability** with feature importance rankings and confusion matrices
- **Achieves ~52% accuracy** on multi-class prediction task

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/nfl-play-predictor.git
cd nfl-play-predictor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train_model.py
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## Details

### Model
- **Algorithm**: XGBoost Multi-class Classifier
- **Features**: 50+ (down/distance, personnel, formations, win probability, etc.)
- **Classes**: 6 (run_left, run_middle, run_right, pass_short, pass_medium, pass_deep)
- **Training Data**: 200K+ plays from 2018-2023 NFL seasons

### Key Features
- Game situation (down, distance, field position)
- Personnel groupings (RBs, TEs, WRs, DL, LB, DBs)
- Formation types (shotgun, no-huddle, etc.)
- Contextual flags (redzone, two-minute drill, etc.)
- Advanced metrics (win probability, expected points)

### Performance
- **Overall Accuracy**: ~52%
- **Training Time**: ~5 minutes on standard laptop
- **Inference**: <100ms per prediction

## Data Source

This project uses the [nflfastR](https://www.nflfastr.com/) play-by-play dataset, which provides comprehensive NFL data from 1999-present.


## Acknowledgments

- NFL data provided by [nflfastR](https://www.nflfastr.com/)
- Built with [Streamlit](https://streamlit.io/), [XGBoost](https://xgboost.readthedocs.io/), and [scikit-learn](https://scikit-learn.org/)
