# NFL Play-Calling Predictor

A machine learning model that predicts offensive play calls in the NFL, classifying each play into one of six types based on game situation. Built with XGBoost and deployed as an interactive Streamlit dashboard.

## Play Types

The model classifies plays into six categories:

- **Run Left / Run Middle / Run Right** — based on `run_location` from play-by-play data
- **Pass Short / Pass Medium / Pass Deep** — based on `air_yards` thresholds (short: <10, medium: 10-20, deep: >20)

## Data

Training data comes from [nflfastR](https://www.nflfastr.com/) play-by-play datasets spanning the 2018-2023 NFL seasons, totaling over 200,000 plays after filtering for valid run/pass plays with complete location and air yard data.

## Feature Engineering

Over 60 features are engineered from raw play-by-play data, including:

- **Game state** — down, yards to go, field position, quarter, game clock, half clock, score differential
- **Personnel** — offensive skill position counts (RB, TE, WR) and defensive front counts (DL, LB, DB), extracted from personnel strings via regex
- **Formation** — shotgun, no-huddle, and top-5 offensive formation types (one-hot encoded)
- **Situational flags** — red zone, goal line, third down, fourth down, two-minute drill
- **Advanced metrics** — win probability and expected points from nflfastR
- **Team tendencies** — one-hot encoded team identifiers for the top 10 most frequent offenses

## Sample Weighting

The model uses custom sample weights to emphasize high-leverage situations where play-calling patterns diverge most from baseline tendencies:

- **Two-minute drill** (2.5x) — under 2 minutes in the half with a one-score game
- **Desperation** (4.0x) — under 2 minutes in the game, trailing, with limited timeouts
- **Must-score** (2.0x) — under 5 minutes remaining, trailing
- **Clock management** (1.5x) — under 5 minutes remaining, leading by multiple scores
- **Fourth down** (2.0x) — all fourth-down plays

This weighting addresses a problem found during development: early model versions leaned heavily on formation (particularly shotgun) as a predictor, which was statistically reasonable but produced unrealistic predictions in time-constrained scenarios. Upweighting these situations taught the model that clock and score context should override formation tendencies when the game state narrows the play-caller's options.

## Model

- **Algorithm**: XGBoost multi-class classifier (`multi:softprob`)
- **Estimators**: 300
- **Max depth**: 8
- **Learning rate**: 0.05
- **Regularization**: L1 (`reg_alpha=1.0`), L2 (`reg_lambda=5.0`), `gamma=2.0`, `min_child_weight=20`
- **Subsampling**: 80% row, 60% column per tree, 60% column per level
- **Overall accuracy**: ~52% on a 6-class prediction task

## Streamlit Dashboard

The app provides an interactive prediction interface with three input panels:

- **Game Situation** — down, yards to go, field position, quarter
- **Game Clock** — minutes/seconds remaining, score differential, timeouts for each team
- **Personnel** — RB/TE/WR counts, shotgun and no-huddle toggles

Clicking "Predict" returns a probability distribution across all six play types, displayed as a horizontal bar chart alongside the top prediction and its confidence. The sidebar provides model diagnostics including feature importance rankings and a confusion matrix.

## Usage
```bash
# Clone and set up environment
git clone https://github.com/jisungl/NFL-Play-Predictor.git
cd NFL-Play-Predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the model (~2GB download, 10-15 min)
python train_model.py

# Launch the dashboard
streamlit run app.py
```

## Tech Stack

- Python
- XGBoost
- scikit-learn
- Streamlit
- pandas, NumPy
- matplotlib, seaborn
- nfl_data_py (nflfastR)