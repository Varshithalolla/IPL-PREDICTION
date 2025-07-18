import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from llm_utils import get_llm_explanation

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("team_encoder.pkl", "rb"))
teams = list(encoder.classes_)
st.set_page_config(page_title="IPL 2025 Win Predictor")
st.title("IPL 2025 WIN PREDICTOR")
st.image("https://slidechef.net/wp-content/uploads/2025/03/IPL-2025-cover-final.jpg", use_container_width=True)

# Input selections
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team", teams)
with col2:
    bowling_team = st.selectbox("Bowling Team", [team for team in teams if team != batting_team])
venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))

venue = st.selectbox("Venue", list(venue_encoder.classes_))
encoded_venue = venue_encoder.transform([venue])[0]

overs = st.slider("Overs Completed", 3.0, 20.0, step=0.1)
runs = st.number_input("Current Runs", 0, 300)
wickets = st.slider("Wickets Lost", 0, 10)
target = st.number_input("Target Score", min_value=1, max_value=300)

# Compute features
remaining_runs = target - runs
remaining_balls = int((20 - overs) * 6)
crr = runs / overs if overs > 0 else 0
rrr = remaining_runs / (remaining_balls / 6) if remaining_balls > 0 else 0

# Encode teams
batting_encoded = encoder.transform([batting_team])[0]
bowling_encoded = encoder.transform([bowling_team])[0]

# Prepare input dataframe
input_dict = {
    "batting_team": [batting_encoded],
    "bowling_team": [bowling_encoded],
    "venue": [encoded_venue],  # Note: must be used during training
    "over_ball": [overs],
    "current_score": [runs],
    "wickets": [wickets],
    "runs_left": [remaining_runs],
    "balls_left": [remaining_balls],
    "crr": [crr],
    "rrr": [rrr],
}

input_df = pd.DataFrame(input_dict)

# Ensure same feature order used during model training
input_df = input_df[["batting_team", "bowling_team", "venue","over_ball", "current_score",
                     "wickets", "runs_left", "balls_left", "crr", "rrr"]]

# Predict button
if st.button("Predict Winner"):
    try:
        # Predict
        proba = model.predict_proba(input_df)[0]
        prob = np.max(proba)

        batting_prob = proba[1] * 100
        bowling_prob = proba[0] * 100

        # Output results
        st.metric(label=f"Win % — {batting_team}", value=f"{batting_prob:.2f}%")
        st.metric(label=f"Win % — {bowling_team}", value=f"{bowling_prob:.2f}%")

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie([batting_prob, bowling_prob],
               labels=[batting_team, bowling_team],
               autopct='%1.1f%%',
               startangle=90,
               colors=['#FFBB00', '#0057e7'],
               explode=(0.05, 0))
        ax.axis('equal')
        st.pyplot(fig)

        # LLM Strategic Insight
        explanation = get_llm_explanation(input_df, prob, batting_team, bowling_team, target, venue)
        st.subheader(" LLM Strategic Insight")
        st.markdown(explanation)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
