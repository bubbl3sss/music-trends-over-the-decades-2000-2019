import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("music_popularity_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Music Popularity Predictor 🎵")

# Get user inputs
energy = st.slider("Energy", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 0.0, 250.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)

# Make sure features are in the correct order
features = np.array([[energy, danceability, tempo, valence, acousticness, speechiness]])

# Scale features
features_scaled = scaler.transform(features)

# Predict popularity
pred = model.predict(features_scaled)

st.write(f"Predicted popularity: {pred[0]:.2f}")

