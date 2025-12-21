import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("music_popularity_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎵 Music Popularity Predictor")

st.write("Adjust the song features to predict popularity:")

danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 60, 200, 120)
valence = st.slider("Valence", 0.0, 1.0, 0.5)

features = np.array([[danceability, energy, tempo, valence]])
features_scaled = scaler.transform(features)

if st.button("Predict Popularity"):
    prediction = model.predict(features_scaled)
    st.success(f"Predicted Popularity Score: {int(prediction[0])}")
