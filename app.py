import streamlit as st
import numpy as np
import joblib

# Load trained model (IMPORTANT: save model first)
model = joblib.load("model.pkl")

st.title("📊 JEE Percentile Predictor")

st.write("Enter your details below:")

# -------------------------
# INPUTS
# -------------------------
maths = st.number_input("Maths Marks", 0, 100)
physics = st.number_input("Physics Marks", 0, 100)
chemistry = st.number_input("Chemistry Marks", 0, 100)

difficulty_map = {"easy": 1, "medium": 2, "hard": 3}

maths_diff = st.selectbox("Maths Difficulty", list(difficulty_map.keys()))
physics_diff = st.selectbox("Physics Difficulty", list(difficulty_map.keys()))
chemistry_diff = st.selectbox("Chemistry Difficulty", list(difficulty_map.keys()))

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Percentile"):
    
    input_data = np.array([[
        maths,
        physics,
        chemistry,
        difficulty_map[maths_diff],
        difficulty_map[physics_diff],
        difficulty_map[chemistry_diff]
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Percentile: {prediction:.2f}")