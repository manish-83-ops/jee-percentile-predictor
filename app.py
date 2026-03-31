import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

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
    st.subheader("📊 Dataset Overview")

df = pd.read_csv("jee_dataset.csv")

fig, ax = plt.subplots()
ax.scatter(df["maths_marks"], df["percentile"], label="Maths", alpha=0.5)
ax.scatter(df["physics_marks"], df["percentile"], label="Physics", alpha=0.5)
ax.scatter(df["chemistry_marks"], df["percentile"], label="Chemistry", alpha=0.5)

ax.set_xlabel("Marks")
ax.set_ylabel("Percentile")
ax.legend()

st.pyplot(fig)
st.subheader("📌 Feature Importance")

importances = model.feature_importances_
features = [
    "Maths Marks",
    "Physics Marks",
    "Chemistry Marks",
    "Maths Diff",
    "Physics Diff",
    "Chemistry Diff"
]

fig2, ax2 = plt.subplots()
ax2.barh(features, importances)
ax2.set_xlabel("Importance")

st.pyplot(fig2)
