import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


mapping = {"easy": 1, "medium": 2, "hard": 3}
df = pd.read_csv("jee_dataset.csv")


df["maths_difficulty"] = df["maths_difficulty"].map(mapping)
df["physics_difficulty"] = df["physics_difficulty"].map(mapping)
df["chemistry_difficulty"] = df["chemistry_difficulty"].map(mapping)

X = df[[
    "maths_marks",
    "physics_marks",
    "chemistry_marks",
    "maths_difficulty",
    "physics_difficulty",
    "chemistry_difficulty"
]]
y=df["percentile"]


X_train , X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X=df.drop(columns=["percentile", "rank"])
y=df["percentile"]


model =RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
import joblib

joblib.dump(model, "model.pkl")