from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import os
import joblib
from app.finance_analysis import analyze_financial_question
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = FastAPI()

df = pd.read_csv("/app/data/sp500_processed.csv", index_col="observation_date", parse_dates=True)

models_dir = "/app/models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "volatility_model.pkl")

@app.get("/")
def read_root():
    return {"message": "Causal ML Server is Running 🚀"}

@app.get("/train-model")
def train_model():
    try:
        X = df[["FED_Rate", "Unemployment_Rate", "CPI"]]
        y = df["Volatility"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        return {"message": "Model trained and saved", "mean_absolute_error": mae}

    except Exception as e:
        return {"error": str(e)}

@app.get("/predict-volatility")
def predict_volatility(fed_rate: float, unemployment: float, cpi: float):
    if not os.path.exists(model_path):
        return {"error": "Model not trained yet. Call /train-model first."}

    model = joblib.load(model_path)
    input_data = np.array([[fed_rate, unemployment, cpi]])
    prediction = model.predict(input_data)[0]

    return {"predicted_volatility": prediction}

@app.get("/analyze")
def analyze_question(question: str = Query(..., title="Economic Question")):
    try:
        response = analyze_financial_question(question)
        return response
    except Exception as e:
        return {"error": str(e)}
