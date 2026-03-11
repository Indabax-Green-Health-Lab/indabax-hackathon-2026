"""
model.py
--------
Entraînement et évaluation du modèle de prédiction de la qualité de l'air
IndabaX Hackathon 2026 — Green Health Lab
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

DATA_PATH = Path("data/processed/data_clean.csv")
MODEL_PATH = Path("models/best_model.pkl")


# ──────────────────────────────────────────
# 1. Chargement
# ──────────────────────────────────────────
def load_processed_data():
    print("📂 Chargement des données prétraitées...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


# ──────────────────────────────────────────
# 2. Préparation features / cible
# ──────────────────────────────────────────
FEATURE_COLS = [
    "temp_min", "temp_max", "temp_mean",
    "wind_speed", "precipitation", "humidity",
    "sunshine_duration", "solar_radiation",
    "month", "day_of_year", "heat_stress_index", "pollution_risk"
]

TARGET_COL = "pm25_proxy"   # À adapter selon la variable cible réelle du dataset


def prepare_xy(df: pd.DataFrame):
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


# ──────────────────────────────────────────
# 3. Entraînement
# ──────────────────────────────────────────
def train_model(X_train, y_train):
    print("🤖 Entraînement du modèle XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────
# 4. Évaluation
# ──────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Résultats sur le jeu de test :")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R²   : {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ──────────────────────────────────────────
# 5. Sauvegarde
# ──────────────────────────────────────────
def save_model(model, scaler=None):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Modèle sauvegardé : {MODEL_PATH}")


def load_model():
    return joblib.load(MODEL_PATH)


# ──────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────
if __name__ == "__main__":
    df = load_processed_data()
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"🔀 Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
