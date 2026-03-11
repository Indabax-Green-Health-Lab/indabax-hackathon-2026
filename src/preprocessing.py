"""
preprocessing.py
----------------
Nettoyage et préparation des données météorologiques
IndabaX Hackathon 2026 — Green Health Lab
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/")
PROCESSED_PATH = Path("data/processed/")


def load_data(filename: str) -> pd.DataFrame:
    """Charge les données brutes depuis le dossier raw."""
    filepath = RAW_PATH / filename
    print(f"📂 Chargement : {filepath}")
    return pd.read_csv(filepath, parse_dates=["date"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et standardise le DataFrame."""
    print("🧹 Nettoyage des données...")

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Colonnes attendues
    expected_cols = [
        "date", "city", "region", "latitude", "longitude",
        "temp_min", "temp_max", "temp_mean",
        "wind_speed", "wind_direction",
        "precipitation", "humidity",
        "sunshine_duration", "solar_radiation"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"⚠️  Colonnes manquantes : {missing}")

    # Traitement des valeurs manquantes (interpolation temporelle par ville)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df.groupby("city")[num_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    # Normaliser les noms de colonnes
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    print(f"✅ Nettoyage terminé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features temporelles et climatiques."""
    print("🔧 Ajout des features...")

    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["season"] = df["month"].map({
        12: "Saison sèche", 1: "Saison sèche", 2: "Saison sèche",
        3: "Transition", 4: "Transition",
        5: "Saison pluvieuse", 6: "Saison pluvieuse",
        7: "Saison pluvieuse", 8: "Saison pluvieuse",
        9: "Saison pluvieuse", 10: "Saison pluvieuse",
        11: "Transition"
    })

    # Indice de stress thermique simplifié
    df["heat_stress_index"] = df["temp_max"] * (1 - df["humidity"] / 100)

    # Conditions favorables à la pollution (vent faible + chaleur)
    df["pollution_risk"] = (
        (df["wind_speed"] < df["wind_speed"].quantile(0.25)) &
        (df["temp_max"] > df["temp_max"].quantile(0.75))
    ).astype(int)

    return df


def save_processed(df: pd.DataFrame, filename: str = "data_clean.csv"):
    """Sauvegarde les données nettoyées."""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_PATH / filename
    df.to_csv(out, index=False)
    print(f"💾 Données sauvegardées : {out}")


if __name__ == "__main__":
    # Pipeline complet
    df = load_data("indabax_cameroon_2026.csv")
    df = clean_data(df)
    df = add_features(df)
    save_processed(df)
    print("\n📊 Aperçu :")
    print(df.head())
    print(df.dtypes)
