#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
TABLEAU DE BORD - QUALITÉ DE L'AIR AU CAMEROUN
Hackathon IndabaX Cameroon 2026
==========================================================================
 Lancer avec : streamlit run dashboard_indabax_corrige_final_fixed.py
==========================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from datetime import date, datetime
from joblib import load as joblib_load

def json_converter(obj):
    """Convertit les types pandas/numpy/datetime en types JSON simples."""
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Categorical):
        return obj.astype(str).tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def payload_vers_json_bytes(payload):
    """Sérialise de manière robuste un payload JSON puis le renvoie en bytes UTF-8."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=json_converter,
    ).encode("utf-8")


def get_target_config(target, task_type):
    configs = {
        "proxy_pm25": {
            "label": "Proxy PM2.5",
            "short_label": "PM2.5",
            "unit": "/100",
            "threshold_label": "Seuil d'alerte",
            "bands": [0, 30, 50, 70, 100],
            "band_labels": ["Bon", "Modéré", "Mauvais", "Dangereux"],
        },
        "score_risque": {
            "label": "Score de risque",
            "short_label": "Score risque",
            "unit": "/100",
            "threshold_label": "Seuil opérationnel",
            "bands": [0, 30, 50, 70, 100],
            "band_labels": ["Faible", "Modéré", "Élevé", "Critique"],
        },
        "indice_stagnation": {
            "label": "Indice de stagnation",
            "short_label": "Stagnation",
            "unit": "/100",
            "threshold_label": "Seuil de vigilance",
            "bands": [0, 30, 50, 70, 100],
            "band_labels": ["Faible", "Modéré", "Élevé", "Critique"],
        },
        "alerte_pollution": {
            "label": "Alerte pollution",
            "short_label": "Alerte",
            "unit": "",
            "threshold_label": "Seuil de décision",
            "bands": None,
            "band_labels": None,
        },
    }
    cfg = configs.get(target, {
        "label": target,
        "short_label": target,
        "unit": "",
        "threshold_label": "Seuil",
        "bands": [0, 30, 50, 70, 100] if task_type == "regression" else None,
        "band_labels": ["Faible", "Modéré", "Élevé", "Critique"] if task_type == "regression" else None,
    }).copy()
    cfg["task_type"] = task_type
    return cfg


def get_target_wording(config):
    label = config["label"]
    target = label.lower()
    if config["task_type"] == "classification":
        return {
            "app_title": "Alerte environnementale - Cameroun",
            "subtitle": "Suivi intelligent des alertes et signaux environnementaux",
            "map_title": f"Carte des niveaux de {label} par ville",
            "analysis_title": f"Analyse des alertes - {label}",
            "climate_title": f"Relations climatiques et {label}",
            "recommendations_title": "Recommandations automatiques",
            "model_explainer": f"Le modèle anticipe {target} à partir des conditions météo récentes et des signaux observés.",
            "wind_title": f"Direction du vent et {label}",
            "wind_explainer": "Cette vue aide à repérer si certains régimes de vent coïncident avec davantage d'alertes.",
            "data_title": f"Données liées à {label}",
        }
    return {
        "app_title": "Indicateurs environnementaux - Cameroun",
        "subtitle": f"Analyse et prévision de {target}",
        "map_title": f"Carte de {label} par ville",
        "analysis_title": f"Analyse de {label}",
        "climate_title": f"Conditions climatiques et {label}",
        "recommendations_title": "Recommandations automatiques",
        "model_explainer": f"Le modèle estime {target} à partir des conditions météo récentes, des lags et des moyennes glissantes.",
        "wind_title": f"Direction du vent et {label}",
        "wind_explainer": f"Cette vue montre comment les régimes de vent s'accompagnent de variations de {target}.",
        "data_title": f"Données de {label}",
    }


def ajouter_niveaux_target(df_in, value_col, config, output_col="niveau"):
    df = df_in.copy()
    if config.get("task_type") == "classification":
        df[output_col] = pd.Series(df[value_col]).map({0: "Pas d'alerte", 1: "Alerte"})
    elif config.get("bands") and config.get("band_labels"):
        df[output_col] = pd.cut(
            df[value_col],
            bins=config["bands"],
            labels=config["band_labels"],
            include_lowest=True,
        )
    return df


def format_metric_value(value, config):
    if pd.isna(value):
        return "N/A"
    if config.get("task_type") == "classification":
        return f"{float(value):.0f}"
    return f"{float(value):.1f}{config.get('unit', '')}"


def get_model_metrics_display(task_type, resume_modele_exporte, r2, rmse, mae):
    if task_type == "classification":
        if resume_modele_exporte:
            best = resume_modele_exporte.get("meilleur_modele", {})
            metriques = best.get("metriques", {})
            return [
                ("Accuracy", metriques.get("Accuracy")),
                ("F1", metriques.get("F1")),
                ("Recall", metriques.get("Recall")),
            ]
        return [("Accuracy", None), ("F1", None), ("Recall", None)]
    return [("R²", r2), ("RMSE", rmse), ("MAE", mae)]

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Qualité de l'Air - Cameroun",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CHEMIN DU FICHIER
# ============================================================
FILEPATH = r"C:\Users\hp\Desktop\TP_exp_AS2\Projet Hackaton\DATASET_FINAL_INDABAX2026.xlsx"
RANDOM_STATE = 42
FAST_SAMPLE_SCATTER = 2000
MODEL_ARTEFACT_PATH = os.path.join("outputs_pipeline_hackathon", "meilleur_modele.joblib")
MODEL_METRICS_PATH = os.path.join("outputs_pipeline_hackathon", "resume_metriques.json")


def calculer_tendance_recente(series):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(s) >= 14:
        return float((s.tail(7).mean() - s.tail(14).head(7).mean()) / 7.0)
    if len(s) >= 2:
        return float((s.iloc[-1] - s.iloc[0]) / max(len(s) - 1, 1))
    return 0.0


def borner_projection(col, value):
    if pd.isna(value):
        return value
    col = str(col)
    if "humidite" in col:
        return float(np.clip(value, 0, 100))
    if "direction_vent" in col:
        return float(value % 360)
    if any(token in col for token in ["vent", "rafales", "rayonnement", "precipitation", "pluie", "duree", "hauteur_couche_limite", "evapotranspiration"]):
        return float(max(0.0, value))
    return float(value)


def inferer_saison(df_ville, mois, fallback=None):
    if "saison" not in df_ville.columns:
        return fallback
    candidats = df_ville.loc[df_ville["mois"] == mois, "saison"].dropna()
    if len(candidats):
        return candidats.mode().iloc[0]
    if fallback is not None:
        return fallback
    saisons = df_ville["saison"].dropna()
    return saisons.mode().iloc[0] if len(saisons) else None


@st.cache_resource(show_spinner=False)
def charger_modele_exporte(filepath):
    if not os.path.exists(filepath):
        return None
    return joblib_load(filepath)


@st.cache_data(show_spinner=False)
def charger_resume_modele(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def encoder_features_pour_modele(df_features, artefact):
    preprocessing = artefact.get("preprocessing", {})
    feature_cols = artefact.get("feature_cols", [])
    cat_cols = preprocessing.get("cat_cols", [])
    num_cols = preprocessing.get("num_cols", [])
    medianes = preprocessing.get("medianes_numeriques", {})
    cat_mappings = preprocessing.get("cat_mappings", {})

    X = df_features.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(medianes.get(col, 0.0))

    for col in cat_cols:
        mapping = cat_mappings.get(col, {})
        X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(np.int32)

    return X[feature_cols].astype(np.float32)


@st.cache_data(show_spinner=False)
def prevoir_proxy_par_modele(df_source, ville, horizon, _artefact):
    if _artefact is None or "ville" not in df_source.columns or "date" not in df_source.columns:
        return pd.DataFrame()

    historique = (
        df_source.loc[df_source["ville"] == ville]
        .sort_values("date")
        .copy()
    )
    if len(historique) < 14:
        return pd.DataFrame()

    working = historique.copy()
    fallback_saison = None
    if "saison" in working.columns and len(working["saison"].dropna()):
        fallback_saison = working["saison"].dropna().iloc[-1]

    colonnes_projection = [
        c for c in [
            "temp_max", "temp_min", "temp_moy", "ressenti_max", "ressenti_min", "ressenti_moy",
            "point_rosee_moy", "humidite_moy", "humidite_min", "humidite_max",
            "pression_mer_moy", "pression_surface_moy", "couverture_nuageuse_moy",
            "nuages_bas_moy", "nuages_moyens_moy", "nuages_hauts_moy",
            "hauteur_couche_limite_moy", "hauteur_couche_limite_min",
            "vitesse_vent_max", "vitesse_vent_10m_moy", "direction_vent", "direction_vent_10m_moy",
            "rafales_max", "precipitation", "heures_precipitation",
            "rayonnement_solaire", "rayonnement_court_total", "rayonnement_court_moy",
            "rayonnement_direct_total", "rayonnement_direct_moy",
            "rayonnement_diffus_total", "rayonnement_diffus_moy",
            "duree_jour", "duree_ensoleillement",
            "humidite_sol_0_7cm", "humidite_sol_7_28cm", "evapotranspiration", "evapotranspiration_fao",
        ] if c in working.columns
    ]
    lag_vars = [c for c in ["temp_max", "precipitation", "vitesse_vent_max", "humidite_moy", "hauteur_couche_limite_moy"] if c in working.columns]

    lignes_futures = []
    for _ in range(horizon):
        precedent = working.iloc[-1].copy()
        prochaine_date = pd.to_datetime(precedent["date"]) + pd.Timedelta(days=1)
        nouvelle_ligne = precedent.copy()
        nouvelle_ligne["date"] = prochaine_date
        nouvelle_ligne["annee"] = prochaine_date.year
        nouvelle_ligne["mois"] = prochaine_date.month
        nouvelle_ligne["jour_annee"] = prochaine_date.dayofyear
        if "trimestre" in working.columns:
            nouvelle_ligne["trimestre"] = int(((prochaine_date.month - 1) // 3) + 1)
        if "jour_semaine" in working.columns:
            nouvelle_ligne["jour_semaine"] = int(prochaine_date.dayofweek)
        if "saison" in working.columns:
            nouvelle_ligne["saison"] = inferer_saison(historique, prochaine_date.month, fallback=fallback_saison)

        for col in colonnes_projection:
            historique_col = pd.to_numeric(working[col], errors="coerce").dropna()
            base = float(historique_col.iloc[-1]) if len(historique_col) else 0.0
            tendance = calculer_tendance_recente(historique_col.tail(14))
            nouvelle_ligne[col] = borner_projection(col, base + tendance)

        if "mois_sin" in working.columns:
            nouvelle_ligne["mois_sin"] = np.sin(2 * np.pi * prochaine_date.month / 12)
        if "mois_cos" in working.columns:
            nouvelle_ligne["mois_cos"] = np.cos(2 * np.pi * prochaine_date.month / 12)
        if "vent_sin" in working.columns and "direction_vent" in nouvelle_ligne.index:
            nouvelle_ligne["vent_sin"] = np.sin(np.deg2rad(nouvelle_ligne["direction_vent"]))
        if "vent_cos" in working.columns and "direction_vent" in nouvelle_ligne.index:
            nouvelle_ligne["vent_cos"] = np.cos(np.deg2rad(nouvelle_ligne["direction_vent"]))
        if {"temp_moy", "point_rosee_moy"}.issubset(working.columns):
            nouvelle_ligne["ecart_point_rosee"] = nouvelle_ligne["temp_moy"] - nouvelle_ligne["point_rosee_moy"]
        if {"rafales_max", "vitesse_vent_max"}.issubset(working.columns):
            nouvelle_ligne["ratio_rafales"] = nouvelle_ligne["rafales_max"] / (nouvelle_ligne["vitesse_vent_max"] + 0.1)
        if "pression_mer_moy" in working.columns:
            nouvelle_ligne["pression_anomalie"] = nouvelle_ligne["pression_mer_moy"] - 1013.25
        if {"temp_max", "ressenti_max"}.issubset(working.columns):
            nouvelle_ligne["ecart_ressenti"] = nouvelle_ligne["temp_max"] - nouvelle_ligne["ressenti_max"]
        if {"duree_ensoleillement", "duree_jour"}.issubset(working.columns):
            nouvelle_ligne["ratio_ensoleillement"] = nouvelle_ligne["duree_ensoleillement"] / (nouvelle_ligne["duree_jour"] + 0.01)

        for col in lag_vars:
            serie_hist = pd.to_numeric(working[col], errors="coerce")
            if f"{col}_lag1" in working.columns:
                nouvelle_ligne[f"{col}_lag1"] = float(serie_hist.iloc[-1])
            if f"{col}_lag7" in working.columns:
                lag_idx = -7 if len(serie_hist) >= 7 else -1
                nouvelle_ligne[f"{col}_lag7"] = float(serie_hist.iloc[lag_idx])
            if f"{col}_moy7j" in working.columns:
                nouvelle_ligne[f"{col}_moy7j"] = float(serie_hist.tail(7).mean())

        lignes_futures.append(nouvelle_ligne.copy())
        working = pd.concat([working, pd.DataFrame([nouvelle_ligne])], ignore_index=True)

    futur = pd.DataFrame(lignes_futures)
    if futur.empty:
        return futur

    X_futur = encoder_features_pour_modele(futur, _artefact)
    preprocessing = _artefact.get("preprocessing", {})
    scaler = _artefact.get("scaler")
    model = _artefact.get("model")
    utiliser_scaler = preprocessing.get("scaler_required", False) and scaler is not None
    X_input = scaler.transform(X_futur.values) if utiliser_scaler else X_futur.values

    predictions = model.predict(X_input)
    futur["prediction"] = predictions

    if _artefact.get("task_type") == "classification":
        if hasattr(model, "predict_proba"):
            futur["score_prediction"] = model.predict_proba(X_input)[:, 1]
        futur["niveau"] = futur["prediction"].map({0: "Pas d'alerte", 1: "Alerte"})
    else:
        cible = _artefact.get("target")
        if cible in {"proxy_pm25", "score_risque", "indice_stagnation"}:
            futur["prediction"] = np.clip(futur["prediction"], 0, 100)
        if cible == "proxy_pm25":
            futur["niveau"] = pd.cut(
                futur["prediction"],
                bins=[0, 30, 50, 70, 100],
                labels=["Bon", "Modéré", "Mauvais", "Dangereux"],
                include_lowest=True,
            )

    futur["source_prevision"] = f"Artefact pipeline - {_artefact.get('model_name', 'modèle exporté')}"
    colonnes_sortie = [c for c in ["date", "prediction", "niveau", "score_prediction", "source_prevision"] if c in futur.columns]
    return futur[colonnes_sortie].reset_index(drop=True)


# ============================================================
# TRADUCTION ET NETTOYAGE
# ============================================================
RENAME = {
    "id":"identifiant","date":"date","city":"ville","region":"region",
    "latitude":"latitude","longitude":"longitude",
    "zone_climatique":"zone_climatique",
    "mois":"mois","annee":"annee","trimestre":"trimestre",
    "saison":"saison","jour_semaine":"jour_semaine",
    "temp_max":"temp_max","temp_min":"temp_min","temp_mean":"temp_moy",
    "feels_like_max":"ressenti_max","feels_like_min":"ressenti_min",
    "feels_like_mean":"ressenti_moy",
    "amplitude_thermique":"amplitude_thermique",
    "temperature_2m_mean":"temp_2m_moy","temperature_2m_min":"temp_2m_min",
    "temperature_2m_max":"temp_2m_max",
    "apparent_temperature_mean":"temp_apparente_moy",
    "apparent_temperature_min":"temp_apparente_min",
    "apparent_temperature_max":"temp_apparente_max",
    "relative_humidity_2m_mean":"humidite_moy",
    "relative_humidity_2m_min":"humidite_min",
    "relative_humidity_2m_max":"humidite_max",
    "dew_point_2m_mean":"point_rosee_moy",
    "surface_pressure_mean":"pression_surface_moy",
    "pressure_msl_mean":"pression_mer_moy",
    "cloud_cover_mean":"couverture_nuageuse_moy",
    "cloud_cover_low_mean":"nuages_bas_moy",
    "cloud_cover_mid_mean":"nuages_moyens_moy",
    "cloud_cover_high_mean":"nuages_hauts_moy",
    "boundary_layer_height_mean":"hauteur_couche_limite_moy",
    "boundary_layer_height_min":"hauteur_couche_limite_min",
    "wind_speed_max":"vitesse_vent_max","wind_gusts_max":"rafales_max",
    "wind_direction":"direction_vent",
    "wind_speed_10m_mean":"vitesse_vent_10m_moy",
    "wind_speed_10m_max":"vitesse_vent_10m_max",
    "wind_direction_10m_mean":"direction_vent_10m_moy",
    "precipitation":"precipitation","rain":"pluie",
    "snowfall":"chute_neige","precipitation_hours":"heures_precipitation",
    "precipitation_sum":"precipitation_totale",
    "precipitation_max":"precipitation_max_horaire",
    "rain_sum":"pluie_totale","rain_max":"pluie_max_horaire",
    "snowfall_sum":"chute_neige_totale",
    "sunrise":"lever_soleil","sunset":"coucher_soleil",
    "daylight_duration":"duree_jour","sunshine_duration":"duree_ensoleillement",
    "solar_radiation":"rayonnement_solaire",
    "shortwave_radiation_sum":"rayonnement_court_total",
    "shortwave_radiation_mean":"rayonnement_court_moy",
    "direct_radiation_sum":"rayonnement_direct_total",
    "direct_radiation_mean":"rayonnement_direct_moy",
    "diffuse_radiation_sum":"rayonnement_diffus_total",
    "diffuse_radiation_mean":"rayonnement_diffus_moy",
    "soil_moisture_0_to_7cm_mean":"humidite_sol_0_7cm",
    "soil_moisture_7_to_28cm_mean":"humidite_sol_7_28cm",
    "evapotranspiration":"evapotranspiration",
    "et0_fao_evapotranspiration_sum":"evapotranspiration_fao",
    "weather_code":"code_meteo","categorie_meteo":"categorie_meteo",
    "indice_confort":"indice_confort","indice_secheresse":"indice_secheresse",
    "lat":"lat_doublon","lon":"lon_doublon",
}

DROP_COLS = [
    "lat_doublon","lon_doublon",
    "temp_2m_moy","temp_2m_min","temp_2m_max",
    "temp_apparente_moy","temp_apparente_min","temp_apparente_max",
    "chute_neige","chute_neige_totale",
    "precipitation_totale","pluie","pluie_totale",
    "identifiant","vitesse_vent_10m_max",
]

COMPOSANTES_BRUTES = {
    "vitesse_vent_max","hauteur_couche_limite_moy",
    "precipitation","temp_max","humidite_moy",
    "rayonnement_solaire","rayonnement_court_total",
}

# ============================================================
# CHARGEMENT ET PRÉPARATION (avec cache Streamlit)
# ============================================================
@st.cache_data(show_spinner="Chargement des données...")
def charger_et_preparer(filepath):
    """Charge, nettoie, crée le proxy PM2.5 et entraîne le modèle."""

    # Chargement
    df = pd.read_excel(filepath, sheet_name=0, header=0, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip().str.replace("\xa0"," ",regex=False)
    df = df.rename(columns=RENAME)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Temporel
    if "annee" not in df.columns: df["annee"] = df["date"].dt.year
    if "mois" not in df.columns:  df["mois"] = df["date"].dt.month
    else: df["mois"] = pd.to_numeric(df["mois"], errors="coerce")
    df["jour_annee"] = df["date"].dt.day_of_year

    for col in ["duree_jour","duree_ensoleillement"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().median() > 100:
                df[col] = df[col] / 3600.0

    # Tri
    sort_cols = [c for c in ["ville","date"] if c in df.columns]
    if sort_cols: df = df.sort_values(sort_cols).reset_index(drop=True)

    # Cycliques
    if "mois" in df.columns:
        df["mois_sin"] = np.sin(2*np.pi*df["mois"]/12)
        df["mois_cos"] = np.cos(2*np.pi*df["mois"]/12)
    if "direction_vent" in df.columns:
        df["vent_sin"] = np.sin(np.deg2rad(df["direction_vent"]))
        df["vent_cos"] = np.cos(np.deg2rad(df["direction_vent"]))

    # Dérivées
    if {"temp_moy","point_rosee_moy"}.issubset(df.columns):
        df["ecart_point_rosee"] = df["temp_moy"] - df["point_rosee_moy"]
    if {"rafales_max","vitesse_vent_max"}.issubset(df.columns):
        df["ratio_rafales"] = df["rafales_max"]/(df["vitesse_vent_max"]+0.1)
    if "pression_mer_moy" in df.columns:
        df["pression_anomalie"] = df["pression_mer_moy"] - 1013.25
    if {"temp_max","ressenti_max"}.issubset(df.columns):
        df["ecart_ressenti"] = df["temp_max"] - df["ressenti_max"]
    if {"duree_ensoleillement","duree_jour"}.issubset(df.columns):
        df["ratio_ensoleillement"] = df["duree_ensoleillement"]/(df["duree_jour"]+0.01)

    # Lags
    group = "ville" if "ville" in df.columns else None
    lag_vars = [c for c in ["temp_max","precipitation","vitesse_vent_max",
                            "humidite_moy","hauteur_couche_limite_moy"] if c in df.columns]
    if group and lag_vars:
        for lag in [1, 7]:
            for col in lag_vars:
                df[f"{col}_lag{lag}"] = df.groupby(group)[col].shift(lag)
        for col in lag_vars:
            df[f"{col}_moy7j"] = (
                df.groupby(group)[col]
                .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
            )

    # Proxy PM2.5
    mask_train = df["annee"] <= (df["annee"].max() - 1)

    def norm01(series):
        s = pd.to_numeric(series, errors="coerce")
        vmin, vmax = s[mask_train].min(), s[mask_train].max()
        d = vmax - vmin
        if d == 0 or pd.isna(d): return pd.Series(0.5, index=s.index)
        return ((s - vmin)/d).clip(0, 1)

    proxy = pd.Series(0.0, index=df.index)
    if "vitesse_vent_max" in df.columns:
        proxy += 0.25 * (1.0 - norm01(df["vitesse_vent_max"]))
    if "hauteur_couche_limite_moy" in df.columns:
        proxy += 0.20 * (1.0 - norm01(df["hauteur_couche_limite_moy"]))
    if "precipitation" in df.columns:
        proxy += 0.15 * (1.0 - norm01(np.log1p(df["precipitation"])))
    if "temp_max" in df.columns:
        proxy += 0.15 * norm01(df["temp_max"])
    if "humidite_moy" in df.columns:
        proxy += 0.15 * (1.0 - norm01(df["humidite_moy"]))
    ray = "rayonnement_solaire" if "rayonnement_solaire" in df.columns else "rayonnement_court_total"
    if ray in df.columns:
        proxy += 0.10 * norm01(df[ray])
    df["proxy_pm25"] = proxy * 100.0

    seuil = df.loc[mask_train, "proxy_pm25"].quantile(0.80)
    df["alerte_pollution"] = (df["proxy_pm25"] >= seuil).astype(int)
    df["niveau_alerte"] = pd.cut(
        df["proxy_pm25"],
        bins=[0, 30, 50, 70, 100],
        labels=["Bon", "Modéré", "Mauvais", "Dangereux"],
        include_lowest=True,
    )

    # Entraînement IA - version corrigée et optimisée
    toutes_cibles = {"proxy_pm25","alerte_pollution","niveau_alerte","indice_stagnation","score_risque"}
    exclure = toutes_cibles | {
        "date","identifiant","lever_soleil","coucher_soleil",
        "pluie_log","dispersion_atmo","ville",
    }
    cat_cols = [c for c in ["region","zone_climatique","saison","categorie_meteo"] if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclure and c not in cat_cols and c != "proxy_pm25"
                and pd.api.types.is_numeric_dtype(df[c])]

    feature_cols = num_cols + cat_cols
    df_clean = df.dropna(subset=["proxy_pm25"]).reset_index(drop=True)
    mask_tr = df_clean["annee"] <= (df_clean["annee"].max() - 1)
    mask_te = ~mask_tr

    X_tr = df_clean.loc[mask_tr, feature_cols].copy()
    X_te = df_clean.loc[mask_te, feature_cols].copy()
    y_tr = df_clean.loc[mask_tr, "proxy_pm25"].values.astype(np.float32)
    y_te = df_clean.loc[mask_te, "proxy_pm25"].values.astype(np.float32)

    for col in cat_cols:
        combined = pd.concat([X_tr[col], X_te[col]]).astype("category")
        codes = combined.cat.codes
        X_tr[col] = codes.iloc[:len(X_tr)].values
        X_te[col] = codes.iloc[len(X_tr):].values
    for col in num_cols:
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_te[col] = X_te[col].fillna(med)

    n_train = len(X_tr)
    split_idx = max(int(n_train * 0.85), 1)
    X_fit = X_tr.iloc[:split_idx].values.astype(np.float32)
    y_fit = y_tr[:split_idx]
    X_val = X_tr.iloc[split_idx:].values.astype(np.float32)
    y_val = y_tr[split_idx:] if split_idx < n_train else y_tr[:0]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.06, max_depth=6,
        num_leaves=31, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=20, reg_alpha=0.05, reg_lambda=0.1,
        n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
    )

    if len(y_val) > 20:
        model_lgb.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
    else:
        model_lgb.fit(X_tr.values.astype(np.float32), y_tr)

    y_pred = model_lgb.predict(X_te.values.astype(np.float32))

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)

    imp = pd.Series(model_lgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    model = {"lightgbm": model_lgb}

    return df, model, r2, rmse, mae, imp, seuil, feature_cols




# ============================================================
# OUTILS PRÉVISION / API
# ============================================================
@st.cache_data(show_spinner=False)
def prevoir_proxy_par_ville(df_source, ville, horizon=7, target_col="proxy_pm25", target_config=None):
    """Prévision rapide des prochains jours à partir de la série quotidienne proxy_pm25."""
    if target_config is None:
        target_config = get_target_config(target_col, "regression")
    serie = (
        df_source.loc[df_source["ville"] == ville, ["date", target_col]]
        .dropna()
        .groupby("date", as_index=False)[target_col].mean()
        .sort_values("date")
    )
    if len(serie) < 7:
        return pd.DataFrame(columns=["date", "prediction", "niveau"])

    y = serie.set_index("date")[target_col].asfreq("D").interpolate(limit_direction="both")
    last7 = float(y.tail(7).mean())
    prev7 = float(y.tail(14).head(7).mean()) if len(y) >= 14 else last7
    drift = (last7 - prev7) / 7.0

    future_idx = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = []
    current = last7
    for _ in range(horizon):
        current = max(0.0, min(100.0, current + drift))
        preds.append(current)

    out = pd.DataFrame({"date": future_idx, "prediction": preds})
    out["niveau"] = pd.cut(
        out["prediction"],
        bins=[0, 30, 50, 70, 100],
        labels=["Bon", "Modéré", "Mauvais", "Dangereux"],
        include_lowest=True,
    )
    return out

@st.cache_data(show_spinner=False)
def construire_payload_api(df_source, forecast_df, seuil_alerte):
    latest_date = pd.to_datetime(df_source["date"]).max() if "date" in df_source.columns else None

    forecast_records = []
    if len(forecast_df):
        forecast_clean = forecast_df.copy()
        for col in forecast_clean.columns:
            if pd.api.types.is_datetime64_any_dtype(forecast_clean[col]):
                forecast_clean[col] = pd.to_datetime(forecast_clean[col], errors="coerce").dt.strftime("%Y-%m-%d")
            elif isinstance(forecast_clean[col].dtype, pd.CategoricalDtype):
                forecast_clean[col] = forecast_clean[col].astype(str)
        forecast_records = forecast_clean.to_dict(orient="records")

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "latest_data_date": None if pd.isna(latest_date) else latest_date.isoformat(),
        "kpis": {
            "proxy_pm25_mean": round(float(df_source["proxy_pm25"].mean()), 2),
            "alert_days_pct": round(float(df_source["alerte_pollution"].mean() * 100), 2),
            "threshold_alert": round(float(seuil_alerte), 2),
            "cities": int(df_source["ville"].nunique()) if "ville" in df_source.columns else 0,
            "regions": int(df_source["region"].nunique()) if "region" in df_source.columns else 0,
        },
        "forecast_next_days": forecast_records,
        "forecast_source": forecast_records[0].get("source_prevision") if forecast_records and "source_prevision" in forecast_records[0] else "projection_tendance",
    }
    return payload


@st.cache_data(show_spinner=False)
def prevoir_cible_par_ville(df_source, ville, horizon=7, target_col="proxy_pm25", target_config=None):
    if target_config is None:
        target_config = get_target_config(target_col, "regression")
    serie = (
        df_source.loc[df_source["ville"] == ville, ["date", target_col]]
        .dropna()
        .groupby("date", as_index=False)[target_col].mean()
        .sort_values("date")
    )
    if len(serie) < 7:
        return pd.DataFrame(columns=["date", "prediction", "niveau"])

    y = serie.set_index("date")[target_col].asfreq("D").interpolate(limit_direction="both")
    last7 = float(y.tail(7).mean())
    prev7 = float(y.tail(14).head(7).mean()) if len(y) >= 14 else last7
    drift = (last7 - prev7) / 7.0

    future_idx = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = []
    current = last7
    for _ in range(horizon):
        current = current + drift
        if target_config.get("task_type") != "classification":
            current = max(0.0, min(100.0, current))
        preds.append(current)

    out = pd.DataFrame({"date": future_idx, "prediction": preds})
    out = ajouter_niveaux_target(out, "prediction", target_config)
    return out


def construire_payload_api_cible(df_source, forecast_df, seuil_alerte, target_col, target_config):
    payload = construire_payload_api(df_source, forecast_df, seuil_alerte)
    payload["kpis"]["target_name"] = target_col
    payload["kpis"]["target_label"] = target_config.get("label")
    payload["kpis"]["task_type"] = target_config.get("task_type")
    payload["kpis"]["target_mean"] = round(float(df_source[target_col].mean()), 2) if target_col in df_source.columns else None
    return payload


@st.cache_data(show_spinner=False)
def calculer_score_risque_ville(df_source):
    """Construit un score de risque 0-100 par ville."""
    if df_source.empty or "ville" not in df_source.columns:
        return pd.DataFrame(columns=["ville", "region", "score_risque", "pm25_moyen", "pct_alerte", "tendance_7j", "niveau_risque"])

    base = df_source.groupby("ville", as_index=False).agg(
        region=("region", "first"),
        pm25_moyen=("proxy_pm25", "mean"),
        pct_alerte=("alerte_pollution", "mean"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )

    # tendance récente
    if "date" in df_source.columns:
        recent = (
            df_source.sort_values("date")
            .groupby(["ville", pd.Grouper(key="date", freq="D")])["proxy_pm25"]
            .mean()
            .reset_index()
        )
        tendances = []
        for ville, g in recent.groupby("ville"):
            g = g.sort_values("date").tail(14)
            if len(g) >= 8:
                last7 = g.tail(7)["proxy_pm25"].mean()
                prev7 = g.head(len(g)-7)["proxy_pm25"].mean() if len(g) > 7 else g.tail(7)["proxy_pm25"].mean()
                tendance = float(last7 - prev7)
            else:
                tendance = 0.0
            tendances.append((ville, tendance))
        tendance_df = pd.DataFrame(tendances, columns=["ville", "tendance_7j"])
        base = base.merge(tendance_df, on="ville", how="left")
    else:
        base["tendance_7j"] = 0.0

    def scale01(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().all():
            return pd.Series(0.0, index=s.index)
        vmin, vmax = s.min(), s.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            return pd.Series(0.5, index=s.index)
        return ((s - vmin) / (vmax - vmin)).clip(0, 1)

    score = (
        0.55 * scale01(base["pm25_moyen"]) +
        0.30 * scale01(base["pct_alerte"] * 100) +
        0.15 * scale01(base["tendance_7j"].clip(lower=0))
    ) * 100

    base["score_risque"] = score.round(1)
    base["pct_alerte"] = (base["pct_alerte"] * 100).round(1)
    base["tendance_7j"] = base["tendance_7j"].round(1)
    base["niveau_risque"] = pd.cut(
        base["score_risque"],
        bins=[0, 30, 50, 70, 100],
        labels=["Faible", "Modéré", "Élevé", "Critique"],
        include_lowest=True,
    )
    return base.sort_values(["score_risque", "pm25_moyen"], ascending=[False, False]).reset_index(drop=True)

def generer_recommandations(df_source, score_ville, seuil_alerte):
    """Génère des recommandations automatiques basées sur les données filtrées."""
    recos = []
    pm25 = float(df_source["proxy_pm25"].mean()) if len(df_source) else 0.0
    pct = float(df_source["alerte_pollution"].mean() * 100) if len(df_source) else 0.0

    if pm25 >= 55:
        recos.append("🔴 **Renforcer l’alerte sanitaire** : diffuser des messages radio/SMS, limiter les activités physiques extérieures et protéger les personnes vulnérables.")
    elif pm25 >= 40:
        recos.append("🟠 **Mettre le système de veille en vigilance** : communication préventive, suivi quotidien des zones sensibles et sensibilisation des écoles/hôpitaux.")
    else:
        recos.append("🟢 **Maintenir la surveillance de routine** : la qualité de l’air reste globalement acceptable, mais la veille météo doit continuer.")

    if pct >= 20:
        recos.append("🚨 **Déployer un protocole régional d’alerte** dans les zones où les jours d’alerte sont fréquents.")
    if "vitesse_vent_max" in df_source.columns and df_source["vitesse_vent_max"].mean() < 12:
        recos.append("💨 **Risque de stagnation atmosphérique** : surveiller les épisodes de vent faible, propices à l’accumulation des polluants.")
    if "precipitation" in df_source.columns and df_source["precipitation"].mean() < 1:
        recos.append("🌧️ **Absence de lessivage atmosphérique** : prévoir un suivi renforcé lors des périodes sèches.")
    if "temp_max" in df_source.columns and df_source["temp_max"].mean() > 32:
        recos.append("🌡️ **Chaleur élevée** : surveiller les réactions photochimiques et les risques respiratoires.")
    if len(score_ville):
        top = score_ville.iloc[0]
        recos.append(f"📍 **Ville prioritaire** : concentrer la surveillance sur **{top['ville']}** ({top['region']}) avec un score de risque de **{top['score_risque']}/100**.")
        if len(score_ville) >= 3:
            top3 = ", ".join(score_ville.head(3)["ville"].tolist())
            recos.append(f"🏙️ **Top 3 villes à suivre** : {top3}.")
    if seuil_alerte:
        recos.append(f"📏 **Seuil opérationnel actuel** : considérer les journées avec **{target_config['label']}** >= **{seuil_alerte:.1f}** comme prioritaires.")
    return recos

def construire_alerte_region(df_source):
    """Tableau des alertes sanitaires par région."""
    if df_source.empty or "region" not in df_source.columns:
        return pd.DataFrame(columns=["region", "pm25_moyen", "pct_alerte", "score_sanitaire", "niveau", "message"])

    reg = df_source.groupby("region", as_index=False).agg(
        pm25_moyen=("proxy_pm25", "mean"),
        pct_alerte=("alerte_pollution", "mean"),
    )
    reg["pct_alerte"] = reg["pct_alerte"] * 100
    reg["score_sanitaire"] = (0.65 * reg["pm25_moyen"] + 0.35 * reg["pct_alerte"]).clip(0, 100).round(1)
    reg["niveau"] = pd.cut(
        reg["score_sanitaire"],
        bins=[0, 30, 50, 70, 100],
        labels=["Vert", "Jaune", "Orange", "Rouge"],
        include_lowest=True,
    )

    def make_msg(row):
        if row["score_sanitaire"] >= 70:
            return "Alerte sanitaire immédiate : communiquer, protéger les publics fragiles, renforcer la surveillance."
        if row["score_sanitaire"] >= 50:
            return "Vigilance élevée : intensifier le suivi, messages préventifs recommandés."
        if row["score_sanitaire"] >= 30:
            return "Surveillance renforcée : risque modéré, suivre l’évolution quotidienne."
        return "Situation globalement stable : veille standard."
    reg["message"] = reg.apply(make_msg, axis=1)
    return reg.sort_values("score_sanitaire", ascending=False).reset_index(drop=True)

# ============================================================
# CHARGEMENT
# ============================================================
try:
    df, model, r2, rmse, mae, importances, seuil_alerte, feature_cols = charger_et_preparer(FILEPATH)
except Exception as e:
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()

artefact_modele = charger_modele_exporte(MODEL_ARTEFACT_PATH)
resume_modele_exporte = charger_resume_modele(MODEL_METRICS_PATH)
modele_jury_actif = artefact_modele is not None
if modele_jury_actif:
    seuil_alerte = artefact_modele.get("preprocessing", {}).get("threshold_alert", seuil_alerte)
active_target = artefact_modele.get("target", "proxy_pm25") if modele_jury_actif else "proxy_pm25"
active_task_type = artefact_modele.get("task_type", "regression") if modele_jury_actif else "regression"
target_config = get_target_config(active_target, active_task_type)
target_wording = get_target_wording(target_config)

# ============================================================
# BARRE LATÉRALE - FILTRES
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Flag_of_Cameroon.svg/200px-Flag_of_Cameroon.svg.png", width=80)
st.sidebar.title("🌍 Filtres")

# Filtre région
regions = sorted(df["region"].dropna().unique())
region_sel = st.sidebar.multiselect("Région", regions, default=regions)

# Filtre ville (dynamique selon la région)
villes_dispo = sorted(df.loc[df["region"].isin(region_sel), "ville"].dropna().unique())
ville_sel = st.sidebar.multiselect("Ville", villes_dispo, default=[])

# Filtre année
annees = sorted(df["annee"].dropna().unique().astype(int))
annee_range = st.sidebar.slider("Période", min_value=min(annees), max_value=max(annees),
                                 value=(min(annees), max(annees)))

# Appliquer les filtres
mask = (df["region"].isin(region_sel)) & (df["annee"] >= annee_range[0]) & (df["annee"] <= annee_range[1])
if ville_sel:
    mask = mask & (df["ville"].isin(ville_sel))
df_filtered = df.loc[mask].copy()
score_ville = calculer_score_risque_ville(df_filtered)
recommandations_auto = generer_recommandations(df_filtered, score_ville, seuil_alerte)
alerte_region_df = construire_alerte_region(df_filtered)
df_filtered = ajouter_niveaux_target(df_filtered, active_target, target_config, output_col="niveau_target")
target_mean = float(df_filtered[active_target].mean()) if active_target in df_filtered.columns and len(df_filtered) else np.nan
model_metrics_display = get_model_metrics_display(active_task_type, resume_modele_exporte, r2, rmse, mae)

# Info dans la sidebar
st.sidebar.caption(
    f"Prévision : {'artefact pipeline connecté' if modele_jury_actif else 'mode dashboard local uniquement'}"
)
st.sidebar.markdown("---")
st.sidebar.metric("Observations filtrées", f"{len(df_filtered):,}")
for label, value in model_metrics_display:
    st.sidebar.metric(label, "N/A" if value is None else f"{float(value):.4f}")
st.sidebar.markdown("---")
st.sidebar.caption("Hackathon IndabaX Cameroon 2026")

# ============================================================
# EN-TÊTE
# ============================================================
st.title(f"🌍 {target_wording['app_title']}")
st.markdown(f"**Hackathon IndabaX Cameroon 2026** - {target_wording['subtitle']}")
st.markdown("---")
st.info("Version corrigée : lecture complète du fichier Excel, code revérifié et optimisé pour éviter les blocages au chargement.")
st.caption(f"Cible active du modèle exporté : `{active_target}` ({active_task_type})")

# ============================================================
# KPI - INDICATEURS PRINCIPAUX
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

pm25_moy = target_mean
pct_alerte = df_filtered["alerte_pollution"].mean() * 100
temp_moy = df_filtered["temp_max"].mean() if "temp_max" in df_filtered.columns else 0
vent_moy = df_filtered["vitesse_vent_max"].mean() if "vitesse_vent_max" in df_filtered.columns else 0
score_risque_moy = score_ville["score_risque"].mean() if len(score_ville) else 0

# Couleur de l'indicateur PM2.5
if pm25_moy < 30: pm_color = "normal"
elif pm25_moy < 50: pm_color = "off"
else: pm_color = "inverse"

col1.metric(
    f"🌫️ {target_config['label']} moyen",
    format_metric_value(pm25_moy, target_config),
    delta=str(df_filtered["niveau_target"].mode().iloc[0]) if "niveau_target" in df_filtered.columns and len(df_filtered["niveau_target"].dropna()) else None,
)
col2.metric(
    "🚨 Jours en alerte",
    f"{pct_alerte:.1f}%",
    delta=f"{target_config['threshold_label']} = {seuil_alerte:.1f}",
)
col3.metric("🌡️ Temp. max moyenne", f"{temp_moy:.1f}°C")
col4.metric("💨 Vent max moyen", f"{vent_moy:.1f} km/h")
col5.metric("🏙️ Score risque moyen", f"{score_risque_moy:.1f}/100")

st.markdown("---")

# Dictionnaire des noms de mois (utilisé dans plusieurs onglets)
mois_noms = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Juin",
             7:"Juil",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}

# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ Carte", "📊 Pollution", "🌡️ Climat vs Pollution",
    "🤖 Modèle IA", "🔮 Prévisions", "📡 API & Mobile", "📋 Données"
])

# ------------------------------------------------------------
# ONGLET 1 : CARTE
# ------------------------------------------------------------
with tab1:
    st.subheader(target_wording["map_title"])

    # Calculer la moyenne par ville
    agg_dict = {
        "target_mean": (active_target, "mean"),
        "latitude": ("latitude", "first"),
        "longitude": ("longitude", "first"),
        "region": ("region", "first"),
        "alerte_pct": ("alerte_pollution", "mean"),
    }
    if "temp_max" in df_filtered.columns:
        agg_dict["temp"] = ("temp_max", "mean")

    carte_data = df_filtered.groupby("ville").agg(**agg_dict).reset_index()
    carte_data.rename(columns={"target_mean": "target_value"}, inplace=True)
    carte_data["alerte_pct"] = (carte_data["alerte_pct"] * 100).round(1)
    carte_data["target_value"] = carte_data["target_value"].round(1)

    # Catégoriser
    carte_data = ajouter_niveaux_target(carte_data, "target_value", target_config)

    fig_carte = px.scatter_mapbox(
        carte_data,
        lat="latitude", lon="longitude",
        size="target_value", color="target_value",
        color_continuous_scale=["green", "yellow", "orange", "red"],
        range_color=[0, 80],
        hover_name="ville",
        hover_data={"region": True, "target_value": ":.1f", "alerte_pct": ":.1f%",
                     "latitude": False, "longitude": False},
        size_max=25,
        zoom=5,
        center={"lat": 5.95, "lon": 10.15},
        mapbox_style="open-street-map",
        title=f"{target_config['label']} moyen par ville",
    )
    fig_carte.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_carte, width="stretch")

    # Légende
    st.markdown("""
    **Légende** : valeurs de la cible active agrégées par ville
    """)

    st.markdown("---")
    st.markdown(f"### {target_wording['recommendations_title']}")
    for reco in recommandations_auto:
        st.markdown(f"- {reco}")

    st.markdown("### Bouton d’alerte sanitaire par région")
    col_alert1, col_alert2 = st.columns([1, 2])
    with col_alert1:
        region_alerte = st.selectbox("Choisir une région", alerte_region_df["region"].tolist() if len(alerte_region_df) else [])
        declencher = st.button("🚨 Notification régionale", width="stretch")
    with col_alert2:
        if len(alerte_region_df):
            region_info = alerte_region_df.loc[alerte_region_df["region"] == region_alerte].iloc[0]
            couleur = {"Vert":"#2ecc71","Jaune":"#f1c40f","Orange":"#e67e22","Rouge":"#e74c3c"}.get(str(region_info["niveau"]), "#95a5a6")
            st.markdown(
                f"<div style='padding:16px;border-radius:12px;background:{couleur};color:white;'>"
                f"<b>{region_info['region']}</b><br>"
                f"Niveau: <b>{region_info['niveau']}</b><br>"
                f"Cible moyenne: <b>{region_info['pm25_moyen']:.1f}</b><br>"
                f"Jours d'alerte: <b>{region_info['pct_alerte']:.1f}%</b><br>"
                f"Score sanitaire: <b>{region_info['score_sanitaire']:.1f}/100</b><br>"
                f"{region_info['message']}"
                f"</div>",
                unsafe_allow_html=True
            )
            if declencher:
                st.error(
                    f"ALERTE SANITAIRE - {region_info['region']} | "
                    f"Niveau {region_info['niveau']} | Score {region_info['score_sanitaire']:.1f}/100. "
                    f"{region_info['message']}"
                )

    st.markdown("### Tableau régional d’alerte")
    if len(alerte_region_df):
        st.dataframe(
            alerte_region_df.rename(columns={
                "region":"Région","pm25_moyen":"PM2.5 moy","pct_alerte":"% alerte",
                "score_sanitaire":"Score sanitaire","niveau":"Niveau","message":"Message"
            }),
            width="stretch",
            height=280,
        )

    st.markdown("### Score de risque par ville")
    if len(score_ville):
        col_r1, col_r2 = st.columns([1.15, 1])
        with col_r1:
            fig_risque = px.bar(
                score_ville.head(15).sort_values("score_risque"),
                x="score_risque", y="ville", color="niveau_risque",
                orientation="h",
                title="Top 15 des villes les plus à risque",
                labels={"score_risque": "Score de risque", "ville": "Ville"},
                color_discrete_map={"Faible":"#2ecc71","Modéré":"#f1c40f","Élevé":"#e67e22","Critique":"#e74c3c"},
            )
            fig_risque.update_layout(height=500)
            st.plotly_chart(fig_risque, width="stretch")
        with col_r2:
            st.dataframe(
                score_ville[["ville","region","pm25_moyen","pct_alerte","tendance_7j","score_risque","niveau_risque"]]
                .rename(columns={
                    "ville":"Ville","region":"Région","pm25_moyen":"PM2.5 moy","pct_alerte":"% alerte",
                    "tendance_7j":"Tendance 7j","score_risque":"Score","niveau_risque":"Niveau"
                }).head(15),
                width="stretch",
                height=500,
            )

# ------------------------------------------------------------
# ONGLET 2 : POLLUTION
# ------------------------------------------------------------
with tab2:
    st.subheader(target_wording["analysis_title"])

    col_a, col_b = st.columns(2)

    # Boxplot par région
    with col_a:
        ordre_reg = df_filtered.groupby("region")[active_target].median().sort_values(ascending=False).index.tolist()
        fig_box = px.box(
            df_filtered, x="region", y=active_target,
            color="region",
            category_orders={"region": ordre_reg},
            title=f"Distribution de {target_config['label']} par région",
            labels={active_target: target_config["label"], "region": "Région"},
        )
        fig_box.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_box, width="stretch")

    # Évolution mensuelle
    with col_b:
        mois_data = df_filtered.groupby("mois")[active_target].mean().reset_index()
        mois_data["mois_nom"] = mois_data["mois"].map(mois_noms)

        fig_mois = px.bar(
            mois_data, x="mois_nom", y=active_target,
            color=active_target,
            color_continuous_scale=["green","yellow","orange","red"],
            range_color=[0, 60],
            title=f"{target_config['label']} moyen par mois",
            labels={active_target: target_config["label"], "mois_nom": "Mois"},
        )
        fig_mois.update_layout(height=450)
        st.plotly_chart(fig_mois, width="stretch")

    # Évolution journalière
    st.markdown(f"### Évolution journalière de {target_config['label']}")
    journalier = df_filtered.groupby("date", as_index=False)[active_target].mean().sort_values("date")
    if len(journalier) > 0:
        journalier["moyenne_mobile_7j"] = journalier[active_target].rolling(7, min_periods=1).mean()

        fig_jour = go.Figure()
        fig_jour.add_trace(go.Scatter(
            x=journalier["date"], y=journalier[active_target],
            mode="lines", name="Journalier", line=dict(width=1.5)
        ))
        fig_jour.add_trace(go.Scatter(
            x=journalier["date"], y=journalier["moyenne_mobile_7j"],
            mode="lines", name="Moyenne mobile 7j", line=dict(width=3)
        ))
        fig_jour.add_hline(
            y=seuil_alerte, line_dash="dash",
            annotation_text=f"{target_config['threshold_label']} = {seuil_alerte:.1f}",
            annotation_position="top left"
        )
        fig_jour.update_layout(
            height=420, title=f"Évolution journalière de {target_config['label']}",
            xaxis_title="Date", yaxis_title=target_config["label"]
        )
        st.plotly_chart(fig_jour, width="stretch")

    # Évolution annuelle
    annuel = df_filtered.groupby(["annee","region"])[active_target].mean().reset_index()
    fig_annuel = px.line(
        annuel, x="annee", y=active_target, color="region",
        title=f"Évolution annuelle de {target_config['label']} par région",
        labels={active_target: target_config["label"], "annee": "Année", "region": "Région"},
        markers=True,
    )
    fig_annuel.update_layout(height=400)
    st.plotly_chart(fig_annuel, width="stretch")

    # Répartition des alertes
    col_c, col_d = st.columns(2)
    with col_c:
        if "niveau_alerte" in df_filtered.columns:
            niveaux = df_filtered["niveau_alerte"].value_counts().reset_index()
            niveaux.columns = ["Niveau", "Jours"]
            color_map = {"Bon": "#2ecc71", "Modéré": "#f1c40f", "Mauvais": "#e67e22", "Dangereux": "#e74c3c"}
            fig_pie = px.pie(
                niveaux, values="Jours", names="Niveau",
                color="Niveau", color_discrete_map=color_map,
                title="Répartition des niveaux d'alerte",
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, width="stretch")

    with col_d:
        alerte_region = df_filtered.groupby("region")["alerte_pollution"].mean().sort_values(ascending=False).reset_index()
        alerte_region["pct"] = (alerte_region["alerte_pollution"] * 100).round(1)
        fig_alerte = px.bar(
            alerte_region, x="region", y="pct",
            color="pct",
            color_continuous_scale=["green","red"],
            title="% de jours en alerte par région",
            labels={"pct": "% Jours alerte", "region": "Région"},
        )
        fig_alerte.update_layout(height=400)
        st.plotly_chart(fig_alerte, width="stretch")

# ------------------------------------------------------------
# ONGLET 3 : CLIMAT VS POLLUTION
# ------------------------------------------------------------
with tab3:
    st.subheader(target_wording["climate_title"])

    # Scatter température vs PM2.5
    col_e, col_f = st.columns(2)

    with col_e:
        if "temp_max" in df_filtered.columns:
            # Échantillonner pour la vitesse
            sample = df_filtered.sample(min(FAST_SAMPLE_SCATTER, len(df_filtered)), random_state=42)
            fig_temp = px.scatter(
                sample, x="temp_max", y=active_target,
                color="region", opacity=0.4,
                title=f"Température vs {target_config['label']}",
                labels={"temp_max": "Température max (°C)", active_target: target_config["label"]},
            )
            fig_temp.update_layout(height=450)
            st.plotly_chart(fig_temp, width="stretch")

    with col_f:
        if "vitesse_vent_max" in df_filtered.columns:
            sample = df_filtered.sample(min(FAST_SAMPLE_SCATTER, len(df_filtered)), random_state=42)
            fig_vent = px.scatter(
                sample, x="vitesse_vent_max", y=active_target,
                color="region", opacity=0.4,
                title=f"Vitesse du vent vs {target_config['label']}",
                labels={"vitesse_vent_max": "Vent max (km/h)", active_target: target_config["label"]},
            )
            fig_vent.update_layout(height=450)
            st.plotly_chart(fig_vent, width="stretch")

    # Heatmap mois x région
    st.markdown("### Carte de chaleur : Pollution par mois et par région")

    # DIRECTION DU VENT ET POLLUTION
    st.markdown("---")
    st.markdown(f"### 🧭 {target_wording['wind_title']}")
    st.markdown(target_wording["wind_explainer"])

    if "direction_vent" in df_filtered.columns:
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            # Rose des vents : fréquence des directions
            vent_data = df_filtered[["direction_vent", active_target]].dropna().copy()

            # Découper en 16 secteurs de 22.5° chacun
            bins_vent = np.arange(0, 360 + 22.5, 22.5)
            labels_vent = [
                "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO",
            ]
            vent_data["secteur"] = pd.cut(
                vent_data["direction_vent"] % 360,
                bins=bins_vent, labels=labels_vent, include_lowest=True,
            )

            # Compter la fréquence par secteur
            freq = vent_data.groupby("secteur", observed=True).size().reset_index(name="frequence")
            freq["secteur"] = pd.Categorical(freq["secteur"], categories=labels_vent, ordered=True)
            freq = freq.sort_values("secteur")

            fig_rose = px.bar_polar(
                freq, r="frequence", theta="secteur",
                color="frequence",
                color_continuous_scale=["#3498db", "#e74c3c"],
                title="Rose des vents (fréquence)",
            )
            fig_rose.update_layout(
                height=450,
                polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
            )
            st.plotly_chart(fig_rose, width="stretch")

        with col_v2:
            # Rose des vents : cible moyenne par direction
            pm_par_dir = vent_data.groupby("secteur", observed=True)[active_target].mean().reset_index()
            pm_par_dir["secteur"] = pd.Categorical(pm_par_dir["secteur"], categories=labels_vent, ordered=True)
            pm_par_dir = pm_par_dir.sort_values("secteur")

            fig_rose_pm = px.bar_polar(
                pm_par_dir, r=active_target, theta="secteur",
                color=active_target,
                color_continuous_scale=["green", "yellow", "orange", "red"],
                range_color=[20, 60],
                title=f"{target_config['label']} moyen par direction du vent",
            )
            fig_rose_pm.update_layout(
                height=450,
                polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
            )
            st.plotly_chart(fig_rose_pm, width="stretch")

        # Comparaison Harmattan vs Mousson
        st.markdown("### Comparaison des régimes de vent")
        vent_data_full = df_filtered[["direction_vent", active_target, "mois"]].dropna().copy()

        def classifier_vent(row):
            d = row["direction_vent"] % 360
            m = row["mois"]
            # Harmattan : vent de NE (315-90°) pendant la saison sèche (nov-mars)
            if m in [11, 12, 1, 2, 3] and (d >= 315 or d <= 90):
                return "Harmattan (NE, Nov-Mar)"
            # Mousson : vent de SO (135-270°) pendant la saison des pluies (mai-oct)
            elif m in [5, 6, 7, 8, 9, 10] and 135 <= d <= 270:
                return "Mousson (SO, Mai-Oct)"
            else:
                return "Autre"

        vent_data_full["regime"] = vent_data_full.apply(classifier_vent, axis=1)

        regime_stats = vent_data_full.groupby("regime")[active_target].agg(["mean", "count"]).reset_index()
        regime_stats.columns = ["Régime", "Valeur moyenne", "Nb jours"]

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            color_regime = {
                "Harmattan (NE, Nov-Mar)": "#e74c3c",
                "Mousson (SO, Mai-Oct)": "#2ecc71",
                "Autre": "#95a5a6",
            }
            fig_regime = px.bar(
                regime_stats, x="Régime", y="Valeur moyenne",
                color="Régime", color_discrete_map=color_regime,
                title=f"{target_config['label']} : Harmattan vs Mousson",
                text="Valeur moyenne",
            )
            fig_regime.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_regime.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_regime, width="stretch")

        with col_h2:
            fig_box_regime = px.box(
                vent_data_full, x="regime", y=active_target,
                color="regime", color_discrete_map=color_regime,
                title=f"Distribution de {target_config['label']} par régime de vent",
                labels={active_target: target_config["label"], "regime": ""},
            )
            fig_box_regime.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box_regime, width="stretch")

    st.markdown("---")
    st.markdown("### Carte de chaleur : Pollution par mois et par région")
    heat_data = df_filtered.groupby(["region","mois"])[active_target].mean().reset_index()
    heat_pivot = heat_data.pivot(index="region", columns="mois", values=active_target)
    heat_pivot.columns = [mois_noms.get(m, str(m)) for m in heat_pivot.columns]

    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale=["green","yellow","orange","red"],
        aspect="auto",
        title=f"{target_config['label']} moyen - Mois x Région",
        labels={"color": target_config["short_label"]},
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, width="stretch")

    # Corrélations
    st.markdown(f"### Corrélation des variables climatiques avec {target_config['label']}")
    corr_vars = [c for c in [
        "temp_max","temp_min","humidite_moy","vitesse_vent_max",
        "hauteur_couche_limite_moy","precipitation","rayonnement_solaire",
        "rayonnement_court_total","pression_mer_moy","couverture_nuageuse_moy",
    ] if c in df_filtered.columns]

    if corr_vars and active_target in df_filtered.columns:
        corr = df_filtered[corr_vars + [active_target]].corr()[active_target].drop(active_target).sort_values()
        fig_corr = px.bar(
            x=corr.values, y=corr.index,
            orientation="h",
            color=corr.values,
            color_continuous_scale=["blue","white","red"],
            range_color=[-0.8, 0.8],
            title=f"Corrélation avec {target_config['label']}",
            labels={"x": "Corrélation", "y": "Variable"},
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, width="stretch")

# ------------------------------------------------------------
# ONGLET 4 : MODÈLE IA
# ------------------------------------------------------------
with tab4:
    st.subheader(f"Performance du modèle IA - {target_config['label']}")
    st.caption(f"Mesures affichées pour la cible `{active_target}`")

    col_g, col_h, col_i, col_j = st.columns(4)
    col_g.metric(model_metrics_display[0][0], "N/A" if model_metrics_display[0][1] is None else f"{float(model_metrics_display[0][1]):.4f}")
    col_h.metric(model_metrics_display[1][0], "N/A" if model_metrics_display[1][1] is None else f"{float(model_metrics_display[1][1]):.4f}")
    col_i.metric(model_metrics_display[2][0], "N/A" if model_metrics_display[2][1] is None else f"{float(model_metrics_display[2][1]):.4f}")
    col_j.metric("Modèle jury", artefact_modele.get("model_name", "non connecté") if modele_jury_actif else "non connecté")

    st.markdown("---")

    # Importance des variables
    st.markdown("### Top 15 des variables les plus importantes")
    top_imp = importances.head(15).reset_index()
    top_imp.columns = ["Variable", "Importance"]

    fig_imp = px.bar(
        top_imp, x="Importance", y="Variable",
        orientation="h",
        color="Importance",
        color_continuous_scale="YlOrRd",
        title="Importance des variables - Modèle optimisé",
    )
    fig_imp.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, width="stretch")

    # Explication
    st.markdown("### Comment fonctionne le modèle ?")
    st.markdown(target_wording["model_explainer"])

# ------------------------------------------------------------
# ONGLET 5 : PRÉVISIONS
# ------------------------------------------------------------
with tab5:
    st.subheader(f"Prévision future - {target_config['label']}")
    st.caption(f"Prévision calculée sur la cible `{active_target}`")

    villes_prev = sorted(df_filtered["ville"].dropna().unique()) if "ville" in df_filtered.columns else []
    if not villes_prev:
        st.info("Aucune ville disponible pour la prévision avec le filtre actuel.")
    else:
        ville_prev = st.selectbox("Ville à prévoir", villes_prev, index=0)
        horizon = st.slider("Horizon de prévision (jours)", min_value=3, max_value=14, value=7)
        if modele_jury_actif:
            forecast_df = prevoir_proxy_par_modele(df, ville_prev, horizon=horizon, _artefact=artefact_modele)
        else:
            forecast_df = prevoir_cible_par_ville(df, ville_prev, horizon=horizon, target_col=active_target, target_config=target_config)
            st.warning("Artefact pipeline absent : retour à la projection simple de tendance.")

        hist = (
            df.loc[df["ville"] == ville_prev, ["date", active_target]]
            .dropna().groupby("date", as_index=False)[active_target].mean()
            .sort_values("date").tail(60)
        )

        if len(forecast_df) == 0:
            st.warning("Pas assez d'historique pour produire une prévision fiable.")
        else:
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Scatter(
                x=hist["date"], y=hist[active_target],
                mode="lines", name="Historique (60 derniers jours)", line=dict(width=2)
            ))
            fig_prev.add_trace(go.Scatter(
                x=forecast_df["date"], y=forecast_df["prediction"],
                mode="lines+markers", name="Prévision", line=dict(width=3, dash="dash")
            ))
            fig_prev.add_hline(
                y=seuil_alerte, line_dash="dot",
                annotation_text=f"{target_config['threshold_label']} = {seuil_alerte:.1f}"
            )
            fig_prev.update_layout(
                height=450,
                title=f"Prévision sur {horizon} jours - {ville_prev}",
                xaxis_title="Date", yaxis_title=target_config["label"]
            )
            st.plotly_chart(fig_prev, width="stretch")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("### Tableau des prévisions")
                st.dataframe(forecast_df, width="stretch", height=320)

            with col_p2:
                worst = forecast_df.sort_values("prediction", ascending=False).head(3).copy()
                st.markdown("### Jours à surveiller")
                for _, row in worst.iterrows():
                    st.metric(
                        label=str(pd.to_datetime(row["date"]).date()),
                        value=format_metric_value(row["prediction"], target_config),
                        delta=str(row["niveau"]) if "niveau" in row and pd.notna(row["niveau"]) else "Surveillance"
                    )

# ------------------------------------------------------------
# ONGLET 6 : API & MOBILE
# ------------------------------------------------------------
with tab6:
    st.subheader("API légère & usage mobile")
    st.caption(f"Payload généré pour la cible `{active_target}`")

    villes_api = sorted(df_filtered["ville"].dropna().unique()) if "ville" in df_filtered.columns else []
    ville_api = villes_api[0] if villes_api else None
    if ville_api and modele_jury_actif:
        forecast_api = prevoir_proxy_par_modele(df, ville_api, horizon=7, _artefact=artefact_modele)
    else:
        forecast_api = prevoir_cible_par_ville(df, ville_api, horizon=7, target_col=active_target, target_config=target_config) if ville_api else pd.DataFrame()
    payload = construire_payload_api_cible(df_filtered, forecast_api, seuil_alerte, active_target, target_config)

    st.markdown("### Payload JSON prêt à exposer en API")
    payload_json_bytes = payload_vers_json_bytes(payload)
    st.json(json.loads(payload_json_bytes.decode("utf-8")))

    st.download_button(
        "📥 Télécharger le JSON API",
        data=payload_json_bytes,
        file_name="air_quality_payload.json",
        mime="application/json",
    )

    st.markdown("### Mode mobile / PWA")
    st.info(
        "Le dashboard est déjà utilisable sur mobile via Streamlit. "
        "Pour une version app mobile, tu peux déployer sur Streamlit Cloud ou Render, "
        "puis ajouter un raccourci sur l'écran d'accueil du téléphone."
    )

    st.markdown("### Idée d'API REST minimale")
    st.code(
        'from fastapi import FastAPI\n'
        'app = FastAPI()\n\n'
        '@app.get("/forecast")\n'
        'def forecast():\n'
        '    return ' + json.dumps({"status": "ok", "message": "Expose le payload JSON du dashboard"}, ensure_ascii=False),
        language="python"
    )

# ------------------------------------------------------------
# ONGLET 7 : DONNÉES
# ------------------------------------------------------------
with tab7:
    st.subheader(target_wording["data_title"])

    cols_afficher = [c for c in [
        "date","ville","region","temp_max","temp_min","humidite_moy",
        "vitesse_vent_max","precipitation",active_target,"niveau_target",
    ] if c in df_filtered.columns]

    st.dataframe(
        df_filtered[cols_afficher].head(500),
        width="stretch",
        height=500,
    )

    st.markdown(f"**{len(df_filtered):,}** observations affichées (500 premières lignes)")

    # Téléchargement CSV
    @st.cache_data
    def convert_csv(dataframe):
        return dataframe[cols_afficher].to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Télécharger les données filtrées (CSV)",
        data=convert_csv(df_filtered),
        file_name="donnees_filtrees.csv",
        mime="text/csv",
    )

# ============================================================
# PIED DE PAGE
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🌍 Hackathon IndabaX Cameroon 2026 - L'IA au service de la résilience climatique et sanitaire"
    "</div>",
    unsafe_allow_html=True,
)


