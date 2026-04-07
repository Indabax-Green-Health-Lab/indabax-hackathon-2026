#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
PIPELINE HACKATHON INDABAX 2026 — VERSION 6 (4 MODÈLES RAPIDES)
==========================================================================

POURQUOI les versions précédentes bloquaient :
  → sklearn Pipeline + ColumnTransformer + cross_val_score
    crée des copies massives du DataFrame en mémoire à chaque pli.
    Sur 87K lignes × 100+ colonnes × 3 plis × 4 modèles = BOOM.

SOLUTION :
  → Tout le prétraitement est fait à la main avec pandas AVANT
  → Chaque modèle est entraîné UNE SEULE FOIS (pas de CV)
  → Évaluation directe sur le jeu de test

4 modèles :
  1. Ridge (linéaire)       — ~5 secondes
  2. LightGBM (boosting)    — ~15 secondes
  3. XGBoost (boosting)     — ~20 secondes
  4. MLP (réseau neurones)  — ~30 secondes

Temps total estimé : 1 à 2 minutes

Auteur : Équipe Hackathon
Date   : 2026
"""

# ============================================================
# 1. INSTALLATIONS
# ============================================================
import sys, subprocess, importlib

def ensure(pkg, name=None):
    try: importlib.import_module(name or pkg)
    except ImportError:
        print(f"📦 Installation de {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure("openpyxl")
ensure("lightgbm")
ensure("xgboost")
ensure("joblib")
ensure("gdown")

# ============================================================
# 2. IMPORTS
# ============================================================
import os, time, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import lightgbm as lgb
import xgboost as xgb
import gdown
from joblib import dump

pd.set_option("display.max_columns", 200)

# ============================================================
# 3. CONFIGURATION
# ============================================================
def telecharger_dataset():
    """Telecharge le dataset depuis Google Drive s'il n'est pas deja present."""
    url = "https://drive.google.com/uc?id=1rywzjwphUrDuml-8yOGBd0qWZHwR90h1&export=download"
    output = "DATASET_FINAL_INDABAX2026.xlsx"
    if not os.path.exists(output):
        print("Telechargement du dataset depuis Google Drive...")
        gdown.download(url, output, quiet=False)
    return output

FILEPATH = telecharger_dataset()
SHEET_NAME = 0
TARGET = "proxy_pm25"
TEST_YEARS = 1
RANDOM_STATE = 42
OUTPUT_DIR = "outputs_pipeline_hackathon"

t_debut = time.time()

def chrono(msg):
    elapsed = time.time() - t_debut
    print(f"  [{elapsed:6.1f}s] {msg}")

def detect_task_type(series):
    """Détecte automatiquement si la cible relève de la régression ou de la classification."""
    s = pd.Series(series).dropna()
    uniques = set(pd.unique(s))
    if uniques and uniques.issubset({0, 1, 0.0, 1.0, False, True}):
        return "classification"
    return "regression"

def get_target_config(target, task_type):
    configs = {
        "proxy_pm25": {
            "label": "Proxy PM2.5",
            "short_label": "PM2.5",
            "description": "niveau estimé de pollution",
            "threshold_label": "Seuil d'alerte",
        },
        "score_risque": {
            "label": "Score de risque",
            "short_label": "Risque",
            "description": "niveau global de risque environnemental",
            "threshold_label": "Seuil opérationnel",
        },
        "indice_stagnation": {
            "label": "Indice de stagnation",
            "short_label": "Stagnation",
            "description": "niveau de stagnation atmosphérique",
            "threshold_label": "Seuil de vigilance",
        },
        "alerte_pollution": {
            "label": "Alerte pollution",
            "short_label": "Alerte",
            "description": "déclenchement d'une alerte pollution",
            "threshold_label": "Seuil de décision",
        },
    }
    cfg = configs.get(target, {
        "label": target,
        "short_label": target,
        "description": f"valeur cible '{target}'",
        "threshold_label": "Seuil",
    }).copy()
    cfg["task_type"] = task_type
    return cfg

def verifier_cible_equivalente(df, target):
    """Signale les cibles strictement équivalentes pour éviter une fausse impression de changement."""
    groupes = [
        {"proxy_pm25", "score_risque"},
    ]
    for groupe in groupes:
        if target not in groupe:
            continue
        autres = [c for c in groupe if c != target and c in df.columns]
        for autre in autres:
            if df[target].equals(df[autre]):
                print(f"\n  ⚠️  La cible '{target}' est strictement identique à '{autre}'.")
                print("     Changer entre ces deux colonnes ne modifiera pas les résultats du modèle.")
                return autre
    return None

def resume_metriques(task_type, y_true, y_pred, y_score=None):
    """Calcule les métriques adaptées au type de tâche."""
    if task_type == "classification":
        scores = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
        }
        if y_score is not None and len(np.unique(y_true)) > 1:
            scores["ROC-AUC"] = roc_auc_score(y_true, y_score)
        return scores

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }



def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def to_serialisable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def sauvegarder_json(filepath, payload):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=to_serialisable)

def sauvegarder_modele(output_dir, model_name, model, scaler, feature_cols, task_type, target, preprocessing=None):
    artefact = {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "task_type": task_type,
        "target": target,
        "preprocessing": preprocessing or {},
    }
    filepath = os.path.join(output_dir, "meilleur_modele.joblib")
    dump(artefact, filepath)
    return filepath

def construire_dataframe_resultats(resultats, task_type, metric_cle):
    lignes = []
    for nom, metrics, duree in resultats:
        row = {"modele": nom, "temps_secondes": round(float(duree), 3)}
        row.update({k: float(v) for k, v in metrics.items()})
        lignes.append(row)
    df_res = pd.DataFrame(lignes)
    return df_res.sort_values(metric_cle, ascending=False).reset_index(drop=True)

def evaluer_sous_groupes(df_test, task_type, target, pred_col, score_col=None, group_cols=None):
    if group_cols is None:
        group_cols = []
    sorties = {}
    for col in group_cols:
        if col not in df_test.columns:
            continue
        rows = []
        for valeur, g in df_test.groupby(col, dropna=False):
            if len(g) < 5:
                continue
            if task_type == "classification":
                metrics = resume_metriques(
                    task_type,
                    g[target].values,
                    g[pred_col].values,
                    g[score_col].values if score_col and score_col in g.columns else None,
                )
            else:
                metrics = resume_metriques(task_type, g[target].values, g[pred_col].values)
            row = {"groupe": valeur, "n": int(len(g))}
            row.update({k: float(v) for k, v in metrics.items()})
            rows.append(row)
        if rows:
            sorties[col] = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return sorties

print("=" * 60)
print("PIPELINE HACKATHON — 4 MODÈLES RAPIDES")
print("=" * 60)

OUTPUT_DIR = ensure_output_dir(OUTPUT_DIR)

# ============================================================
# 4. TRADUCTION DES COLONNES
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

# Colonnes inutiles à supprimer
DROP_COLS = [
    "lat_doublon","lon_doublon",
    "temp_2m_moy","temp_2m_min","temp_2m_max",
    "temp_apparente_moy","temp_apparente_min","temp_apparente_max",
    "chute_neige","chute_neige_totale",
    "precipitation_totale","pluie","pluie_totale",
    "identifiant","vitesse_vent_10m_max",
]

# Les 7 variables brutes du proxy PM2.5
# Exclues des features pour éviter R² = 1.0
COMPOSANTES_BRUTES = {
    "vitesse_vent_max","hauteur_couche_limite_moy",
    "precipitation","temp_max","humidite_moy",
    "rayonnement_solaire","rayonnement_court_total",
}

# ============================================================
# 5. CHARGEMENT
# ============================================================
chrono("Chargement du fichier...")

if not os.path.exists(FILEPATH):
    print(f"\n❌ FICHIER INTROUVABLE : {FILEPATH}")
    print("Vérifiez le chemin et relancez.")
    sys.exit(1)

try:
    df = pd.read_excel(FILEPATH, sheet_name=SHEET_NAME, header=0, engine="openpyxl")
except Exception as e:
    print(f"\n❌ ERREUR DE LECTURE : {e}")
    print("Essayez de sauvegarder le fichier en .csv depuis Excel, puis changez FILEPATH.")
    sys.exit(1)

# Nettoyage des noms de colonnes
df.columns = df.columns.astype(str).str.strip().str.replace("\xa0"," ",regex=False)

# Traduction en français
df = df.rename(columns=RENAME)

# Suppression des doublons et colonnes inutiles
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

# Conversion date
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

chrono(f"✅ {df.shape[0]:,} lignes × {df.shape[1]} colonnes chargées")

# Audit rapide
if "ville" in df.columns:
    print(f"       {df['ville'].nunique()} villes")
if "annee" in df.columns:
    print(f"       Années : {int(df['annee'].min())} → {int(df['annee'].max())}")

# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================
chrono("Feature engineering...")

# Temporel
if "date" in df.columns:
    if "annee" not in df.columns: df["annee"] = df["date"].dt.year
    if "mois" not in df.columns:  df["mois"] = df["date"].dt.month
    else: df["mois"] = pd.to_numeric(df["mois"], errors="coerce")
    df["jour_annee"] = df["date"].dt.day_of_year

# Durées secondes → heures
for col in ["duree_jour", "duree_ensoleillement"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].dropna().median() > 100:
            df[col] = df[col] / 3600.0

# Tri par ville + date
sort_cols = [c for c in ["ville", "date"] if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols).reset_index(drop=True)

# Cycliques mois
if "mois" in df.columns:
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)

# Cycliques vent
if "direction_vent" in df.columns:
    df["vent_sin"] = np.sin(np.deg2rad(df["direction_vent"]))
    df["vent_cos"] = np.cos(np.deg2rad(df["direction_vent"]))

# Variables dérivées
if {"temp_moy", "point_rosee_moy"}.issubset(df.columns):
    df["ecart_point_rosee"] = df["temp_moy"] - df["point_rosee_moy"]

if {"rafales_max", "vitesse_vent_max"}.issubset(df.columns):
    df["ratio_rafales"] = df["rafales_max"] / (df["vitesse_vent_max"] + 0.1)

if "pression_mer_moy" in df.columns:
    df["pression_anomalie"] = df["pression_mer_moy"] - 1013.25

if {"temp_max", "ressenti_max"}.issubset(df.columns):
    df["ecart_ressenti"] = df["temp_max"] - df["ressenti_max"]

if {"duree_ensoleillement", "duree_jour"}.issubset(df.columns):
    df["ratio_ensoleillement"] = df["duree_ensoleillement"] / (df["duree_jour"] + 0.01)

# Lags par ville (J-1 et J-7 seulement)
group = "ville" if "ville" in df.columns else None
lag_vars = [c for c in [
    "temp_max", "precipitation", "vitesse_vent_max",
    "humidite_moy", "hauteur_couche_limite_moy",
] if c in df.columns]

if group and lag_vars:
    chrono("Lags J-1 et J-7...")
    for lag in [1, 7]:
        for col in lag_vars:
            df[f"{col}_lag{lag}"] = df.groupby(group)[col].shift(lag)

    chrono("Moyennes glissantes 7j...")
    for col in lag_vars:
        df[f"{col}_moy7j"] = (
            df.groupby(group)[col]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        )

chrono(f"→ {df.shape[1]} colonnes")

# ============================================================
# 7. PROXY PM2.5
# ============================================================
chrono("Construction du proxy PM2.5...")

annee_max = int(df["annee"].max())
cutoff = annee_max - TEST_YEARS
mask_train = df["annee"] <= cutoff

def norm01(series):
    """Normalise entre 0 et 1 avec bornes du TRAIN uniquement."""
    s = pd.to_numeric(series, errors="coerce")
    vmin = s[mask_train].min()
    vmax = s[mask_train].max()
    d = vmax - vmin
    if d == 0 or pd.isna(d):
        return pd.Series(0.5, index=s.index)
    return ((s - vmin) / d).clip(0, 1)

# 6 composantes pondérées
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

# Alerte pollution
seuil = df.loc[mask_train, "proxy_pm25"].quantile(0.80)
df["alerte_pollution"] = (df["proxy_pm25"] >= seuil).astype(int)
print(f"       Seuil alerte Q80 : {seuil:.1f}")

# Sous-indices
if {"vitesse_vent_max","hauteur_couche_limite_moy"}.issubset(df.columns):
    df["indice_stagnation"] = (
        0.6*(1.0-norm01(df["vitesse_vent_max"])) +
        0.4*(1.0-norm01(df["hauteur_couche_limite_moy"]))
    ) * 100.0

df["score_risque"] = proxy * 100.0

chrono(f"Proxy PM2.5 → moy={df['proxy_pm25'].mean():.1f}, "
       f"min={df['proxy_pm25'].min():.1f}, max={df['proxy_pm25'].max():.1f}")

# ============================================================
# 8. SÉLECTION DES FEATURES
# ============================================================
chrono("Sélection des features...")

# Supprimer lignes sans cible
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
mask_train = df["annee"] <= cutoff
mask_test = df["annee"] > cutoff
task_type = detect_task_type(df[TARGET])
target_equivalente = verifier_cible_equivalente(df, TARGET)
target_config = get_target_config(TARGET, task_type)

print(f"       Cible : {target_config['label']} ({TARGET})")
print(f"       Type   : {'classification binaire' if task_type == 'classification' else 'regression'}")
print(f"       But    : predire {target_config['description']}")
if target_equivalente is not None:
    print(f"       Référence identique détectée : {target_equivalente}")

# Colonnes exclues
toutes_cibles = {"proxy_pm25","alerte_pollution","indice_stagnation","score_risque"}
exclure = toutes_cibles | COMPOSANTES_BRUTES | {
    "date","identifiant","lever_soleil","coucher_soleil",
    "amplitude_thermique","pluie_log","dispersion_atmo","ville",
}

# Colonnes catégorielles (encodées en entiers pour LightGBM/XGBoost)
cat_cols = [c for c in ["region","zone_climatique","saison","categorie_meteo"] if c in df.columns]

# Colonnes numériques
num_cols = [
    c for c in df.columns
    if c not in exclure and c not in cat_cols and c != TARGET
    and pd.api.types.is_numeric_dtype(df[c])
]

# Sécurité anti-fuite
num_cols = [c for c in num_cols if c not in COMPOSANTES_BRUTES]

feature_cols = num_cols + cat_cols
print(f"       {len(num_cols)} numériques + {len(cat_cols)} catégorielles = {len(feature_cols)} features")

# ============================================================
# 9. PRÉPARATION TRAIN / TEST (SANS SKLEARN PIPELINE)
# ============================================================
chrono("Séparation train/test...")

X_train = df.loc[mask_train, feature_cols].copy()
X_test  = df.loc[mask_test,  feature_cols].copy()
y_train = df.loc[mask_train, TARGET].values
y_test  = df.loc[mask_test,  TARGET].values
meta_cols_test = [c for c in ["date", "ville", "region", "saison", "annee"] if c in df.columns]
df_test_meta = df.loc[mask_test, meta_cols_test].reset_index(drop=True).copy()

if task_type == "classification":
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

print(f"       Train : {len(X_train):,} | Test : {len(X_test):,}")

# Encoder les catégorielles en entiers
cat_mappings = {}
for col in cat_cols:
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype("category")
    cat_mappings[col] = {str(cat): int(idx) for idx, cat in enumerate(combined.cat.categories.tolist())}
    codes = combined.cat.codes
    X_train[col] = codes.iloc[:len(X_train)].values
    X_test[col]  = codes.iloc[len(X_train):].values

# Remplir les NaN numériques par la médiane du train
chrono("Imputation des NaN...")
medianes = {}
for col in num_cols:
    med = X_train[col].median()
    medianes[col] = med
    X_train[col] = X_train[col].fillna(med)
    X_test[col]  = X_test[col].fillna(med)

# Convertir en numpy pour la vitesse
X_train_np = X_train.values.astype(np.float32)
X_test_np  = X_test.values.astype(np.float32)
y_train_np = y_train.astype(np.float32 if task_type == "regression" else np.int32)

# Libérer les DataFrames pandas (on garde seulement les numpy)
del X_train, X_test
import gc; gc.collect()

# Version normalisée pour Ridge et MLP (qui en ont besoin)
chrono("Normalisation pour Ridge/MLP...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled  = scaler.transform(X_test_np)

chrono("Données prêtes !")

# ============================================================
# 10. ENTRAÎNEMENT DES 4 MODÈLES
# ============================================================
print("\n" + "=" * 60)
print("ENTRAÎNEMENT DES 4 MODÈLES")
print("=" * 60)

resultats = []

# Dictionnaires pour stocker modèles et prédictions
all_models = {}
all_preds = {}
all_scores = {}
metric_cle = "F1" if task_type == "classification" else "R2"

# ─── Modèle 1 : linéaire ───
try:
    chrono("1/4 Linéaire...")
    t0 = time.time()
    if task_type == "classification":
        model_ridge = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        nom_modele = "LogisticRegression"
    else:
        model_ridge = Ridge(alpha=1.0)
        nom_modele = "Ridge"
    model_ridge.fit(X_train_scaled, y_train_np)
    pred_ridge = model_ridge.predict(X_test_scaled)
    score_ridge = model_ridge.predict_proba(X_test_scaled)[:, 1] if task_type == "classification" else None
    d = time.time() - t0
    metrics = resume_metriques(task_type, y_test, pred_ridge, score_ridge)
    resultats.append((nom_modele, metrics, d))
    all_models[nom_modele] = model_ridge
    all_preds[nom_modele] = pred_ridge
    all_scores[nom_modele] = score_ridge
    if task_type == "classification":
        print(f"       Accuracy={metrics['Accuracy']:.4f} | F1={metrics['F1']:.4f} | Recall={metrics['Recall']:.4f} | {d:.1f}s ✅")
    else:
        print(f"       R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.4f} | {d:.1f}s ✅")
except Exception as e:
    print(f"       ❌ Modèle linéaire a échoué : {e}")

# ─── Modèle 2 : LightGBM ───
try:
    chrono("2/4 LightGBM...")
    t0 = time.time()
    if task_type == "classification":
        model_lgb = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
        )
    else:
        model_lgb = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
        )
    model_lgb.fit(X_train_np, y_train_np)
    pred_lgb = model_lgb.predict(X_test_np)
    score_lgb = model_lgb.predict_proba(X_test_np)[:, 1] if task_type == "classification" else None
    d = time.time() - t0
    metrics = resume_metriques(task_type, y_test, pred_lgb, score_lgb)
    resultats.append(("LightGBM", metrics, d))
    all_models["LightGBM"] = model_lgb
    all_preds["LightGBM"] = pred_lgb
    all_scores["LightGBM"] = score_lgb
    if task_type == "classification":
        print(f"       Accuracy={metrics['Accuracy']:.4f} | F1={metrics['F1']:.4f} | Recall={metrics['Recall']:.4f} | {d:.1f}s ✅")
    else:
        print(f"       R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.4f} | {d:.1f}s ✅")
except Exception as e:
    print(f"       ❌ LightGBM a échoué : {e}")

# ─── Modèle 3 : XGBoost ───
try:
    chrono("3/4 XGBoost...")
    t0 = time.time()
    if task_type == "classification":
        model_xgb = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=1, random_state=RANDOM_STATE, verbosity=0,
            tree_method="hist", eval_metric="logloss",
        )
    else:
        model_xgb = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=1, random_state=RANDOM_STATE, verbosity=0,
            tree_method="hist",
        )
    model_xgb.fit(X_train_np, y_train_np)
    pred_xgb = model_xgb.predict(X_test_np)
    score_xgb = model_xgb.predict_proba(X_test_np)[:, 1] if task_type == "classification" else None
    d = time.time() - t0
    metrics = resume_metriques(task_type, y_test, pred_xgb, score_xgb)
    resultats.append(("XGBoost", metrics, d))
    all_models["XGBoost"] = model_xgb
    all_preds["XGBoost"] = pred_xgb
    all_scores["XGBoost"] = score_xgb
    if task_type == "classification":
        print(f"       Accuracy={metrics['Accuracy']:.4f} | F1={metrics['F1']:.4f} | Recall={metrics['Recall']:.4f} | {d:.1f}s ✅")
    else:
        print(f"       R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.4f} | {d:.1f}s ✅")
except Exception as e:
    print(f"       ❌ XGBoost a échoué : {e}")

# ─── Modèle 4 : MLP (réseau de neurones) ───
try:
    chrono("4/4 MLP...")
    t0 = time.time()
    if task_type == "classification":
        model_mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
    else:
        model_mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
    model_mlp.fit(X_train_scaled, y_train_np)
    pred_mlp = model_mlp.predict(X_test_scaled)
    score_mlp = model_mlp.predict_proba(X_test_scaled)[:, 1] if task_type == "classification" else None
    d = time.time() - t0
    metrics = resume_metriques(task_type, y_test, pred_mlp, score_mlp)
    resultats.append(("MLP", metrics, d))
    all_models["MLP"] = model_mlp
    all_preds["MLP"] = pred_mlp
    all_scores["MLP"] = score_mlp
    if task_type == "classification":
        print(f"       Accuracy={metrics['Accuracy']:.4f} | F1={metrics['F1']:.4f} | Recall={metrics['Recall']:.4f} | {d:.1f}s ✅")
    else:
        print(f"       R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.4f} | {d:.1f}s ✅")
except Exception as e:
    print(f"       ❌ MLP a échoué : {e}")

# Vérification qu'au moins un modèle a fonctionné
if len(resultats) == 0:
    print("\n❌ AUCUN MODÈLE N'A FONCTIONNÉ.")
    sys.exit(1)

# ============================================================
# 11. TABLEAU COMPARATIF
# ============================================================
print("\n" + "=" * 60)
print("COMPARAISON DES 4 MODÈLES")
print("=" * 60)
if task_type == "classification":
    print(f"  {'Modèle':<20s} {'Acc.':>8s} {'F1':>8s} {'Recall':>8s} {'Temps':>8s}")
    print("  " + "─" * 58)
else:
    print(f"  {'Modèle':<20s} {'R²':>8s} {'RMSE':>8s} {'MAE':>8s} {'Temps':>8s}")
    print("  " + "─" * 58)

for nom, metrics, dur in sorted(resultats, key=lambda x: x[1][metric_cle], reverse=True):
    marqueur = " ← MEILLEUR" if metrics[metric_cle] == max(r[1][metric_cle] for r in resultats) else ""
    if task_type == "classification":
        print(f"  {nom:<20s} {metrics['Accuracy']:>8.4f} {metrics['F1']:>8.4f} {metrics['Recall']:>8.4f} {dur:>6.1f}s{marqueur}")
    else:
        print(f"  {nom:<20s} {metrics['R2']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} {dur:>6.1f}s{marqueur}")

# Identifier le meilleur
best_name = max(resultats, key=lambda x: x[1][metric_cle])[0]
best_model = all_models[best_name]
best_pred = all_preds[best_name]
best_metrics = next(metrics for nom, metrics, _ in resultats if nom == best_name)
best_score = max(r[1][metric_cle] for r in resultats)

print(f"\n  [MEILLEUR] Modele pour {target_config['short_label']} : {best_name} ({metric_cle} = {best_score:.4f})")

if task_type == "regression":
    if best_score > 0.99:
        print("  ⚠️  R² > 0.99 → Possible fuite de données")
    elif best_score > 0.80:
        print(f"  [OK] Tres bon modele pour predire {target_config['description']}")
    elif best_score > 0.65:
        print(f"  [OK] Bon modele pour predire {target_config['description']}")
else:
    if best_score > 0.95:
        print("  ⚠️  Score très élevé → vérifiez qu'il n'y a pas de cible trop proche des features")
    elif best_score > 0.80:
        print(f"  [OK] Tres bon modele pour classer {target_config['description']}")

# ============================================================
# 11B. EXPORTS DES RÉSULTATS
# ============================================================
chrono("Sauvegarde des artefacts et des métriques...")

df_resultats = construire_dataframe_resultats(resultats, task_type, metric_cle)
filepath_resultats = os.path.join(OUTPUT_DIR, "comparaison_modeles.csv")
df_resultats.to_csv(filepath_resultats, index=False, encoding="utf-8-sig")

score_col = "score_prediction" if task_type == "classification" else None
df_predictions_test = df_test_meta.copy()
df_predictions_test[TARGET] = y_test
df_predictions_test["prediction"] = best_pred
if task_type == "classification" and all_scores.get(best_name) is not None:
    df_predictions_test[score_col] = all_scores[best_name]

filepath_predictions = os.path.join(OUTPUT_DIR, "predictions_test.csv")
df_predictions_test.to_csv(filepath_predictions, index=False, encoding="utf-8-sig")

evaluations_groupes = evaluer_sous_groupes(
    df_predictions_test,
    task_type=task_type,
    target=TARGET,
    pred_col="prediction",
    score_col=score_col,
    group_cols=["region", "ville"],
)

fichiers_groupes = {}
for nom_groupe, df_groupe in evaluations_groupes.items():
    filepath_groupe = os.path.join(OUTPUT_DIR, f"resultats_par_{nom_groupe}.csv")
    df_groupe.to_csv(filepath_groupe, index=False, encoding="utf-8-sig")
    fichiers_groupes[nom_groupe] = filepath_groupe

preprocessing = {
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "medianes_numeriques": medianes,
    "cat_mappings": cat_mappings,
    "scaler_required": best_name in {"LogisticRegression", "Ridge", "MLP"},
    "threshold_alert": float(seuil),
    "test_years": int(TEST_YEARS),
}
filepath_modele = sauvegarder_modele(
    OUTPUT_DIR,
    best_name,
    best_model,
    scaler if preprocessing["scaler_required"] else None,
    feature_cols,
    task_type,
    TARGET,
    preprocessing=preprocessing,
)

resume_export = {
    "target": TARGET,
    "task_type": task_type,
    "metric_cle": metric_cle,
    "train_rows": int(len(y_train)),
    "test_rows": int(len(y_test)),
    "n_features": int(len(feature_cols)),
    "meilleur_modele": {
        "nom": best_name,
        "score": float(best_score),
        "metriques": best_metrics,
    },
    "resultats_modeles": df_resultats.to_dict(orient="records"),
    "fichiers_generes": {
        "modele": filepath_modele,
        "comparaison_modeles": filepath_resultats,
        "predictions_test": filepath_predictions,
        "resultats_groupes": fichiers_groupes,
    },
}

filepath_resume = os.path.join(OUTPUT_DIR, "resume_metriques.json")
sauvegarder_json(filepath_resume, resume_export)

print(f"  💾 Modèle sauvegardé : {os.path.basename(filepath_modele)}")
print(f"  💾 Métriques JSON : {os.path.basename(filepath_resume)}")
print(f"  💾 Résultats test : {os.path.basename(filepath_predictions)}")
if fichiers_groupes:
    print("  💾 Résultats par groupe : " + ", ".join(os.path.basename(p) for p in fichiers_groupes.values()))
else:
    print("  ℹ️ Aucun export par région/ville : colonnes absentes du jeu de test")

# ============================================================
# 12. IMPORTANCE DES VARIABLES (MEILLEUR MODÈLE)
# ============================================================
print("\n" + "─" * 60)
print(f"TOP 20 VARIABLES - {best_name} ({target_config['short_label']})")
print("─" * 60)

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
elif hasattr(best_model, "coef_"):
    imp = pd.Series(np.abs(best_model.coef_.ravel()), index=feature_cols).sort_values(ascending=False)
else:
    imp = None

if imp is not None:
    for i, (feat, val) in enumerate(imp.head(20).items(), 1):
        bar = "█" * int(val / imp.max() * 30)
        print(f"  {i:>2}. {feat:<45s} {val:>8.1f}  {bar}")

    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        top = imp.head(20).sort_values()
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))
        top.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Top 20 — Importance ({best_name})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig("importance_variables.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  📊 Sauvegardé : importance_variables.png")
    except:
        pass

# ============================================================
# 13. GRAPHIQUES
# ============================================================

# Prédit vs Réel
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if task_type == "classification":
        classes = np.array([0, 1])
        vrais = np.array([(y_test == c).sum() for c in classes])
        preds = np.array([(best_pred == c).sum() for c in classes])
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, vrais, width=width, label="Réel", color="steelblue")
        ax.bar(x + width/2, preds, width=width, label="Prédit", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_xlabel("Classe")
        ax.set_ylabel("Effectif")
        ax.set_title(f"Réel vs Prédit — {best_name}")
        ax.legend()
    else:
        ax.scatter(y_test, best_pred, alpha=0.3, s=8, color="steelblue")
        lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Parfait")
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Prédictions")
        ax.set_title(f"Prédit vs Réel — {best_name}")
        ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    if task_type == "classification":
        erreurs = (best_pred != y_test).astype(int)
        ax.hist(erreurs, bins=[-0.5, 0.5, 1.5], color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Erreur"])
        ax.set_xlabel("Issue")
        ax.set_ylabel("Fréquence")
        ax.set_title("Répartition des prédictions")
    else:
        residus = y_test - best_pred
        ax.hist(residus, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Résidu")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution des résidus")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("predictions_vs_reel.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Sauvegardé : predictions_vs_reel.png")
except:
    pass

# Comparaison des 4 modèles en barres
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    noms = [r[0] for r in resultats]
    scores = [r[1][metric_cle] for r in resultats]
    colors = ["#2ecc71" if r == max(scores) else "#3498db" for r in scores]
    bars = ax.bar(noms, scores, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel(metric_cle)
    ax.set_title("Comparaison des 4 modèles", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.15 if max(scores) > 0 else 1)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("comparaison_modeles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Sauvegardé : comparaison_modeles.png")
except:
    pass

# Proxy par région
try:
    if "region" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ordre = df.groupby("region")["proxy_pm25"].median().sort_values(ascending=False).index
        data_plot = [df.loc[df["region"]==r, "proxy_pm25"].dropna().values for r in ordre]
        bp = ax.boxplot(data_plot, labels=ordre, showfliers=False, patch_artist=True)
        for p in bp["boxes"]:
            p.set_facecolor("steelblue")
            p.set_alpha(0.7)
        ax.set_xticklabels(ordre, rotation=45, ha="right")
        ax.set_title(f"{target_config['label']} par region", fontsize=14, fontweight="bold")
        ax.set_ylabel(target_config["label"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{TARGET}_par_region.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [FIGURE] Sauvegarde : {TARGET}_par_region.png")
except:
    pass

# Matrice de confusion et courbe ROC (classification uniquement)
if task_type == "classification":
    try:
        cm = confusion_matrix(y_test, best_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1])
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(f"Matrice de confusion — {best_name}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "matrice_confusion.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print("  📊 Sauvegardé : matrice_confusion.png")
    except Exception as e:
        print(f"  ⚠️ Matrice de confusion non générée : {e}")

    try:
        if best_name in all_scores and all_scores[best_name] is not None and len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, all_scores[best_name])
            auc_value = roc_auc_score(y_test, all_scores[best_name])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.3f}")
            ax.plot([0, 1], [0, 1], "r--", lw=1)
            ax.set_xlabel("Taux de faux positifs")
            ax.set_ylabel("Taux de vrais positifs")
            ax.set_title(f"Courbe ROC — {best_name}")
            ax.legend(loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "courbe_roc.png"), dpi=150, bbox_inches="tight")
            plt.show()
            print("  📊 Sauvegardé : courbe_roc.png")
    except Exception as e:
        print(f"  ⚠️ Courbe ROC non générée : {e}")

# ============================================================
# 14. STATISTIQUES FINALES
# ============================================================
print("\n" + "─" * 60)
print(f"STATISTIQUES - {target_config['label']} ({TARGET})")
print("─" * 60)
if task_type == "classification":
    print(df[TARGET].value_counts(dropna=False).sort_index().to_string())
    print("\nMétriques du meilleur modèle :")
    for nom_metric, valeur in best_metrics.items():
        print(f"  {nom_metric:<10s}: {valeur:.4f}")
else:
    print(df[TARGET].describe().to_string())

duree = time.time() - t_debut
print("\n" + "=" * 60)
print(f"✅ TERMINÉ EN {duree:.0f} SECONDES ({duree/60:.1f} min)")
print("=" * 60)
