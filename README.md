# 🌍 AirWatch Cameroun — Prédiction de la Qualité de l'Air

**Hackathon IndabaX Cameroon 2026** — L'IA au service de la résilience climatique et sanitaire

---

## 📋 Résumé

AirWatch Cameroun est une solution complète de veille intelligente de la qualité de l'air, développée dans le cadre du Hackathon IndabaX Cameroon 2026. En l'absence de capteurs PM2.5 au Cameroun, nous construisons un **proxy PM2.5** à partir de 6 variables météorologiques, puis nous le prédisons avec un moteur IA multi-modèles et le restituons via un dashboard interactif.

| Donnée | Valeur |
|--------|--------|
| Observations | 87 240 |
| Villes couvertes | 42 (10 régions) |
| Période | 2020 – 2025 |
| Meilleur modèle | LightGBM (R² > 0.80) |

---

## 🏗️ Architecture du Projet

```
hackathon-indabax-2026/
│
├── Pipe_hackaton_ameliore.py              # Pipeline ML (4 modèles)
├── dashboard_indabax_corrige_final_fixed.py  # Dashboard Streamlit (7 onglets)
├── rapport_technique.docx                 # Rapport technique complet
├── pitch_indabax_2026.pptx                # Pitch deck (7 slides)
├── requirements.txt                       # Dépendances Python
└── README.md
```

---

## 🔬 Proxy PM2.5 — Méthodologie

Le Cameroun ne dispose pas de réseau de capteurs de particules fines. Notre proxy est un indicateur synthétique basé sur 6 composantes météorologiques dont la corrélation avec les concentrations en PM2.5 est documentée dans la littérature scientifique :

| Composante | Poids | Justification |
|------------|-------|---------------|
| Stagnation du vent | 25% | Vent faible → polluants accumulés |
| Couche limite basse | 20% | Atmosphère bouchée → piégeage au sol |
| Absence de pluie | 15% | Pas de lessivage atmosphérique |
| Chaleur | 15% | Réactions photochimiques → ozone |
| Sécheresse | 15% | Air sec → poussières en suspension |
| Rayonnement solaire | 10% | UV → ozone troposphérique |

La normalisation est calculée **sur le jeu d'entraînement uniquement** (anti-leakage). Les composantes brutes sont **exclues** des features du modèle.

---

## 🤖 Pipeline de Prédiction

Le pipeline (`Pipe_hackaton_ameliore.py`) entraîne et compare 4 modèles :

| Modèle | Type | Avantage |
|--------|------|----------|
| Ridge | Linéaire | Rapide, baseline interprétable |
| LightGBM | Gradient Boosting | Très performant, catégorielles natives |
| XGBoost | Gradient Boosting | Robuste aux outliers |
| MLP | Réseau de neurones | Non-linéarités complexes |

**Validation temporelle** : 2020-2024 (train) vs 2025 (test) — aucune fuite temporelle.

Le meilleur modèle est automatiquement exporté en `.joblib` pour être utilisé par le dashboard.

### Exécution

```bash
python Pipe_hackaton_ameliore.py
```

Le pipeline génère le dossier `outputs_pipeline_hackathon/` contenant le modèle exporté, les métriques JSON et les prédictions.

---

## 📊 Dashboard Interactif

Le dashboard (`dashboard_indabax_corrige_final_fixed.py`) est une application Streamlit à 7 onglets :

| # | Onglet | Contenu |
|---|--------|---------|
| 1 | 🗺️ Carte | Carte Mapbox, score de risque par ville, alertes sanitaires par région |
| 2 | 📊 Pollution | Boxplot régional, évolution mensuelle/annuelle, niveaux d'alerte |
| 3 | 🌡️ Climat vs Pollution | Scatter plots, rose des vents, analyse Harmattan vs Mousson |
| 4 | 🤖 Modèle IA | Métriques R²/RMSE/MAE, importance des variables |
| 5 | 🔮 Prévisions | Prévision 3-14 jours par ville avec courbe historique |
| 6 | 📡 API & Mobile | Payload JSON exportable, exemple FastAPI, guide PWA |
| 7 | 📋 Données | Exploration brute filtrée, export CSV |

### Exécution

```bash
# 1. Exécuter le pipeline d'abord (recommandé)
python Pipe_hackaton_ameliore.py

# 2. Lancer le dashboard
streamlit run dashboard_indabax_corrige_final_fixed.py
```

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-pseudo>/hackathon-indabax-2026.git
cd hackathon-indabax-2026

# Installer les dépendances
pip install -r requirements.txt

# Placer le dataset Excel dans le même dossier
# Modifier FILEPATH dans les scripts si nécessaire
```

### Prérequis

- Python 3.10+
- Dataset `DATASET_FINAL_INDABAX2026.xlsx` (fourni par les organisateurs)

---

## 🎯 Impact Sociétal

| Utilisateur | Usage |
|-------------|-------|
| Ministère de la Santé | Anticiper les pics, protocoles d'urgence respiratoire |
| Collectivités locales | Alertes ciblées, communication préventive |
| Hôpitaux | Anticiper les afflux de patients respiratoires |
| Population | App mobile : niveau de pollution du jour + conseils |

La solution est **100% open-source**, déployable sans infrastructure coûteuse, et **réplicable** dans tout pays sans réseau de capteurs (Tchad, Niger, RCA...).

---

## 📁 Livrables

- ✅ **Pipeline ML** : script Python fonctionnel et documenté
- ✅ **Dashboard interactif** : Streamlit, 7 onglets, filtres dynamiques
- ✅ **Rapport technique** : méthodologie, résultats, impact
- ✅ **Pitch deck** : 7 slides
- ✅ **Vidéo de démonstration** : 3 minutes

---

## 🛠️ Stack Technologique

| Technologie | Usage |
|-------------|-------|
| Python 3.10+ | Langage principal |
| pandas / numpy | Manipulation de données |
| LightGBM / XGBoost | Modèles de prédiction |
| scikit-learn | Ridge, MLP, métriques |
| Streamlit | Dashboard interactif |
| Plotly | Visualisations (scatter, boxplot, mapbox) |
| openpyxl | Lecture du dataset Excel |

---

## 👥 Équipe

Hackathon IndabaX Cameroon 2026

---

*L'IA au service de la résilience climatique et sanitaire au Cameroun*
