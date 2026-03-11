# 🌍 IndabaX Cameroon Hackathon 2026
## L'IA au service de la résilience climatique et sanitaire au Cameroun

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-En%20développement-yellow)
![Team](https://img.shields.io/badge/Équipe-Green%20Health%20Lab-green)

---

## 📌 Description du projet

Ce projet est développé dans le cadre du **IndabaX Cameroon Hackathon 2026**.

Le Cameroun fait face à une dégradation croissante de la qualité de l'air, amplifiée
par la variabilité climatique (vagues de chaleur, vents stagnants, tempêtes de poussière).
Notre solution utilise l'IA pour **prédire les indicateurs de pollution atmosphérique**
et fournir des **outils d'aide à la décision** accessibles aux décideurs et aux communautés.

---

## 🎯 Objectifs

- 🔮 **Prédire** les indicateurs de pollution (proxy PM2.5) à partir de données météorologiques
- 🗺️ **Identifier** les facteurs climatiques aggravants par région
- 📊 **Visualiser** les résultats via un dashboard interactif (Streamlit)

---

## 📦 Dataset

- **Couverture** : 42 villes, 10 régions du Cameroun
- **Période** : Données journalières — Janvier 2020 à Décembre 2025
- **Volume** : 87 240 observations
- **Variables** :
  - Météo : température (min/max/moy), vitesse/direction du vent, précipitations, humidité
  - Solaire : durée d'ensoleillement, rayonnement solaire
  - Géographique : ville, région, latitude, longitude

> ⚠️ Les données brutes ne doivent PAS être committées sur GitHub. Voir `.gitignore`.

---

## 🗂️ Structure du projet

```
indabax-hackathon-2026/
│
├── data/
│   ├── raw/                    # Données brutes (ne pas modifier)
│   └── processed/              # Données nettoyées
│
├── notebooks/
│   ├── 01_exploration.ipynb    # Analyse exploratoire (EDA)
│   ├── 02_preprocessing.ipynb  # Nettoyage des données
│   └── 03_modeling.ipynb       # Entraînement du modèle
│
├── src/
│   ├── preprocessing.py        # Pipeline de nettoyage
│   ├── features.py             # Feature engineering
│   ├── model.py                # Modèle ML (entraînement + évaluation)
│   └── dashboard.py            # Application Streamlit
│
├── models/
│   └── best_model.pkl          # Modèle sauvegardé
│
├── reports/
│   └── figures/                # Graphiques exportés
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Indabax-Green-Health-Lab/indabax-hackathon-2026.git
cd indabax-hackathon-2026

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancer le dashboard

```bash
streamlit run src/dashboard.py
```

---

## 📊 Livrables attendus

| # | Livrable | Statut |
|---|----------|--------|
| 1 | Modèle prédictif (notebook Python) | 🔄 En cours |
| 2 | Dashboard interactif (Streamlit) | 🔄 En cours |
| 3 | Pitch deck (5–7 slides) | 🔄 En cours |
| 4 | Vidéo démo (max 3 min) | ⏳ À faire |

---

## 🏆 Critères d'évaluation

| Critère | Points |
|---------|--------|
| Performance technique (modèle, code, innovation) | 35 pts |
| UX & Design du dashboard | 20 pts |
| Pertinence sociétale & impact local | 25 pts |
| Communication (pitch, storytelling) | 15 pts |
| Documentation & reproductibilité | 5 pts |
| **Bonus** (API, temps réel, déploiement cloud) | +10 pts |

---

## 👥 Équipe — Green Health Lab

| Nom | Rôle |
|-----|------|
| À compléter | Chef de projet / ML Engineer |
| À compléter | Data Scientist |
| À compléter | Développeur Dashboard |
| À compléter | Communication & Pitch |

---

## 📄 Licence

MIT License — IndabaX Green Health Lab 2026
