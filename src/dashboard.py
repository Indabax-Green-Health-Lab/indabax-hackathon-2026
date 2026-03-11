"""
dashboard.py
------------
Dashboard interactif — Qualité de l'air au Cameroun
IndabaX Hackathon 2026 — Green Health Lab

Lancer : streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 Qualité de l'air — Cameroun",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path("data/processed/data_clean.csv")


# ─────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        # Données de démonstration si fichier absent
        st.warning("⚠️ Fichier de données non trouvé. Affichage en mode démo.")
        np.random.seed(42)
        cities = ["Yaoundé", "Douala", "Garoua", "Maroua", "Ngaoundéré",
                  "Bertoua", "Ebolowa", "Bafoussam", "Bamenda", "Kribi"]
        regions = ["Centre", "Littoral", "Nord", "Extrême-Nord", "Adamaoua",
                   "Est", "Sud", "Ouest", "Nord-Ouest", "Sud-Ouest"]
        dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
        records = []
        for city, region in zip(cities, regions):
            for date in dates:
                records.append({
                    "date": date, "city": city, "region": region,
                    "temp_max": np.random.normal(32, 5),
                    "humidity": np.random.normal(70, 15),
                    "wind_speed": np.random.exponential(3),
                    "pm25_proxy": np.random.gamma(3, 10),
                    "pollution_risk": np.random.choice([0, 1], p=[0.7, 0.3]),
                    "season": np.random.choice(["Saison sèche", "Saison pluvieuse", "Transition"])
                })
        return pd.DataFrame(records)


df = load_data()

# ─────────────────────────────────────────────────
# Sidebar — Filtres
# ─────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/IndabaX-Cameroon%202026-green", width=200)
st.sidebar.title("🔧 Filtres")

regions = sorted(df["region"].unique())
selected_regions = st.sidebar.multiselect("Régions", regions, default=regions[:3])

year_range = st.sidebar.slider(
    "Période",
    min_value=int(df["date"].dt.year.min()),
    max_value=int(df["date"].dt.year.max()),
    value=(2023, 2025)
)

# Filtrage
filtered = df[
    (df["region"].isin(selected_regions)) &
    (df["date"].dt.year.between(*year_range))
]

# ─────────────────────────────────────────────────
# En-tête principal
# ─────────────────────────────────────────────────
st.title("🌍 Tableau de bord — Qualité de l'air au Cameroun")
st.caption("IndabaX Cameroon Hackathon 2026 | Green Health Lab")
st.divider()

# ─────────────────────────────────────────────────
# KPIs — métriques clés
# ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    avg_pm25 = filtered["pm25_proxy"].mean() if "pm25_proxy" in filtered else 0
    st.metric("🌫️ PM2.5 moyen", f"{avg_pm25:.1f} µg/m³",
              delta="⚠️ Élevé" if avg_pm25 > 35 else "✅ Acceptable")
with col2:
    risk_rate = filtered["pollution_risk"].mean() * 100 if "pollution_risk" in filtered else 0
    st.metric("🔴 Jours à risque", f"{risk_rate:.1f}%")
with col3:
    st.metric("🏙️ Villes couvertes", filtered["city"].nunique())
with col4:
    st.metric("📅 Observations", f"{len(filtered):,}")

st.divider()

# ─────────────────────────────────────────────────
# Graphiques principaux
# ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Tendances", "🗺️ Carte", "🔍 Analyse par région"])

with tab1:
    st.subheader("Évolution mensuelle du PM2.5 par région")
    monthly = (
        filtered.copy()
        .assign(month=filtered["date"].dt.to_period("M").astype(str))
        .groupby(["month", "region"])["pm25_proxy"]
        .mean()
        .reset_index()
    )
    fig = px.line(monthly, x="month", y="pm25_proxy", color="region",
                  labels={"pm25_proxy": "PM2.5 moyen (µg/m³)", "month": "Mois"},
                  template="plotly_white")
    fig.add_hline(y=35, line_dash="dash", line_color="red",
                  annotation_text="Seuil OMS (35 µg/m³)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Carte de chaleur — Risque de pollution par ville")
    city_stats = filtered.groupby(["city", "region"]).agg(
        pm25_mean=("pm25_proxy", "mean"),
        risk_rate=("pollution_risk", "mean")
    ).reset_index()

    # Coordonnées approximatives (à remplacer par les vraies du dataset)
    coords = {
        "Yaoundé": (3.848, 11.502), "Douala": (4.061, 9.742),
        "Garoua": (9.301, 13.397), "Maroua": (10.591, 14.316),
        "Ngaoundéré": (7.321, 13.584), "Bertoua": (4.578, 13.685),
        "Ebolowa": (2.900, 11.149), "Bafoussam": (5.476, 10.421),
        "Bamenda": (5.959, 10.146), "Kribi": (2.940, 9.909)
    }
    city_stats["lat"] = city_stats["city"].map(lambda c: coords.get(c, (4, 12))[0])
    city_stats["lon"] = city_stats["city"].map(lambda c: coords.get(c, (4, 12))[1])

    fig_map = px.scatter_mapbox(
        city_stats, lat="lat", lon="lon", color="pm25_mean",
        size="risk_rate", hover_name="city", hover_data=["region", "pm25_mean"],
        color_continuous_scale="Reds", mapbox_style="carto-positron",
        zoom=5, center={"lat": 5.5, "lon": 12.5},
        labels={"pm25_mean": "PM2.5 moyen"}
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("Distribution PM2.5 par saison et région")
    if "season" in filtered.columns:
        fig_box = px.box(
            filtered, x="region", y="pm25_proxy", color="season",
            labels={"pm25_proxy": "PM2.5 (µg/m³)", "region": "Région"},
            template="plotly_white"
        )
        fig_box.update_xaxes(tickangle=30)
        st.plotly_chart(fig_box, use_container_width=True)

    # Corrélation météo / pollution
    st.subheader("Corrélation : facteurs climatiques vs PM2.5")
    num_cols = ["temp_max", "humidity", "wind_speed", "pm25_proxy"]
    available = [c for c in num_cols if c in filtered.columns]
    if len(available) > 1:
        corr = filtered[available].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             aspect="auto", title="Matrice de corrélation")
        st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────
st.divider()
st.caption("🔬 Green Health Lab | IndabaX Cameroon 2026 | Données : 42 villes, 2020–2025")
