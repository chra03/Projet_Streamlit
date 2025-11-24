# Projet_Streamlit

#🚖NYC Yellow Taxi Data

📌 Contexte

Ce projet porte sur les données officielles des taxis jaunes de New York.
Chaque course contient : la date, la distance, le prix, le pourboire, le type de paiement, et les zones de pickup/dropoff.
Ces données sont idéales pour analyser la mobilité urbaine et créer des visualisations interactives.


🎯 Objectif

Créer un dashboard interactif Streamlit permettant :

-d’explorer les trajets (prix, distance, durée),

-d’afficher une cartographie des zones de pickup/dropoff,

-d’analyser les patterns temporels (heures, jours),

-d’intégrer un modèle simple de prédiction (pourboire ou prix).


📊 Fonctionnalités prévues

KPIs : prix moyen, pourboire moyen, distance moyenne

Graphiques Plotly (histogrammes, courbes temporelles, top zones)

Carte interactive (Folium ou Mapbox)

Prédiction ML (Random Forest)

Analyse des zones NYC Taxi


🚀 Lancer le projet

pip install -r requirements.txt
streamlit run accueil.py
