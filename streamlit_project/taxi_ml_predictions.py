import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import os
import joblib
import hashlib
from math import radians, cos, sin, asin, sqrt

# ============= CHEMINS DES MODÈLES SAUVEGARDÉS =============
MODEL_DIR = "models"
TIP_MODEL_PATH = os.path.join(MODEL_DIR, "tip_predictor.joblib")
CLUSTER_MODEL_PATH = os.path.join(MODEL_DIR, "cluster_model.joblib")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.joblib")
MODEL_SCORE_PATH = os.path.join(MODEL_DIR, "model_score.joblib")

# Créer le dossier models s'il n'existe pas
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ============= TYPES DE PAIEMENT OFFICIELS =============
PAYMENT_TYPES = {
    0: {"name": "Flex Fare trip", "icon": "🔄", "color": "#8b5cf6"},
    1: {"name": "Credit card", "icon": "💳", "color": "#00d4ff"},
    2: {"name": "Cash", "icon": "💵", "color": "#00ff88"},
    3: {"name": "No charge", "icon": "🆓", "color": "#ff9500"},
    4: {"name": "Dispute", "icon": "⚠️", "color": "#ff006e"},
    5: {"name": "Unknown", "icon": "❓", "color": "#6b6b6b"},
    6: {"name": "Voided trip", "icon": "🚫", "color": "#dc2626"}
}

def get_payment_name(code):
    """Retourne le nom du type de paiement"""
    return PAYMENT_TYPES.get(code, PAYMENT_TYPES[5])["name"]

def get_payment_icon(code):
    """Retourne l'icône du type de paiement"""
    return PAYMENT_TYPES.get(code, PAYMENT_TYPES[5])["icon"]

def get_payment_color(code):
    """Retourne la couleur du type de paiement"""
    return PAYMENT_TYPES.get(code, PAYMENT_TYPES[5])["color"]


class TaxiMLPredictor:
    """
    Module de prédiction ML pour le dashboard NYC Taxi
    """
    
    def __init__(self):
        self.tip_predictor = None
        self.customer_clusterer = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prépare les features pour les modèles"""
        df = df.copy()
        
        # Extraction temporelle
        df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
        df['pickup_day'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.dayofweek
        df['pickup_month'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.month
        df['is_weekend'] = df['pickup_day'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['pickup_hour'].isin([7,8,9,17,18,19]).astype(int)
        
        # Durée du trajet
        df['trip_duration'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - 
                              pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds() / 60
        
        # Vitesse moyenne
        df['avg_speed'] = np.where(df['trip_duration'] > 0, 
                                   df['trip_distance'] / (df['trip_duration'] / 60), 0)
        
        # Ratio prix/distance
        df['price_per_mile'] = np.where(df['trip_distance'] > 0,
                                        df['fare_amount'] / df['trip_distance'], 0)
        
        # Normalisation du payment_type (s'assurer qu'il est dans la plage valide 0-6)
        if 'payment_type' in df.columns:
            df['payment_type'] = df['payment_type'].clip(0, 6).fillna(5).astype(int)
        
        return df
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calcule la distance en miles entre deux points GPS (formule de Haversine)"""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        miles = 3959 * c
        return miles
    
    def estimate_fare(self, distance, is_rush_hour=False, payment_type=1):
        """Estime le tarif en fonction de la distance et du type de paiement"""
        base_fare = 2.50
        per_mile = 2.50
        rush_hour_surcharge = 1.00 if is_rush_hour else 0
        
        # Ajustements selon le type de paiement
        if payment_type == 0:  # Flex Fare - tarif variable
            per_mile = 2.25  # Légèrement réduit
        elif payment_type == 3:  # No charge
            return 0.0
        elif payment_type == 6:  # Voided trip
            return 0.0
        
        fare = base_fare + (distance * per_mile) + rush_hour_surcharge
        return fare
    
    def train_tip_predictor(self, df):
        """Entraîne le modèle de prédiction de pourboire"""
        features = ['trip_distance', 'fare_amount', 'pickup_hour', 'pickup_day',
                   'is_weekend', 'is_rush_hour', 'passenger_count', 'payment_type',
                   'PULocationID', 'DOLocationID', 'avg_speed']
        
        X = df[features].fillna(0)
        y = df['tip_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.tip_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1  # Utiliser tous les CPU disponibles
        )
        self.tip_predictor.fit(X_train, y_train)
        
        score = self.tip_predictor.score(X_test, y_test)
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.tip_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return score, feature_importance
    
    def predict_tip(self, trip_distance, fare_amount, pickup_hour, pickup_day,
                   passenger_count, payment_type, pu_location, do_location, avg_speed):
        """Prédit le pourboire pour une course donnée"""
        is_weekend = 1 if pickup_day in [5, 6] else 0
        is_rush_hour = 1 if pickup_hour in [7, 8, 9, 17, 18, 19] else 0
        
        # Cas spéciaux où le pourboire est généralement 0
        if payment_type in [2, 3, 4, 5, 6]:  # Cash, No charge, Dispute, Unknown, Voided
            # Cash: pourboires non enregistrés électroniquement
            # No charge, Dispute, Voided: pas de transaction normale
            if payment_type == 2:  # Cash - on peut estimer un pourboire moyen
                # Estimation basée sur le tarif (15% en moyenne pour cash)
                return max(0, fare_amount * 0.15)
            else:
                return 0.0
        
        features = np.array([[
            trip_distance, fare_amount, pickup_hour, pickup_day,
            is_weekend, is_rush_hour, passenger_count, payment_type,
            pu_location, do_location, avg_speed
        ]])
        
        prediction = self.tip_predictor.predict(features)[0]
        return max(0, prediction)
    
    def train_customer_segmentation(self, df):
        """Entraîne le modèle de segmentation des clients"""
        features_cluster = ['trip_distance', 'fare_amount', 'tip_amount', 
                           'extra', 'avg_speed']
        
        X = df[features_cluster].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        self.customer_clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.customer_clusterer.fit(X_scaled)
        
        return self.scaler
    
    def predict_clusters(self, df):
        """Prédit les clusters pour de nouvelles données"""
        features_cluster = ['trip_distance', 'fare_amount', 'tip_amount', 
                           'extra', 'avg_speed']
        
        X = df[features_cluster].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        clusters = self.customer_clusterer.predict(X_scaled)
        
        # PCA pour visualisation
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        cluster_names = {
            0: "Économique",
            1: "Standard",
            2: "Premium"
        }
        
        return X_pca, clusters, cluster_names


# ============= FONCTIONS DE CACHE ET PERSISTANCE =============

def get_data_hash(df):
    """Génère un hash unique basé sur les données pour détecter les changements"""
    # Utiliser la forme et quelques statistiques pour créer un hash rapide
    data_signature = f"{df.shape}_{df['total_amount'].sum():.2f}_{df['trip_distance'].mean():.4f}"
    return hashlib.md5(data_signature.encode()).hexdigest()[:16]


def models_exist():
    """Vérifie si les modèles sauvegardés existent"""
    return (os.path.exists(TIP_MODEL_PATH) and 
            os.path.exists(FEATURE_IMPORTANCE_PATH) and
            os.path.exists(MODEL_SCORE_PATH))


def save_models(predictor, score, feature_importance):
    """Sauvegarde les modèles sur disque"""
    joblib.dump(predictor, TIP_MODEL_PATH)
    joblib.dump(score, MODEL_SCORE_PATH)
    joblib.dump(feature_importance, FEATURE_IMPORTANCE_PATH)


def load_models():
    """Charge les modèles depuis le disque"""
    predictor = joblib.load(TIP_MODEL_PATH)
    score = joblib.load(MODEL_SCORE_PATH)
    feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)
    return predictor, score, feature_importance


@st.cache_resource(show_spinner="🔄 Chargement du modèle ML...")
def get_trained_predictor(_df, data_hash):
    """
    Récupère ou entraîne le modèle de prédiction.
    Utilise le cache Streamlit + persistance sur disque.
    
    Le underscore devant _df indique à Streamlit de ne pas hasher ce paramètre.
    """
    # Vérifier si les modèles existent sur disque
    if models_exist():
        try:
            predictor, score, feature_importance = load_models()
            # Vérifier que le modèle est valide
            if predictor.tip_predictor is not None:
                return predictor, score, feature_importance
        except Exception as e:
            st.warning(f"Erreur lors du chargement du modèle: {e}. Ré-entraînement...")
    
    # Entraîner un nouveau modèle
    predictor = TaxiMLPredictor()
    df_ml = predictor.prepare_features(_df)
    score, feature_importance = predictor.train_tip_predictor(df_ml)
    
    # Entraîner aussi le modèle de clustering
    predictor.train_customer_segmentation(df_ml)
    
    # Sauvegarder sur disque
    save_models(predictor, score, feature_importance)
    
    return predictor, score, feature_importance


@st.cache_data(show_spinner="📊 Préparation des données...")
def get_prepared_features(_df, data_hash):
    """
    Prépare les features des données (avec cache).
    """
    predictor = TaxiMLPredictor()
    return predictor.prepare_features(_df)


@st.cache_data(show_spinner="🎯 Calcul de la segmentation...")
def get_clustering_visualization(_df_ml, sample_size=500):
    """
    Calcule la visualisation du clustering (avec cache).
    Réduit le nombre de points pour améliorer les performances.
    """
    # Échantillonner les données
    df_sample = _df_ml.sample(min(sample_size, len(_df_ml)), random_state=42)
    
    features_cluster = ['trip_distance', 'fare_amount', 'tip_amount', 'extra', 'avg_speed']
    X = df_sample[features_cluster].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    cluster_names = {
        0: "Économique",
        1: "Standard", 
        2: "Premium"
    }
    
    return X_pca, clusters, cluster_names


# ============= FONCTION PRINCIPALE DE VISUALISATION =============

def create_ml_visualization(df, zones_coords):
    """Crée la section ML avec visualisations D3.js - VERSION OPTIMISÉE"""
    
    st.markdown("""
    <div class="chart-header">
        <h2 class="chart-title">🤖 Intelligence Artificielle & Prédictions</h2>
        <span class="chart-badge">MACHINE LEARNING</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ============= CHARGEMENT OPTIMISÉ DES MODÈLES =============
    # Calculer le hash des données pour détecter les changements
    data_hash = get_data_hash(df)
    
    # Charger ou entraîner le modèle (CACHÉ!)
    predictor, score, feature_importance = get_trained_predictor(df, data_hash)
    
    # Préparer les features (CACHÉ!)
    df_ml = get_prepared_features(df, data_hash)
    
    # Indicateur de statut du modèle
    model_status = "✅ Modèle chargé depuis le cache" if models_exist() else "🆕 Nouveau modèle entraîné"
    
    tab1, tab2 = st.tabs([
        " Prédiction de Pourboire",
        " Segmentation Clients"
        
    ])
    
    with tab1:
        st.markdown("####  Prédicteur de Pourboire")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 1px solid #252525; 
                 border-radius: 8px; padding: 0.6rem 0.85rem; margin-bottom: 1rem;">
                <p style="color: #a8a8a8; margin: 0; font-size: 0.75rem;">
                    <strong style="color: #00ff88;">{model_status}</strong> • 
                    Score R²: <strong style="color: #00d4ff;">{score:.2%}</strong> • 
                    Random Forest (100 arbres)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Zones populaires avec coordonnées
            popular_zones = {
                'Manhattan Midtown': {'id': 161, 'lat': 40.7580, 'lon': -73.9855},
                'JFK Airport': {'id': 132, 'lat': 40.6413, 'lon': -73.7781},
                'LaGuardia Airport': {'id': 138, 'lat': 40.7769, 'lon': -73.8740},
                'Times Square': {'id': 236, 'lat': 40.7580, 'lon': -73.9855},
                'Upper East Side': {'id': 237, 'lat': 40.7736, 'lon': -73.9566},
                'Upper West Side': {'id': 238, 'lat': 40.7870, 'lon': -73.9754},
                'East Village': {'id': 79, 'lat': 40.7264, 'lon': -73.9818},
                'Financial District': {'id': 87, 'lat': 40.7074, 'lon': -74.0113},
                'Brooklyn Heights': {'id': 61, 'lat': 40.6958, 'lon': -73.9961},
                'Williamsburg': {'id': 265, 'lat': 40.7081, 'lon': -73.9571},
                'Central Park': {'id': 43, 'lat': 40.7829, 'lon': -73.9654},
                'Chelsea': {'id': 68, 'lat': 40.7465, 'lon': -74.0014},
                'SoHo': {'id': 211, 'lat': 40.7233, 'lon': -74.0030},
                'Greenwich Village': {'id': 114, 'lat': 40.7336, 'lon': -74.0027}
            }
            
            col_zone1, col_zone2 = st.columns(2)
            
            with col_zone1:
                pu_zone_name = st.selectbox(
                    "🟢 Zone de départ", 
                    list(popular_zones.keys()),
                    help="Sélectionnez la zone de pickup"
                )
            
            with col_zone2:
                do_zone_name = st.selectbox(
                    "🔴 Zone d'arrivée", 
                    list(popular_zones.keys()),
                    index=1,
                    help="Sélectionnez la zone de dropoff"
                )
            
            pu_data = popular_zones[pu_zone_name]
            do_data = popular_zones[do_zone_name]
            
            calculated_distance = predictor.calculate_distance(
                pu_data['lat'], pu_data['lon'],
                do_data['lat'], do_data['lon']
            )
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #8b5cf610, #8b5cf605); 
                 border: 1px solid #8b5cf6; border-radius: 8px; padding: 0.6rem; margin: 0.6rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #a8a8a8; font-size: 0.75rem;">📍 Distance calculée:</span>
                    <span style="color: #8b5cf6; font-size: 1.1rem; font-weight: 700;">
                        {calculated_distance:.2f} miles
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                pickup_hour = st.slider("🕐 Heure", 0, 23, 12)
                day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
                pickup_day = st.selectbox("📅 Jour", range(7), format_func=lambda x: day_names[x])
            
            with col_param2:
                passenger_count = st.slider("👥 Passagers", 1, 6, 1)
                
                # Sélection du type de paiement avec tous les types officiels
                payment_options = list(PAYMENT_TYPES.keys())
                payment_type = st.selectbox(
                    "💳 Type de paiement",
                    options=payment_options,
                    format_func=lambda x: f"{PAYMENT_TYPES[x]['icon']} {PAYMENT_TYPES[x]['name']}",
                    index=1,  # Credit card par défaut
                    help="Type de paiement selon la documentation officielle NYC TLC"
                )
            
            # Afficher les détails du type de paiement sélectionné
            selected_payment = PAYMENT_TYPES[payment_type]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {selected_payment['color']}15, {selected_payment['color']}05); 
                 border: 1px solid {selected_payment['color']}; border-radius: 8px; padding: 0.5rem; margin: 0.4rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">{selected_payment['icon']}</span>
                    <div>
                        <span style="color: {selected_payment['color']}; font-weight: 600; font-size: 0.85rem;">
                            {selected_payment['name']}
                        </span>
                        <span style="color: #6b6b6b; font-size: 0.7rem; margin-left: 0.5rem;">
                            (Code: {payment_type})
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            is_rush = pickup_hour in [7, 8, 9, 17, 18, 19]
            if is_rush:
                avg_speed = max(10, 20 - (calculated_distance * 0.5))
            else:
                avg_speed = min(35, 25 + (calculated_distance * 0.3))
            
            # Calcul du tarif estimé (fonction locale pour éviter les problèmes de cache)
            def calculate_estimated_fare(distance, is_rush_hour=False, pmt_type=1):
                """Estime le tarif en fonction de la distance et du type de paiement"""
                base_fare = 2.50
                per_mile = 2.50
                rush_hour_surcharge = 1.00 if is_rush_hour else 0
                
                # Ajustements selon le type de paiement
                if pmt_type == 0:  # Flex Fare - tarif variable
                    per_mile = 2.25  # Légèrement réduit
                elif pmt_type == 3:  # No charge
                    return 0.0
                elif pmt_type == 6:  # Voided trip
                    return 0.0
                
                fare = base_fare + (distance * per_mile) + rush_hour_surcharge
                return fare
            
            estimated_fare = calculate_estimated_fare(calculated_distance, is_rush, payment_type)
            
            # Message d'avertissement pour certains types de paiement
            if payment_type in [3, 6]:  # No charge ou Voided
                st.markdown(f"""
                <div style="background: rgba(255, 0, 110, 0.1); border: 1px solid #ff006e; 
                     border-radius: 8px; padding: 0.5rem; margin: 0.4rem 0;">
                    <p style="color: #ff006e; margin: 0; font-size: 0.75rem;">
                        ⚠️ <strong>{selected_payment['name']}</strong>: Pas de tarif applicable
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif payment_type == 4:  # Dispute
                st.markdown(f"""
                <div style="background: rgba(255, 149, 0, 0.1); border: 1px solid #ff9500; 
                     border-radius: 8px; padding: 0.5rem; margin: 0.4rem 0;">
                    <p style="color: #ff9500; margin: 0; font-size: 0.75rem;">
                        ⚠️ <strong>Dispute</strong>: Transaction contestée - pourboire non prévisible
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif payment_type == 2:  # Cash
                st.markdown(f"""
                <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; 
                     border-radius: 8px; padding: 0.5rem; margin: 0.4rem 0;">
                    <p style="color: #00ff88; margin: 0; font-size: 0.75rem;">
                        💵 <strong>Cash</strong>: Pourboire estimé (~15% du tarif) - non enregistré électroniquement
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 1px solid #252525; 
                 border-radius: 8px; padding: 0.6rem; margin: 0.6rem 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                    <div>
                        <span style="color: #a8a8a8; font-size: 0.7rem;">💵 Tarif estimé:</span>
                        <div style="color: #00d4ff; font-size: 1rem; font-weight: 700;">${estimated_fare:.2f}</div>
                    </div>
                    <div>
                        <span style="color: #a8a8a8; font-size: 0.7rem;">⚡ Vitesse estimée:</span>
                        <div style="color: #00ff88; font-size: 1rem; font-weight: 700;">{avg_speed:.0f} mph</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎯 PRÉDIRE LE POURBOIRE", use_container_width=True, type="primary"):
                predicted_tip = predictor.predict_tip(
                    calculated_distance, estimated_fare, pickup_hour, pickup_day,
                    passenger_count, payment_type, pu_data['id'], do_data['id'], avg_speed
                )
                
                tip_percentage = (predicted_tip / estimated_fare * 100) if estimated_fare > 0 else 0
                total_amount = estimated_fare + predicted_tip
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ff006e10, #ff006e05); 
                         border: 1px solid #ff006e; border-radius: 10px; padding: 1rem; text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">💰</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #ff006e;">${predicted_tip:.2f}</div>
                        <div style="font-size: 0.65rem; color: #a8a8a8; text-transform: uppercase;">Pourboire Prédit</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #00d4ff10, #00d4ff05); 
                         border: 1px solid #00d4ff; border-radius: 10px; padding: 1rem; text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">📊</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #00d4ff;">{tip_percentage:.1f}%</div>
                        <div style="font-size: 0.65rem; color: #a8a8a8; text-transform: uppercase;">Pourcentage</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res3:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #00ff8810, #00ff8805); 
                         border: 1px solid #00ff88; border-radius: 10px; padding: 1rem; text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🧾</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #00ff88;">${total_amount:.2f}</div>
                        <div style="font-size: 0.65rem; color: #a8a8a8; text-transform: uppercase;">Total Course</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                avg_tip = df_ml['tip_amount'].mean()
                comparison = "supérieur" if predicted_tip > avg_tip else "inférieur"
                diff_pct = abs((predicted_tip - avg_tip) / avg_tip * 100) if avg_tip > 0 else 0
                is_weekend = pickup_day in [5, 6]
                
                st.markdown(f"""
                <div style="background: #1a1a1a; border: 1px solid #252525; 
                     border-radius: 8px; padding: 0.85rem; margin-top: 0.85rem; border-left: 3px solid #8b5cf6;">
                    <h4 style="color: #8b5cf6; margin: 0 0 0.5rem 0; font-size: 0.8rem;">📋 Analyse</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; color: #a8a8a8; font-size: 0.75rem; line-height: 1.5;">
                        <div>
                            <p style="margin: 0;"><strong style="color: #fff;">Trajet:</strong> {pu_zone_name[:12]} → {do_zone_name[:12]}</p>
                            <p style="margin: 0;">📏 {calculated_distance:.2f} miles</p>
                        </div>
                        <div>
                            <p style="margin: 0;">{'🚗 Rush hour' if is_rush else '🛣️ Heure normale'} • {'📅 Weekend' if is_weekend else '💼 Semaine'}</p>
                            <p style="margin: 0;">{selected_payment['icon']} {selected_payment['name']}</p>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #252525; font-size: 0.75rem;">
                        <p style="margin: 0;">Pourboire <strong style="color: {'#00ff88' if predicted_tip > avg_tip else '#ff006e'};">{comparison}</strong> à la moyenne (${avg_tip:.2f}) — diff: <strong style="color: #00d4ff;">{diff_pct:.1f}%</strong></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="guide-box">
                <h4>📖 Guide</h4>
                <p><strong>Itinéraire:</strong></p>
                <p>• Choisir départ/arrivée</p>
                <p>• Distance auto-calculée</p>
                <p style="margin-top: 0.4rem;"><strong>Facteurs:</strong></p>
                <p>• Rush hours (7-9h, 17-19h)</p>
                <p>• Carte = + tips</p>
                <p style="margin-top: 0.4rem;"><strong>🎯 Score R²:</strong></p>
                <p style="color: #00ff88; font-weight: 600;">{score:.1%}</p>
                <p style="margin-top: 0.4rem;"><strong>📊 Top Variables:</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, row in feature_importance.head(4).iterrows():
                st.markdown(f"""
                <div style="margin: 4px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span style="font-size: 0.65rem; color: #a8a8a8;">{row['feature'][:12]}</span>
                        <span style="font-size: 0.65rem; color: #00d4ff; font-weight: 600;">{row['importance']*100:.1f}%</span>
                    </div>
                    <div style="background: #252525; height: 4px; border-radius: 2px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #ff006e, #00d4ff); width: {row['importance']*100}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### 🎯 Segmentation des Clients par ML")
        
        # Clustering avec CACHE - ne recalcule que si nécessaire
        X_pca, clusters, cluster_names = get_clustering_visualization(df_ml, sample_size=500)
        
        scatter_data = []
        for i in range(len(X_pca)):
            scatter_data.append({
                'x': float(X_pca[i, 0]),
                'y': float(X_pca[i, 1]),
                'z': float(X_pca[i, 2]) if X_pca.shape[1] > 2 else 0,
                'cluster': int(clusters[i]),
                'name': cluster_names[clusters[i]]
            })
        
        html_3d = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script src="https://unpkg.com/d3-3d@0.0.10/build/d3-3d.min.js"></script>
            <style>
                body {{ 
                    font-family: 'Inter', sans-serif; 
                    background: #141414;
                    margin: 0;
                    padding: 12px;
                }}
                .point {{
                    cursor: pointer;
                    transition: all 0.25s ease;
                }}
                .point:hover {{
                    r: 6;
                    filter: brightness(1.4) drop-shadow(0 0 8px currentColor);
                }}
                .legend-item {{
                    cursor: pointer;
                }}
                .title {{
                    fill: #00ff88;
                    font-size: 13px;
                    font-weight: 700;
                }}
            </style>
        </head>
        <body>
            <div id="scatter3d"></div>
            <script>
                const data = {json.dumps(scatter_data)};
                
                const width = 700;
                const height = 420;
                
                const svg = d3.select("#scatter3d")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                const g = svg.append("g")
                    .attr("transform", "translate(350,210)");
                
                svg.append("text")
                    .attr("class", "title")
                    .attr("x", width / 2)
                    .attr("y", 22)
                    .attr("text-anchor", "middle")
                    .text("🎯 Segmentation 3D - {len(scatter_data)} points");
                
                const colors = ["#ff006e", "#00d4ff", "#00ff88"];
                const clusterNames = ["Économique", "Standard", "Premium"];
                
                let alpha = 0;
                let beta = 0;
                let startAngle = Math.PI/4;
                
                const point3d = d3._3d()
                    .x(d => d.x * 80)
                    .y(d => d.y * 80)
                    .z(d => d.z * 80)
                    .origin([0, 0])
                    .rotateY(startAngle)
                    .rotateX(-startAngle);
                
                const points3d = point3d(data);
                
                const points = g.selectAll(".point")
                    .data(points3d)
                    .enter()
                    .append("circle")
                    .attr("class", "point")
                    .attr("cx", d => d.projected.x)
                    .attr("cy", d => d.projected.y)
                    .attr("r", 3)
                    .attr("fill", d => colors[d.cluster])
                    .attr("fill-opacity", 0.65)
                    .attr("stroke", "#ffffff")
                    .attr("stroke-width", 0.4)
                    .style("opacity", 0);
                
                points.transition()
                    .duration(1500)
                    .delay((d, i) => i * 1.5)
                    .style("opacity", 1);
                
                const legend = svg.append("g")
                    .attr("transform", "translate(40, 70)");
                
                clusterNames.forEach((name, i) => {{
                    const item = legend.append("g")
                        .attr("class", "legend-item")
                        .attr("transform", `translate(0, ${{i * 20}})`)
                        .on("mouseover", function() {{
                            points.style("opacity", d => d.cluster === i ? 1 : 0.1);
                        }})
                        .on("mouseout", function() {{
                            points.style("opacity", 1);
                        }});
                    
                    item.append("circle")
                        .attr("r", 5)
                        .attr("fill", colors[i]);
                    
                    item.append("text")
                        .attr("x", 12)
                        .attr("y", 4)
                        .attr("fill", "#ffffff")
                        .style("font-size", "10px")
                        .text(name);
                }});
                
                function rotate() {{
                    alpha += 0.008;
                    beta += 0.004;
                    
                    const newPoints = point3d.rotateY(alpha).rotateX(beta)(data);
                    
                    points.data(newPoints)
                        .attr("cx", d => d.projected.x)
                        .attr("cy", d => d.projected.y);
                    
                    requestAnimationFrame(rotate);
                }}
                
                rotate();
            </script>
        </body>
        </html>
        """
        
        st.components.v1.html(html_3d, height=450)
        
        cluster_stats = pd.DataFrame(clusters, columns=['cluster'])
        cluster_counts = cluster_stats['cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Distribution des Segments")
            for i, name in cluster_names.items():
                if i in cluster_counts.index:
                    pct = cluster_counts[i] / len(clusters) * 100
                    st.markdown(f"""
                    <div style="margin: 6px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                            <span>{name}</span>
                            <span style="color: #00d4ff;">{pct:.1f}%</span>
                        </div>
                        <div style="background: #252525; height: 12px; border-radius: 6px;">
                            <div style="background: linear-gradient(90deg, #ff006e, #00d4ff); 
                                       width: {pct}%; height: 100%; border-radius: 6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### Caractéristiques")
            st.markdown("""
            <div class="guide-box">
                <p><strong>🔴 Économique:</strong> Courts trajets, faibles tarifs</p>
                <p><strong>🔵 Standard:</strong> Trajets moyens, tips standards</p>
                <p><strong>🟢 Premium:</strong> Longues distances, tips élevés</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton pour forcer le ré-entraînement si nécessaire
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🔄 Forcer ré-entraînement", help="Supprime le cache et ré-entraîne les modèles"):
                # Supprimer les fichiers de modèles
                for path in [TIP_MODEL_PATH, CLUSTER_MODEL_PATH, FEATURE_IMPORTANCE_PATH, MODEL_SCORE_PATH]:
                    if os.path.exists(path):
                        os.remove(path)
                # Vider le cache Streamlit
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
    
    
    

__all__ = ['create_ml_visualization', 'TaxiMLPredictor', 'PAYMENT_TYPES', 'get_payment_name', 'get_payment_icon', 'get_payment_color']
