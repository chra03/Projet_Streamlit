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
from math import radians, cos, sin, asin, sqrt

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
        
        return df
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calcule la distance en miles entre deux points GPS (formule de Haversine)"""
        # Convertir en radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Formule de Haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Rayon de la Terre en miles
        miles = 3959 * c
        return miles
    
    def estimate_fare(self, distance, is_rush_hour=False):
        """Estime le tarif en fonction de la distance"""
        base_fare = 2.50
        per_mile = 2.50
        rush_hour_surcharge = 1.00 if is_rush_hour else 0
        
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
            random_state=42
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
        
        features = np.array([[
            trip_distance, fare_amount, pickup_hour, pickup_day,
            is_weekend, is_rush_hour, passenger_count, payment_type,
            pu_location, do_location, avg_speed
        ]])
        
        prediction = self.tip_predictor.predict(features)[0]
        return max(0, prediction)
    
    def customer_segmentation(self, df):
        """Segmentation des clients"""
        features_cluster = ['trip_distance', 'fare_amount', 'tip_amount', 
                           'extra', 'avg_speed']
        
        X = df[features_cluster].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # KMeans clustering avec k=3
        self.customer_clusterer = KMeans(n_clusters=3, random_state=42)
        clusters = self.customer_clusterer.fit_predict(X_scaled)
        
        # PCA pour visualisation
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        cluster_names = {
            0: "Économique",
            1: "Standard",
            2: "Premium"
        }
        
        return X_pca, clusters, cluster_names


def create_ml_visualization(df, zones_coords):
    """Crée la section ML avec visualisations D3.js"""
    
    st.markdown("""
    <div class="chart-header">
        <h2 class="chart-title">🤖 Intelligence Artificielle & Prédictions</h2>
        <span class="chart-badge">MACHINE LEARNING</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialiser le prédicteur
    predictor = TaxiMLPredictor()
    
    # Préparer les données
    df_ml = predictor.prepare_features(df)
    
    # Entraîner le modèle
    score, feature_importance = predictor.train_tip_predictor(df_ml)
    
    # Tabs pour différentes analyses
    tab1, tab2 = st.tabs([
        "🎯 Prédiction de Pourboire",
        "👥 Segmentation Clients"
    ])
    
    with tab1:
        st.markdown("### 🔮 Prédicteur de Pourboire avec Random Forest")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: var(--bg-tertiary); border: 1px solid var(--border-color); 
                 border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                <p style="color: var(--text-secondary); margin: 0;">
                    <strong style="color: var(--accent-green);">Modèle entraîné</strong> • 
                    Score R²: <strong style="color: var(--accent-blue);">{score:.2%}</strong> • 
                    Random Forest avec 100 arbres
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🗺️ Itinéraire du Trajet")
            
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
            
            # Calculer automatiquement la distance
            pu_data = popular_zones[pu_zone_name]
            do_data = popular_zones[do_zone_name]
            
            calculated_distance = predictor.calculate_distance(
                pu_data['lat'], pu_data['lon'],
                do_data['lat'], do_data['lon']
            )
            
            # Afficher la distance calculée
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #8b5cf615, #8b5cf605); 
                 border: 1px solid #8b5cf6; border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #a8a8a8;">📏 Distance calculée:</span>
                    <span style="color: #8b5cf6; font-size: 1.5rem; font-weight: 700;">
                        {calculated_distance:.2f} miles
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### ⚙️ Paramètres de la Course")
            
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                pickup_hour = st.slider(
                    "🕐 Heure de pickup", 
                    0, 23, 12,
                    help="Heure de prise en charge (0-23h)"
                )
                
                day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
                pickup_day = st.selectbox(
                    "📅 Jour de la semaine", 
                    range(7), 
                    format_func=lambda x: day_names[x],
                    help="Jour de la semaine"
                )
            
            with col_param2:
                passenger_count = st.slider(
                    "👥 Nombre de passagers", 
                    1, 6, 1,
                    help="Nombre de passagers dans le taxi"
                )
                
                payment_type = st.selectbox(
                    "💳 Type de paiement", 
                    [1, 2], 
                    format_func=lambda x: "Carte bancaire" if x == 1 else "Espèces",
                    help="Mode de paiement"
                )
            
            # Calculer la vitesse moyenne estimée (basée sur l'heure et la distance)
            is_rush = pickup_hour in [7, 8, 9, 17, 18, 19]
            if is_rush:
                avg_speed = max(10, 20 - (calculated_distance * 0.5))  # Plus lent en rush hour
            else:
                avg_speed = min(35, 25 + (calculated_distance * 0.3))  # Plus rapide hors rush
            
            # Calculer le tarif estimé
            estimated_fare = predictor.estimate_fare(calculated_distance, is_rush)
            
            st.markdown(f"""
            <div style="background: var(--bg-tertiary); border: 1px solid var(--border-color); 
                 border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <span style="color: #a8a8a8; font-size: 0.875rem;">💵 Tarif estimé:</span>
                        <div style="color: #00d4ff; font-size: 1.25rem; font-weight: 700;">
                            ${estimated_fare:.2f}
                        </div>
                    </div>
                    <div>
                        <span style="color: #a8a8a8; font-size: 0.875rem;">⚡ Vitesse estimée:</span>
                        <div style="color: #00ff88; font-size: 1.25rem; font-weight: 700;">
                            {avg_speed:.0f} mph
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🎯 PRÉDIRE LE POURBOIRE", use_container_width=True, type="primary"):
                # Faire la prédiction
                predicted_tip = predictor.predict_tip(
                    calculated_distance, estimated_fare, pickup_hour, pickup_day,
                    passenger_count, payment_type, pu_data['id'], do_data['id'], avg_speed
                )
                
                tip_percentage = (predicted_tip / estimated_fare * 100) if estimated_fare > 0 else 0
                total_amount = estimated_fare + predicted_tip
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Résultats de la Prédiction")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ff006e15, #ff006e05); 
                         border: 2px solid #ff006e; border-radius: 16px; padding: 2rem; text-align: center;">
                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">💵</div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: #ff006e; margin-bottom: 0.5rem;">
                            ${predicted_tip:.2f}
                        </div>
                        <div style="font-size: 0.875rem; color: #a8a8a8; text-transform: uppercase; letter-spacing: 0.05em;">
                            Pourboire Prédit
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #00d4ff15, #00d4ff05); 
                         border: 2px solid #00d4ff; border-radius: 16px; padding: 2rem; text-align: center;">
                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: #00d4ff; margin-bottom: 0.5rem;">
                            {tip_percentage:.1f}%
                        </div>
                        <div style="font-size: 0.875rem; color: #a8a8a8; text-transform: uppercase; letter-spacing: 0.05em;">
                            Pourcentage
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res3:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #00ff8815, #00ff8805); 
                         border: 2px solid #00ff88; border-radius: 16px; padding: 2rem; text-align: center;">
                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">💰</div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: #00ff88; margin-bottom: 0.5rem;">
                            ${total_amount:.2f}
                        </div>
                        <div style="font-size: 0.875rem; color: #a8a8a8; text-transform: uppercase; letter-spacing: 0.05em;">
                            Total Course
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Comparaison avec la moyenne
                avg_tip = df_ml['tip_amount'].mean()
                comparison = "supérieur" if predicted_tip > avg_tip else "inférieur"
                diff_pct = abs((predicted_tip - avg_tip) / avg_tip * 100)
                
                is_weekend = pickup_day in [5, 6]
                
                st.markdown(f"""
                <div style="background: var(--bg-tertiary); border: 1px solid var(--border-color); 
                     border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; border-left: 4px solid #8b5cf6;">
                    <h4 style="color: #8b5cf6; margin-top: 0;">📈 Analyse Détaillée</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; color: var(--text-secondary); line-height: 1.8;">
                        <div>
                            <p><strong style="color: #fff;">Trajet:</strong></p>
                            <p>🟢 {pu_zone_name}</p>
                            <p>🔴 {do_zone_name}</p>
                            <p>📏 {calculated_distance:.2f} miles</p>
                        </div>
                        <div>
                            <p><strong style="color: #fff;">Contexte:</strong></p>
                            <p>{'🔥 Rush hour' if is_rush else '✨ Heure normale'}</p>
                            <p>{'🎉 Weekend' if is_weekend else '💼 Semaine'}</p>
                            <p>{'💳 Carte' if payment_type == 1 else '💵 Cash'}</p>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #2a2a2a;">
                        <p>Le pourboire prédit est <strong style="color: {'#00ff88' if predicted_tip > avg_tip else '#ff006e'};">
                        {comparison}</strong> à la moyenne de <strong>${avg_tip:.2f}</strong> 
                        (différence: <strong style="color: #00d4ff;">{diff_pct:.1f}%</strong>)</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="guide-box">
                <h4>💡 Guide d'Utilisation</h4>
                <p><strong>1. Sélection Itinéraire:</strong></p>
                <p>• Choisissez départ et arrivée</p>
                <p>• Distance calculée automatiquement</p>
                <p>• Tarif estimé selon le trajet</p>
                
                <p><strong>2. Facteurs Temporels:</strong></p>
                <p>• Rush hours (7-9h, 17-19h)</p>
                <p>• Weekend vs Semaine</p>
                <p>• Impact sur vitesse et tarif</p>
                
                <p><strong>3. Autres Paramètres:</strong></p>
                <p>• Type de paiement (carte = + tips)</p>
                <p>• Nombre de passagers</p>
                <p>• Zones premium (aéroports)</p>
                
                <p style="margin-top: 20px;"><strong>🎯 Précision:</strong></p>
                <p>Score R² du modèle: <strong style="color: #00ff88;">{:.1%}</strong></p>
                
                <p style="margin-top: 20px;"><strong>🔝 Top Variables:</strong></p>
            """.format(score), unsafe_allow_html=True)
            
            # Afficher top 5 features
            for idx, row in feature_importance.head(5).iterrows():
                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 0.75rem; color: #a8a8a8;">{row['feature']}</span>
                        <span style="font-size: 0.75rem; color: #00d4ff; font-weight: 600;">
                            {row['importance']*100:.1f}%
                        </span>
                    </div>
                    <div style="background: #2a2a2a; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #ff006e, #00d4ff); 
                             width: {row['importance']*100}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Segmentation des Clients par Machine Learning")
        
        # Clustering
        X_pca, clusters, cluster_names = predictor.customer_segmentation(df_ml.sample(min(1000, len(df_ml))))
        
        # Préparer les données pour visualisation 3D
        scatter_data = []
        for i in range(len(X_pca)):
            scatter_data.append({
                'x': float(X_pca[i, 0]),
                'y': float(X_pca[i, 1]),
                'z': float(X_pca[i, 2]) if X_pca.shape[1] > 2 else 0,
                'cluster': int(clusters[i]),
                'name': cluster_names[clusters[i]]
            })
        
        # Visualisation D3.js 3D
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
                    padding: 20px;
                }}
                .point {{
                    cursor: pointer;
                    transition: all 0.3s ease;
                }}
                .point:hover {{
                    r: 8;
                    filter: brightness(1.5) drop-shadow(0 0 10px currentColor);
                }}
                .legend-item {{
                    cursor: pointer;
                }}
                .title {{
                    fill: #00ff88;
                    font-size: 16px;
                    font-weight: 700;
                }}
            </style>
        </head>
        <body>
            <div id="scatter3d"></div>
            <script>
                const data = {json.dumps(scatter_data)};
                
                const width = 800;
                const height = 600;
                
                const svg = d3.select("#scatter3d")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                const g = svg.append("g")
                    .attr("transform", "translate(400,300)");
                
                // Title
                svg.append("text")
                    .attr("class", "title")
                    .attr("x", width / 2)
                    .attr("y", 30)
                    .attr("text-anchor", "middle")
                    .text("👥 Segmentation 3D des Clients - KMeans (k=3)");
                
                const colors = ["#ff006e", "#00d4ff", "#00ff88"];
                const clusterNames = ["Économique", "Standard", "Premium"];
                
                // 3D projection
                let alpha = 0;
                let beta = 0;
                let startAngle = Math.PI/4;
                
                const point3d = d3._3d()
                    .x(d => d.x * 100)
                    .y(d => d.y * 100)
                    .z(d => d.z * 100)
                    .origin([0, 0])
                    .rotateY(startAngle)
                    .rotateX(-startAngle);
                
                const points3d = point3d(data);
                
                // Draw points
                const points = g.selectAll(".point")
                    .data(points3d)
                    .enter()
                    .append("circle")
                    .attr("class", "point")
                    .attr("cx", d => d.projected.x)
                    .attr("cy", d => d.projected.y)
                    .attr("r", 4)
                    .attr("fill", d => colors[d.cluster])
                    .attr("fill-opacity", 0.7)
                    .attr("stroke", "#ffffff")
                    .attr("stroke-width", 0.5)
                    .style("opacity", 0);
                
                // Animate appearance
                points.transition()
                    .duration(2000)
                    .delay((d, i) => i * 2)
                    .style("opacity", 1);
                
                // Legend
                const legend = svg.append("g")
                    .attr("transform", "translate(50, 100)");
                
                clusterNames.forEach((name, i) => {{
                    const item = legend.append("g")
                        .attr("class", "legend-item")
                        .attr("transform", `translate(0, ${{i * 25}})`)
                        .on("mouseover", function() {{
                            points.style("opacity", d => d.cluster === i ? 1 : 0.1);
                        }})
                        .on("mouseout", function() {{
                            points.style("opacity", 1);
                        }});
                    
                    item.append("circle")
                        .attr("r", 6)
                        .attr("fill", colors[i]);
                    
                    item.append("text")
                        .attr("x", 15)
                        .attr("y", 5)
                        .attr("fill", "#ffffff")
                        .style("font-size", "12px")
                        .text(name);
                }});
                
                // Rotation animation
                function rotate() {{
                    alpha += 0.01;
                    beta += 0.005;
                    
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
        
        st.components.v1.html(html_3d, height=650)
        
        # Statistiques des clusters
        cluster_stats = pd.DataFrame(clusters, columns=['cluster'])
        cluster_counts = cluster_stats['cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribution des Segments")
            for i, name in cluster_names.items():
                if i in cluster_counts.index:
                    pct = cluster_counts[i] / len(clusters) * 100
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{name}</span>
                            <span>{pct:.1f}%</span>
                        </div>
                        <div style="background: #2a2a2a; height: 20px; border-radius: 10px;">
                            <div style="background: linear-gradient(90deg, #ff006e, #00d4ff); 
                                       width: {pct}%; height: 100%; border-radius: 10px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Caractéristiques des Segments")
            st.markdown("""
            <div class="guide-box">
                <p><strong>🔴 Économique:</strong> Courts trajets, faibles tarifs, vitesse modérée</p>
                <p><strong>🔵 Standard:</strong> Trajets moyens, tips standards, extras occasionnels</p>
                <p><strong>🟢 Premium:</strong> Longues distances, tips élevés, vitesse rapide, extras fréquents</p>
            </div>
            """, unsafe_allow_html=True)

# Export de la fonction pour intégration
__all__ = ['create_ml_visualization', 'TaxiMLPredictor']
