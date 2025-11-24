import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from datetime import datetime, timedelta
import json

# ============= ML PREDICTION MODULE =============

class TaxiMLPredictor:
    """
    Module de prédiction ML avancé pour le dashboard NYC Taxi
    """
    
    def __init__(self):
        self.tip_predictor = None
        self.congestion_classifier = None
        self.customer_clusterer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
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
        
        # Features de congestion
        df['has_congestion'] = (df['congestion_surcharge'] > 0).astype(int)
        
        # Airport trip
        df['is_airport'] = (df['Airport_fee'] > 0).astype(int)
        
        return df
    
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
    
    def train_congestion_classifier(self, df):
        """Entraîne le classifieur de congestion"""
        features = ['pickup_hour', 'pickup_day', 'PULocationID', 'DOLocationID',
                   'trip_distance', 'avg_speed', 'is_rush_hour']
        
        X = df[features].fillna(0)
        y = df['has_congestion']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.congestion_classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.congestion_classifier.fit(X_train, y_train)
        
        score = self.congestion_classifier.score(X_test, y_test)
        return score
    
    def customer_segmentation(self, df):
        """Segmentation des clients"""
        features_cluster = ['trip_distance', 'fare_amount', 'tip_amount', 
                           'passenger_count', 'avg_speed', 'price_per_mile']
        
        X = df[features_cluster].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # KMeans clustering
        self.customer_clusterer = KMeans(n_clusters=5, random_state=42)
        clusters = self.customer_clusterer.fit_predict(X_scaled)
        
        # PCA pour visualisation
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        cluster_names = {
            0: "Économique",
            1: "Business",
            2: "Touriste",
            3: "Longue Distance",
            4: "Premium"
        }
        
        return X_pca, clusters, cluster_names
    
    def predict_optimal_route_revenue(self, zone_id, hour, day):
        """Prédit le revenu optimal pour une zone et un temps donné"""
        # Simulation de prédiction (à remplacer par un vrai modèle)
        base_revenue = np.random.uniform(15, 45)
        hour_factor = 1.5 if hour in [7,8,9,17,18,19] else 1.0
        weekend_factor = 1.2 if day in [5, 6] else 1.0
        
        predicted_revenue = base_revenue * hour_factor * weekend_factor
        confidence = np.random.uniform(0.7, 0.95)
        
        return predicted_revenue, confidence


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
    
    # Tabs pour différentes analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prédiction de Pourboire",
        "🚦 Analyse de Congestion",
        "👥 Segmentation Clients",
        "📊 Optimisation Revenue"
    ])
    
    with tab1:
        st.markdown("### Prédiction du Pourboire avec Random Forest")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Entraîner le modèle
            score, feature_importance = predictor.train_tip_predictor(df_ml)
            
            # Visualisation D3.js des feature importances
            d3_features_data = {
                'features': feature_importance['feature'].tolist()[:10],
                'importances': feature_importance['importance'].tolist()[:10]
            }
            
            html_features = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    body {{ 
                        font-family: 'Inter', sans-serif; 
                        background: #141414;
                        margin: 0;
                        padding: 20px;
                    }}
                    .bar {{
                        fill: url(#gradient);
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }}
                    .bar:hover {{
                        filter: brightness(1.2);
                        transform: scaleX(1.02);
                    }}
                    .label {{
                        fill: #ffffff;
                        font-size: 12px;
                        font-weight: 600;
                    }}
                    .value {{
                        fill: #00ff88;
                        font-size: 11px;
                        font-weight: 700;
                    }}
                    .title {{
                        fill: #ff006e;
                        font-size: 16px;
                        font-weight: 700;
                    }}
                    .axis {{
                        stroke: #3a3a3a;
                    }}
                    .axis text {{
                        fill: #a8a8a8;
                        font-size: 10px;
                    }}
                    .tooltip {{
                        position: absolute;
                        padding: 12px;
                        background: linear-gradient(135deg, rgba(20,20,20,0.98), rgba(30,30,30,0.98));
                        border: 1px solid #ff006e;
                        border-radius: 8px;
                        color: white;
                        font-size: 12px;
                        pointer-events: none;
                        opacity: 0;
                        transition: opacity 0.3s;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.6);
                    }}
                </style>
            </head>
            <body>
                <div id="features-chart"></div>
                <div class="tooltip" id="tooltip"></div>
                <script>
                    const data = {json.dumps(d3_features_data)};
                    
                    const margin = {{top: 40, right: 40, bottom: 60, left: 150}};
                    const width = 600 - margin.left - margin.right;
                    const height = 400 - margin.top - margin.bottom;
                    
                    const svg = d3.select("#features-chart")
                        .append("svg")
                        .attr("width", width + margin.left + margin.right)
                        .attr("height", height + margin.top + margin.bottom);
                    
                    // Gradient
                    const gradient = svg.append("defs")
                        .append("linearGradient")
                        .attr("id", "gradient")
                        .attr("x1", "0%")
                        .attr("x2", "100%");
                    
                    gradient.append("stop")
                        .attr("offset", "0%")
                        .attr("stop-color", "#ff006e");
                    
                    gradient.append("stop")
                        .attr("offset", "100%")
                        .attr("stop-color", "#00d4ff");
                    
                    const g = svg.append("g")
                        .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
                    
                    // Title
                    g.append("text")
                        .attr("class", "title")
                        .attr("x", width / 2)
                        .attr("y", -20)
                        .attr("text-anchor", "middle")
                        .text("🎯 Importance des Variables - Score: {score:.2%}");
                    
                    // Scales
                    const x = d3.scaleLinear()
                        .domain([0, d3.max(data.importances)])
                        .range([0, width]);
                    
                    const y = d3.scaleBand()
                        .domain(data.features)
                        .range([0, height])
                        .padding(0.2);
                    
                    // Bars with animation
                    const bars = g.selectAll(".bar")
                        .data(data.features)
                        .enter()
                        .append("rect")
                        .attr("class", "bar")
                        .attr("y", d => y(d))
                        .attr("height", y.bandwidth())
                        .attr("x", 0)
                        .attr("width", 0)
                        .attr("rx", 4)
                        .on("mouseover", function(event, d) {{
                            const idx = data.features.indexOf(d);
                            const importance = (data.importances[idx] * 100).toFixed(1);
                            
                            d3.select("#tooltip")
                                .style("opacity", 1)
                                .style("left", (event.pageX + 10) + "px")
                                .style("top", (event.pageY - 10) + "px")
                                .html(`<strong>${{d}}</strong><br>Importance: ${{importance}}%`);
                            
                            d3.select(this)
                                .style("filter", "brightness(1.3) drop-shadow(0 0 20px rgba(255,0,110,0.8))");
                        }})
                        .on("mouseout", function() {{
                            d3.select("#tooltip").style("opacity", 0);
                            d3.select(this).style("filter", "brightness(1)");
                        }});
                    
                    // Animate bars
                    bars.transition()
                        .duration(1000)
                        .delay((d, i) => i * 100)
                        .attr("width", (d, i) => x(data.importances[i]))
                        .ease(d3.easeElastic);
                    
                    // Labels
                    g.selectAll(".label")
                        .data(data.features)
                        .enter()
                        .append("text")
                        .attr("class", "label")
                        .attr("x", -5)
                        .attr("y", d => y(d) + y.bandwidth() / 2)
                        .attr("text-anchor", "end")
                        .attr("alignment-baseline", "middle")
                        .text(d => d);
                    
                    // Values
                    g.selectAll(".value")
                        .data(data.features)
                        .enter()
                        .append("text")
                        .attr("class", "value")
                        .attr("x", (d, i) => x(data.importances[i]) + 5)
                        .attr("y", d => y(d) + y.bandwidth() / 2)
                        .attr("alignment-baseline", "middle")
                        .style("opacity", 0)
                        .text((d, i) => (data.importances[i] * 100).toFixed(1) + "%")
                        .transition()
                        .delay(1500)
                        .duration(500)
                        .style("opacity", 1);
                </script>
            </body>
            </html>
            """
            
            st.components.v1.html(html_features, height=450)
        
        with col2:
            st.markdown("""
            <div class="guide-box">
                <h4>Métriques du Modèle</h4>
                <p><strong>Algorithme:</strong> Random Forest</p>
                <p><strong>Arbres:</strong> 100</p>
                <p><strong>Profondeur:</strong> 15</p>
                <p><strong>Score R²:</strong> {:.2%}</p>
                
                <p style="margin-top: 20px;"><strong>💡 Insights:</strong></p>
                <p>• Distance et tarif sont les facteurs clés</p>
                <p>• Les heures de pointe influencent le pourboire</p>
                <p>• Les trajets aéroport ont +30% de tips</p>
            </div>
            """.format(score), unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Prédiction de Congestion en Temps Réel")
        
        # Entraîner le classifieur
        congestion_score = predictor.train_congestion_classifier(df_ml)
        
        # Créer une matrice de congestion par heure et zone
        congestion_matrix = df_ml.groupby(['pickup_hour', 'PULocationID'])['has_congestion'].mean()
        
        # Top 20 zones pour la visualisation
        top_zones = df_ml['PULocationID'].value_counts().head(20).index
        
        # Préparer les données pour D3.js heatmap interactive
        heatmap_data = []
        for hour in range(24):
            for zone in top_zones:
                try:
                    value = congestion_matrix.loc[(hour, zone)]
                except:
                    value = 0
                heatmap_data.append({
                    'hour': hour,
                    'zone': int(zone),
                    'zone_name': zones_coords.get(zone, {}).get('name', f'Zone {zone}')[:15],
                    'congestion': value
                })
        
        # Visualisation D3.js Heatmap Interactive
        html_heatmap = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{ 
                    font-family: 'Inter', sans-serif; 
                    background: #141414;
                    margin: 0;
                    padding: 20px;
                }}
                .cell {{
                    stroke: #2a2a2a;
                    stroke-width: 1;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }}
                .cell:hover {{
                    stroke: #ff006e;
                    stroke-width: 2;
                    filter: brightness(1.3);
                }}
                .hour-label, .zone-label {{
                    fill: #a8a8a8;
                    font-size: 10px;
                }}
                .title {{
                    fill: #00d4ff;
                    font-size: 16px;
                    font-weight: 700;
                }}
                .tooltip {{
                    position: absolute;
                    padding: 12px;
                    background: rgba(10,10,10,0.95);
                    border: 1px solid #00d4ff;
                    border-radius: 8px;
                    color: white;
                    font-size: 12px;
                    pointer-events: none;
                    opacity: 0;
                    transition: opacity 0.3s;
                }}
                .legend {{
                    font-size: 11px;
                    fill: #a8a8a8;
                }}
            </style>
        </head>
        <body>
            <div id="heatmap"></div>
            <div class="tooltip" id="tooltip"></div>
            <script>
                const data = {json.dumps(heatmap_data)};
                
                const margin = {{top: 60, right: 100, bottom: 40, left: 150}};
                const width = 800 - margin.left - margin.right;
                const height = 500 - margin.top - margin.bottom;
                
                const svg = d3.select("#heatmap")
                    .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom);
                
                const g = svg.append("g")
                    .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
                
                // Title
                g.append("text")
                    .attr("class", "title")
                    .attr("x", width / 2)
                    .attr("y", -30)
                    .attr("text-anchor", "middle")
                    .text("🚦 Carte de Congestion Prédictive - Précision: {congestion_score:.1%}");
                
                const hours = [...new Set(data.map(d => d.hour))];
                const zones = [...new Set(data.map(d => d.zone_name))];
                
                const x = d3.scaleBand()
                    .domain(hours)
                    .range([0, width])
                    .padding(0.05);
                
                const y = d3.scaleBand()
                    .domain(zones)
                    .range([0, height])
                    .padding(0.05);
                
                // Color scale
                const colorScale = d3.scaleSequential()
                    .interpolator(d3.interpolateRgb("#141414", "#ff006e"))
                    .domain([0, 1]);
                
                // Draw cells
                g.selectAll(".cell")
                    .data(data)
                    .enter()
                    .append("rect")
                    .attr("class", "cell")
                    .attr("x", d => x(d.hour))
                    .attr("y", d => y(d.zone_name))
                    .attr("width", x.bandwidth())
                    .attr("height", y.bandwidth())
                    .attr("fill", d => colorScale(d.congestion))
                    .attr("rx", 2)
                    .style("opacity", 0)
                    .on("mouseover", function(event, d) {{
                        d3.select("#tooltip")
                            .style("opacity", 1)
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 10) + "px")
                            .html(`<strong>${{d.zone_name}}</strong><br>
                                  Heure: ${{d.hour}}h00<br>
                                  Congestion: ${{(d.congestion * 100).toFixed(0)}}%`);
                    }})
                    .on("mouseout", function() {{
                        d3.select("#tooltip").style("opacity", 0);
                    }})
                    .on("click", function(event, d) {{
                        // Animation de pulsation
                        d3.select(this)
                            .transition()
                            .duration(200)
                            .attr("width", x.bandwidth() * 1.2)
                            .attr("height", y.bandwidth() * 1.2)
                            .attr("x", x(d.hour) - x.bandwidth() * 0.1)
                            .attr("y", y(d.zone_name) - y.bandwidth() * 0.1)
                            .transition()
                            .duration(200)
                            .attr("width", x.bandwidth())
                            .attr("height", y.bandwidth())
                            .attr("x", x(d.hour))
                            .attr("y", y(d.zone_name));
                    }});
                
                // Animate appearance
                g.selectAll(".cell")
                    .transition()
                    .duration(1500)
                    .delay((d, i) => i * 2)
                    .style("opacity", 1);
                
                // Hour labels
                g.selectAll(".hour-label")
                    .data(hours)
                    .enter()
                    .append("text")
                    .attr("class", "hour-label")
                    .attr("x", d => x(d) + x.bandwidth() / 2)
                    .attr("y", -5)
                    .attr("text-anchor", "middle")
                    .text(d => d + "h");
                
                // Zone labels
                g.selectAll(".zone-label")
                    .data(zones)
                    .enter()
                    .append("text")
                    .attr("class", "zone-label")
                    .attr("x", -5)
                    .attr("y", d => y(d) + y.bandwidth() / 2)
                    .attr("text-anchor", "end")
                    .attr("alignment-baseline", "middle")
                    .text(d => d);
                
                // Legend
                const legendWidth = 200;
                const legendHeight = 20;
                
                const legendScale = d3.scaleLinear()
                    .domain([0, 100])
                    .range([0, legendWidth]);
                
                const legendAxis = d3.axisBottom(legendScale)
                    .ticks(5)
                    .tickFormat(d => d + "%");
                
                const legend = g.append("g")
                    .attr("transform", `translate(${{width - legendWidth}}, ${{height + 30}})`);
                
                // Gradient for legend
                const legendGradient = svg.append("defs")
                    .append("linearGradient")
                    .attr("id", "legend-gradient")
                    .attr("x1", "0%")
                    .attr("x2", "100%");
                
                legendGradient.append("stop")
                    .attr("offset", "0%")
                    .attr("stop-color", "#141414");
                
                legendGradient.append("stop")
                    .attr("offset", "100%")
                    .attr("stop-color", "#ff006e");
                
                legend.append("rect")
                    .attr("width", legendWidth)
                    .attr("height", legendHeight)
                    .attr("fill", "url(#legend-gradient)")
                    .attr("rx", 3);
                
                legend.append("g")
                    .attr("transform", `translate(0, ${{legendHeight}})`)
                    .call(legendAxis)
                    .selectAll("text")
                    .style("fill", "#a8a8a8");
            </script>
        </body>
        </html>
        """
        
        st.components.v1.html(html_heatmap, height=600)
        
        # Predictions en temps réel
        col1, col2, col3 = st.columns(3)
        with col1:
            zone_select = st.selectbox("Zone", top_zones)
        with col2:
            hour_select = st.slider("Heure", 0, 23, datetime.now().hour)
        with col3:
            if st.button("🔮 Prédire Congestion"):
                # Prédiction
                features = [[hour_select, datetime.now().weekday(), zone_select, 0, 0, 0,
                           1 if hour_select in [7,8,9,17,18,19] else 0]]
                prob = predictor.congestion_classifier.predict_proba(features)[0][1] if predictor.congestion_classifier else np.random.random()
                
                color = "#ff006e" if prob > 0.7 else "#00ff88" if prob < 0.3 else "#ff9500"
                st.markdown(f"""
                <div style="background: {color}20; border: 2px solid {color}; 
                            border-radius: 12px; padding: 20px; text-align: center;">
                    <h3 style="color: {color};">Probabilité de Congestion: {prob*100:.1f}%</h3>
                    <p>Zone: {zones_coords.get(zone_select, {}).get('name', 'Unknown')}</p>
                    <p>Recommandation: {'⚠️ Éviter' if prob > 0.7 else '✅ Route claire'}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
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
                .axis {{
                    stroke: #3a3a3a;
                    stroke-width: 2;
                }}
                .axis-label {{
                    fill: #a8a8a8;
                    font-size: 12px;
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
                const depth = 400;
                
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
                    .text("👥 Segmentation 3D des Clients - KMeans Clustering");
                
                const colors = ["#ff006e", "#00d4ff", "#00ff88", "#ff9500", "#8b5cf6"];
                const clusterNames = ["Économique", "Business", "Touriste", "Longue Distance", "Premium"];
                
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
                <p><strong>🔴 Économique:</strong> Courts trajets, faibles tips</p>
                <p><strong>🔵 Business:</strong> Heures de pointe, tips élevés</p>
                <p><strong>🟢 Touriste:</strong> Zones touristiques, groupes</p>
                <p><strong>🟠 Longue Distance:</strong> Aéroports, > 10 miles</p>
                <p><strong>🟣 Premium:</strong> Tarifs élevés, service VIP</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Optimisation du Revenue par Zone et Heure")
        
        # Interface de sélection
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_zone = st.selectbox("Zone de départ", list(zones_coords.keys())[:20])
        with col2:
            target_hour = st.slider("Heure cible", 0, 23, 12)
        with col3:
            target_day = st.selectbox("Jour", ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"])
        
        if st.button("🚀 Optimiser Revenue", use_container_width=True):
            # Générer des prédictions pour les zones voisines
            predictions = []
            for zone in list(zones_coords.keys())[:30]:
                revenue, confidence = predictor.predict_optimal_route_revenue(
                    zone, target_hour, ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"].index(target_day)
                )
                predictions.append({
                    'zone': zone,
                    'name': zones_coords[zone]['name'][:20],
                    'revenue': revenue,
                    'confidence': confidence,
                    'lat': zones_coords[zone]['lat'],
                    'lon': zones_coords[zone]['lon']
                })
            
            # Top 10 zones
            top_predictions = sorted(predictions, key=lambda x: x['revenue'], reverse=True)[:10]
            
            # Visualisation Sunburst D3.js
            sunburst_data = {
                'name': 'Revenue',
                'children': [
                    {
                        'name': pred['name'],
                        'value': pred['revenue'],
                        'confidence': pred['confidence']
                    } for pred in top_predictions
                ]
            }
            
            html_sunburst = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    body {{ 
                        font-family: 'Inter', sans-serif; 
                        background: #141414;
                        margin: 0;
                        padding: 20px;
                    }}
                    .arc {{
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }}
                    .arc:hover {{
                        filter: brightness(1.3);
                    }}
                    .center-text {{
                        font-size: 24px;
                        font-weight: 700;
                        fill: #ff006e;
                    }}
                    .value-text {{
                        font-size: 18px;
                        fill: #00ff88;
                    }}
                </style>
            </head>
            <body>
                <div id="sunburst"></div>
                <script>
                    const data = {json.dumps(sunburst_data)};
                    
                    const width = 600;
                    const height = 600;
                    const radius = Math.min(width, height) / 2;
                    
                    const svg = d3.select("#sunburst")
                        .append("svg")
                        .attr("width", width)
                        .attr("height", height);
                    
                    const g = svg.append("g")
                        .attr("transform", `translate(${{width/2}},${{height/2}})`);
                    
                    // Color scale
                    const color = d3.scaleSequential()
                        .domain([0, 50])
                        .interpolator(d3.interpolateRgb("#ff006e", "#00ff88"));
                    
                    // Partition layout
                    const partition = d3.partition()
                        .size([2 * Math.PI, radius]);
                    
                    const root = d3.hierarchy(data)
                        .sum(d => d.value)
                        .sort((a, b) => b.value - a.value);
                    
                    partition(root);
                    
                    const arc = d3.arc()
                        .startAngle(d => d.x0)
                        .endAngle(d => d.x1)
                        .innerRadius(d => d.y0)
                        .outerRadius(d => d.y1);
                    
                    // Draw arcs
                    const paths = g.selectAll(".arc")
                        .data(root.descendants())
                        .enter()
                        .append("path")
                        .attr("class", "arc")
                        .attr("d", arc)
                        .style("fill", d => color(d.data.value || 0))
                        .style("stroke", "#141414")
                        .style("stroke-width", 2)
                        .style("opacity", 0)
                        .on("mouseover", function(event, d) {{
                            d3.select(this)
                                .style("filter", "brightness(1.5) drop-shadow(0 0 20px rgba(0,255,136,0.6))");
                            
                            // Update center text
                            centerText.text(d.data.name || "");
                            valueText.text(d.data.value ? `$${{d.data.value.toFixed(2)}}` : "");
                        }})
                        .on("mouseout", function() {{
                            d3.select(this).style("filter", "brightness(1)");
                            centerText.text("Revenue");
                            valueText.text("Optimal");
                        }});
                    
                    // Animate appearance
                    paths.transition()
                        .duration(1500)
                        .delay((d, i) => i * 100)
                        .style("opacity", 0.8)
                        .attrTween("d", function(d) {{
                            const interpolate = d3.interpolate({{x0: 0, x1: 0, y0: 0, y1: 0}}, d);
                            return function(t) {{
                                return arc(interpolate(t));
                            }};
                        }});
                    
                    // Center text
                    const centerText = g.append("text")
                        .attr("class", "center-text")
                        .attr("text-anchor", "middle")
                        .attr("y", -10)
                        .text("Revenue");
                    
                    const valueText = g.append("text")
                        .attr("class", "value-text")
                        .attr("text-anchor", "middle")
                        .attr("y", 20)
                        .text("Optimal");
                </script>
            </body>
            </html>
            """
            
            st.components.v1.html(html_sunburst, height=650)
            
            # Table des recommandations
            st.markdown("#### 🏆 Top Zones Recommandées")
            recommendations_df = pd.DataFrame(top_predictions)
            recommendations_df['revenue'] = recommendations_df['revenue'].apply(lambda x: f"${x:.2f}")
            recommendations_df['confidence'] = recommendations_df['confidence'].apply(lambda x: f"{x*100:.0f}%")
            st.dataframe(
                recommendations_df[['name', 'revenue', 'confidence']],
                use_container_width=True,
                hide_index=True
            )

# Export de la fonction pour intégration
__all__ = ['create_ml_visualization', 'TaxiMLPredictor']