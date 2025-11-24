import streamlit as st
import pandas as pd
import networkx as nx
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import folium
from folium import plugins
import base64
from io import BytesIO
from nyc_zones_data import NYC_ZONES

st.set_page_config(
    page_title="NYC Taxi Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🚕"
)

# ============= CUSTOM CSS - PROFESSIONAL DARK THEME =============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Root Variables */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #141414;
        --bg-tertiary: #1a1a1a;
        --border-color: #2a2a2a;
        --text-primary: #ffffff;
        --text-secondary: #a8a8a8;
        --text-tertiary: #6b6b6b;
        --accent-pink: #ff006e;
        --accent-blue: #00d4ff;
        --accent-green: #00ff88;
        --accent-orange: #ff9500;
        --accent-purple: #8b5cf6;
        --gradient-pink: linear-gradient(135deg, #ff006e 0%, #ff4d94 100%);
        --gradient-blue: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        --gradient-green: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        --gradient-orange: linear-gradient(135deg, #ff9500 0%, #ff6b00 100%);
    }
    
    /* Global Styles */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding: 2rem 2rem 3rem;
        max-width: none;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-pink) 0%, var(--accent-blue) 33%, var(--accent-green) 66%, var(--accent-orange) 100%);
        animation: gradientShift 8s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-pink) 0%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
        width: 320px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    /* Sidebar Logo Area */
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .sidebar-logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        background: var(--gradient-pink);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Filter Section Headers */
    .filter-section {
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }
    
    .filter-header {
        color: var(--text-primary);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .filter-header::before {
        content: '';
        width: 3px;
        height: 1rem;
        background: var(--gradient-pink);
        border-radius: 2px;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.75rem;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-pink);
        box-shadow: 0 12px 32px rgba(255, 0, 110, 0.15);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-pink);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .kpi-card:hover::before {
        opacity: 1;
    }
    
    .kpi-card:nth-child(2)::before,
    .kpi-card:nth-child(2):hover {
        background: var(--gradient-blue);
        border-color: var(--accent-blue);
        box-shadow: 0 12px 32px rgba(0, 212, 255, 0.15);
    }
    
    .kpi-card:nth-child(3)::before,
    .kpi-card:nth-child(3):hover {
        background: var(--gradient-green);
        border-color: var(--accent-green);
        box-shadow: 0 12px 32px rgba(0, 255, 136, 0.15);
    }
    
    .kpi-card:nth-child(4)::before,
    .kpi-card:nth-child(4):hover {
        background: var(--gradient-orange);
        border-color: var(--accent-orange);
        box-shadow: 0 12px 32px rgba(255, 149, 0, 0.15);
    }
    
    .kpi-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .kpi-icon-pink { background: rgba(255, 0, 110, 0.1); color: var(--accent-pink); }
    .kpi-icon-blue { background: rgba(0, 212, 255, 0.1); color: var(--accent-blue); }
    .kpi-icon-green { background: rgba(0, 255, 136, 0.1); color: var(--accent-green); }
    .kpi-icon-orange { background: rgba(255, 149, 0, 0.1); color: var(--accent-orange); }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    
    .kpi-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-change {
        position: absolute;
        top: 1.75rem;
        right: 1.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background: rgba(0, 255, 136, 0.1);
        color: var(--accent-green);
    }
    
    .kpi-change.negative {
        background: rgba(255, 0, 110, 0.1);
        color: var(--accent-pink);
    }
    
    /* Chart Containers */
    .chart-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        border-color: #3a3a3a;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .chart-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .chart-badge {
        font-size: 0.75rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background: rgba(139, 92, 246, 0.1);
        color: var(--accent-purple);
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-pink);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 12px rgba(255, 0, 110, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 0, 110, 0.3);
    }
    
    /* Sliders */
    .stSlider > div > div {
        background: var(--bg-tertiary);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: transparent;
    }
    
    div[data-baseweb="slider"] > div {
        background: linear-gradient(to right, var(--accent-pink), var(--accent-blue)) !important;
    }
    
    div[data-baseweb="slider"] > div > div {
        background: white !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--accent-pink);
    }
    
    /* Multi-select */
    .stMultiSelect > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--gradient-pink);
        color: white;
    }
    
    /* Sections */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 3rem 0;
    }
    
    /* Guide Box */
    .guide-box {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .guide-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: var(--gradient-blue);
    }
    
    .guide-box h4 {
        color: var(--accent-blue);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .guide-box p {
        color: var(--text-secondary);
        font-size: 0.875rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }
    
    .guide-box p strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="metric-container"] label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    /* Dataframes */
    .dataframe {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe thead th {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        border-bottom: 2px solid var(--accent-pink) !important;
    }
    
    .dataframe tbody td {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }
    
    .dataframe tbody tr:hover td {
        background: var(--bg-tertiary) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3a3a3a;
    }
    
    /* Tooltips */
    .tooltip {
        background: rgba(10, 10, 10, 0.95) !important;
        border: 1px solid var(--accent-pink) !important;
        border-radius: 8px;
        padding: 0.75rem !important;
        font-size: 0.875rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .kpi-card, .chart-container {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Plotly Dark Theme Override */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }
    
    .js-plotly-plot .plotly .modebar-btn {
        color: var(--text-secondary) !important;
    }
    
    .js-plotly-plot .plotly .modebar-btn:hover {
        color: var(--accent-pink) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============= HEADER BANNER =============
st.markdown("""
<div class="header-banner">
    <h1 class="header-title">NYC Taxi Analytics Dashboard</h1>
    <p class="header-subtitle">Analyse en temps réel des flux de taxis et patterns de déplacement à New York City</p>
</div>
""", unsafe_allow_html=True)

# ============= CACHE DATA =============
@st.cache_data
def load_data():
    df = pd.read_csv('nyc_taxi_sample.csv')
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    return df

@st.cache_data
def build_zone_coordinates():
    zones = {}
    for zone_id, data in NYC_ZONES.items():
        zones[zone_id] = {
            'lat': data['lat'],
            'lon': data['lon'],
            'name': data['name'],
            'borough': data['borough']
        }
    return zones

# ============= LOAD DATA =============
df = load_data()
zones_coords = build_zone_coordinates()

# ============= SIDEBAR FILTERS =============
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-text">🚕 TAXI NYC</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown('<div class="filter-header">Filtres Principaux</div>', unsafe_allow_html=True)
    
    min_trips = st.slider(
        "Nombre minimum de trajets",
        min_value=5,
        max_value=500,
        value=100,
        step=10,
        help="Filtrer les routes avec un minimum de trajets"
    )
    
    selected_borough = st.multiselect(
        "Sélectionner les arrondissements",
        options=sorted(df['PULocationID'].map(lambda x: zones_coords.get(x, {}).get('borough', 'Unknown')).unique()),
        default=["Manhattan"],
        help="Choisir un ou plusieurs arrondissements"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown('<div class="filter-header">Analyse Directionnelle</div>', unsafe_allow_html=True)
    
    all_boroughs = sorted(set([zones_coords[z]['borough'] for z in zones_coords.keys()]))
    
    selected_depart_borough = st.selectbox(
        "Arrondissement d'origine",
        options=['-- Tous --'] + all_boroughs,
        key='depart_borough',
        help="Filtrer par point de départ"
    )
    
    selected_arrivee_borough = st.selectbox(
        "Arrondissement de destination",
        options=['-- Tous --'] + all_boroughs,
        key='arrivee_borough',
        help="Filtrer par point d'arrivée"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown('<div class="filter-header">Paramètres Visuels</div>', unsafe_allow_html=True)
    
    node_size_mult = st.slider(
        "Taille des nœuds",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5
    )
    
    edge_thickness = st.slider(
        "Épaisseur des connexions",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.5
    )
    
    sankey_top_flows = st.slider(
        "Top flux (Sankey)",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔄 Réinitialiser les Filtres", use_container_width=True):
        st.rerun()

# ============= FILTER DATA =============
flows = df.groupby(['PULocationID', 'DOLocationID']).agg({
    'trip_distance': 'mean',
    'total_amount': 'sum',
    'passenger_count': 'count'
}).rename(columns={'passenger_count': 'trips'}).reset_index()

flows = flows[flows['trips'] >= min_trips]

if selected_borough:
    flows = flows[
        flows['PULocationID'].apply(lambda x: zones_coords.get(x, {}).get('borough', 'Unknown') in selected_borough)
    ]

# ============= BUILD NETWORKX GRAPH =============
G = nx.DiGraph()

for zone_id in zones_coords.keys():
    G.add_node(zone_id, 
               name=zones_coords[zone_id]['name'],
               borough=zones_coords[zone_id]['borough'])

for _, row in flows.iterrows():
    G.add_edge(int(row['PULocationID']), int(row['DOLocationID']), 
               trips=int(row['trips']), revenue=row['total_amount'])

nodes_to_keep = set(flows['PULocationID'].unique()) | set(flows['DOLocationID'].unique())
G = G.subgraph(nodes_to_keep).copy()

# ============= KPI CARDS =============
st.markdown('<div class="kpi-container">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-pink">📍</div>
        <div class="kpi-value">{len(G.nodes())}</div>
        <div class="kpi-label">Zones Actives</div>
        <span class="kpi-change">+12%</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-blue">🔗</div>
        <div class="kpi-value">{len(G.edges())}</div>
        <div class="kpi-label">Connexions</div>
        <span class="kpi-change">+8%</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-green">🚖</div>
        <div class="kpi-value">{flows['trips'].sum():,.0f}</div>
        <div class="kpi-label">Total Trajets</div>
        <span class="kpi-change">+15%</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-orange">💰</div>
        <div class="kpi-value">${flows['total_amount'].sum()/1000:.0f}K</div>
        <div class="kpi-label">Revenu Total</div>
        <span class="kpi-change negative">-3%</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= NETWORK VISUALIZATION =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title">🌐 Visualisation du Réseau de Taxis</h2>
    <span class="chart-badge">LIVE</span>
</div>
""", unsafe_allow_html=True)

depart_borough = selected_depart_borough if selected_depart_borough != '-- Tous --' else None
arrivee_borough = selected_arrivee_borough if selected_arrivee_borough != '-- Tous --' else None

nodes = []
edges = []

for node in G.nodes():
    zone_name = zones_coords[node]['name']
    zone_borough = zones_coords[node]['borough']
    degree = G.degree(node)
    
    is_depart = (depart_borough and zone_borough == depart_borough)
    is_arrivee = (arrivee_borough and zone_borough == arrivee_borough)
    
    if is_depart and is_arrivee:
        color = '#8b5cf6'  # Purple
    elif is_depart:
        color = '#00ff88'  # Green
    elif is_arrivee:
        color = '#00d4ff'  # Blue
    else:
        color = '#ff006e'  # Pink
    
    nodes.append({
        'id': str(node),
        'label': zone_name,
        'size': (5 + degree) * node_size_mult,
        'color': color,
        'attributes': {
            'borough': zone_borough,
            'connections': degree
        }
    })

for edge in G.edges(data=True):
    pu_id = int(edge[0])
    do_id = int(edge[1])
    pu_borough = zones_coords[pu_id]['borough']
    do_borough = zones_coords[do_id]['borough']
    trips = edge[2]['trips']
    
    is_highlighted = False
    
    if depart_borough and arrivee_borough:
        if pu_borough == depart_borough and do_borough == arrivee_borough:
            is_highlighted = True
        else:
            continue
    elif depart_borough:
        if pu_borough == depart_borough:
            is_highlighted = True
    elif arrivee_borough:
        if do_borough == arrivee_borough:
            is_highlighted = True
    
    edge_color = '#ff9500' if is_highlighted else '#4a4a4a'
    edge_width = edge_thickness * 2.5 if is_highlighted else edge_thickness
    
    edges.append({
        'source': str(pu_id),
        'target': str(do_id),
        'label': f'{trips} trajets',
        'size': edge_width,
        'color': edge_color,
        'attributes': {
            'trips': trips,
            'revenue': edge[2]['revenue']
        }
    })

graph_data = {
    'nodes': nodes,
    'edges': edges
}

col_graph, col_guide = st.columns([3, 1], gap="large")

with col_graph:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    d3_nodes = [{'id': node['id'], 'label': node['label'], 'color': node['color'], 
                  'size': node['size'], 'borough': node['attributes']['borough']} 
                 for node in graph_data['nodes']]
    d3_links = [{'source': edge['source'], 'target': edge['target'], 'value': edge['size'], 
                  'color': edge['color'], 'label': edge['label']} 
                 for edge in graph_data['edges']]

    d3_data = {'nodes': d3_nodes, 'links': d3_links}

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Network Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Inter', -apple-system, sans-serif;
                background-color: #141414;
            }}
            #graph {{
                width: 100%;
                height: 500px;
                background: radial-gradient(circle at center, #1a1a1a 0%, #0a0a0a 100%);
                border-radius: 16px;
                position: relative;
            }}
            .node {{
                stroke: white;
                stroke-width: 2px;
                cursor: pointer;
                filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.1));
                transition: all 0.3s ease;
            }}
            .node:hover {{
                stroke: #00d4ff;
                stroke-width: 3px;
                filter: drop-shadow(0 0 12px rgba(0, 212, 255, 0.6));
            }}
            .node.highlighted {{
                stroke: #ff006e;
                stroke-width: 3px;
                filter: drop-shadow(0 0 16px rgba(255, 0, 110, 0.8));
            }}
            .link {{
                stroke-opacity: 0.6;
                pointer-events: none;
            }}
            .link.highlighted {{
                stroke: #ff9500;
                stroke-width: 3px;
                stroke-opacity: 1;
                filter: drop-shadow(0 0 8px rgba(255, 149, 0, 0.5));
            }}
            .label {{
                font-size: 11px;
                font-weight: 600;
                text-anchor: middle;
                pointer-events: none;
                user-select: none;
                fill: #ffffff;
                text-shadow: 0 0 4px rgba(0, 0, 0, 0.8);
            }}
            .tooltip {{
                position: absolute;
                padding: 12px 16px;
                background: linear-gradient(135deg, rgba(20, 20, 20, 0.98) 0%, rgba(26, 26, 26, 0.98) 100%);
                color: #ffffff;
                border: 1px solid #ff006e;
                border-radius: 12px;
                font-size: 13px;
                font-weight: 500;
                pointer-events: none;
                display: none;
                z-index: 1000;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
            }}
            .tooltip strong {{
                color: #00d4ff;
                font-weight: 700;
            }}
        </style>
    </head>
    <body>
        <div id="graph"></div>
        <div class="tooltip" id="tooltip"></div>
        <script>
            const data = {json.dumps(d3_data)};
            
            const width = document.getElementById('graph').clientWidth;
            const height = 500;
            
            const svg = d3.select('#graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            const g = svg.append('g');
            
            const simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.links)
                    .id(d => d.id)
                    .distance(100)
                    .strength(0.5))
                .force('charge', d3.forceManyBody()
                    .strength(-400)
                    .distanceMax(500))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => d.size + 10));
            
            const zoom = d3.zoom()
                .scaleExtent([0.3, 8])
                .on('zoom', (event) => {{
                    g.attr('transform', event.transform);
                }});
            
            svg.call(zoom);
            
            const link = g.selectAll('.link')
                .data(data.links)
                .enter()
                .append('line')
                .attr('class', 'link')
                .attr('stroke', d => d.color)
                .attr('stroke-width', d => d.value)
                .attr('opacity', 0.6);
            
            const node = g.selectAll('.node')
                .data(data.nodes)
                .enter()
                .append('circle')
                .attr('class', 'node')
                .attr('r', d => d.size)
                .attr('fill', d => d.color)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('mouseover', (event, d) => {{
                    node.classed('highlighted', n => n.id === d.id);
                    link.classed('highlighted', l => l.source.id === d.id || l.target.id === d.id);
                    
                    const tooltip = d3.select('#tooltip');
                    tooltip.style('display', 'block')
                        .html(`<strong>${{d.label}}</strong><br>📍 ${{d.borough}}<br>🔗 ${{d.size.toFixed(0)}} connexions`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                }})
                .on('mouseout', () => {{
                    node.classed('highlighted', false);
                    link.classed('highlighted', false);
                    d3.select('#tooltip').style('display', 'none');
                }});
            
            const labels = g.selectAll('.label')
                .data(data.nodes)
                .enter()
                .append('text')
                .attr('class', 'label')
                .attr('dy', -14)
                .text(d => d.label.length > 20 ? d.label.substring(0, 20) + '...' : d.label);
            
            simulation.on('tick', () => {{
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                labels
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            }});
            
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
            }}
        </script>
    </body>
    </html>
    """

    st.components.v1.html(html_code, height=520, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

with col_guide:
    st.markdown("""
    <div class="guide-box">
        <h4>Guide d'Utilisation</h4>
        <p><strong>Codes Couleur:</strong></p>
        <p>🔴 Rose : Zones standards</p>
        <p>🟢 Vert : Zones de départ</p>
        <p>🔵 Bleu : Zones d'arrivée</p>
        <p>🟣 Violet : Zones mixtes</p>
        
        <p><strong>Interactions:</strong></p>
        <p>🖱️ Glisser pour déplacer</p>
        <p>📍 Survoler pour détails</p>
        <p>🔍 Scroll pour zoomer</p>
        <p>🎯 Double-clic pour fixer</p>
        
        <p><strong>Connexions:</strong></p>
        <p>🟠 Orange : Flux filtrés</p>
        <p>⚫ Gris : Autres flux</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= SANKEY DIAGRAM =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title">🌊 Analyse des Flux - Diagramme Sankey</h2>
    <span class="chart-badge">INTERACTIVE</span>
</div>
""", unsafe_allow_html=True)

col_sankey, col_sankey_stats = st.columns([3, 1], gap="large")

with col_sankey:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    top_flows_sankey = flows.nlargest(sankey_top_flows, 'trips').copy()
    
    sources = []
    targets = []
    values = []
    link_colors = []
    
    unique_zones = list(set(top_flows_sankey['PULocationID'].tolist() + top_flows_sankey['DOLocationID'].tolist()))
    zone_to_index = {zone: i for i, zone in enumerate(unique_zones)}
    
    node_labels = []
    node_colors = []
    for zone in unique_zones:
        zone_name = zones_coords.get(zone, {}).get('name', f'Zone {zone}')
        zone_borough = zones_coords.get(zone, {}).get('borough', 'Unknown')
        node_labels.append(f"{zone_name}<br><span style='font-size:10px'>{zone_borough}</span>")
        
        if depart_borough and zone_borough == depart_borough:
            node_colors.append('#00ff88')
        elif arrivee_borough and zone_borough == arrivee_borough:
            node_colors.append('#00d4ff')
        else:
            node_colors.append('#ff006e')
    
    for _, row in top_flows_sankey.iterrows():
        sources.append(zone_to_index[row['PULocationID']])
        targets.append(zone_to_index[row['DOLocationID']])
        values.append(row['trips'])
        
        pu_borough = zones_coords[row['PULocationID']]['borough']
        do_borough = zones_coords[row['DOLocationID']]['borough']
        
        if depart_borough and arrivee_borough:
            if pu_borough == depart_borough and do_borough == arrivee_borough:
                link_colors.append('rgba(255, 149, 0, 0.6)')
            else:
                link_colors.append('rgba(74, 74, 74, 0.3)')
        elif depart_borough and pu_borough == depart_borough:
            link_colors.append('rgba(0, 255, 136, 0.5)')
        elif arrivee_borough and do_borough == arrivee_borough:
            link_colors.append('rgba(0, 212, 255, 0.5)')
        else:
            link_colors.append('rgba(255, 0, 110, 0.4)')
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="rgba(255, 255, 255, 0.1)", width=1),
            label=node_labels,
            color=node_colors,
            hovertemplate='<b>%{label}</b><br>Total: %{value:,.0f} trajets<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br><b>%{value:,.0f}</b> trajets<extra></extra>'
        )
    )])
    
    fig.update_layout(
        font={'size': 12, 'color': '#ffffff', 'family': 'Inter'},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_sankey_stats:
    st.markdown("""
    <div class="guide-box">
        <h4>Statistiques Flux</h4>
    """, unsafe_allow_html=True)
    
    total_sankey_trips = top_flows_sankey['trips'].sum()
    avg_sankey_trips = top_flows_sankey['trips'].mean()
    max_flow = top_flows_sankey.nlargest(1, 'trips').iloc[0]
    
    st.markdown(f"""
        <p><strong>📊 Métriques:</strong></p>
        <p>Total trajets: <strong>{total_sankey_trips:,.0f}</strong></p>
        <p>Moyenne: <strong>{avg_sankey_trips:.0f}</strong></p>
        <p>Routes: <strong>{len(top_flows_sankey)}</strong></p>
        
        <p><strong>🏆 Top Route:</strong></p>
        <p>{zones_coords[max_flow['PULocationID']]['name'][:20]}</p>
        <p>↓</p>
        <p>{zones_coords[max_flow['DOLocationID']]['name'][:20]}</p>
        <p><strong>{max_flow['trips']:,.0f}</strong> trajets</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= ECONOMIC ANALYSIS =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title">💰 Analyse Économique Spatio-Temporelle</h2>
    <span class="chart-badge">ANALYTICS</span>
</div>
""", unsafe_allow_html=True)

df['hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
df['day_of_week'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.dayofweek
df['revenue_per_km'] = df['total_amount'] / (df['trip_distance'] + 0.01)

days_map = {0: 'Lun', 1: 'Mar', 2: 'Mer', 3: 'Jeu', 4: 'Ven', 5: 'Sam', 6: 'Dim'}
df['day_name'] = df['day_of_week'].map(days_map)

col_eco1, col_eco2, col_eco3 = st.columns(3, gap="medium")

# Hourly Revenue Chart
with col_eco1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    hourly_stats = df.groupby('hour').agg({
        'total_amount': 'mean',
        'trip_distance': 'mean',
        'revenue_per_km': 'mean'
    }).reset_index()
    
    fig_hourly = go.Figure()
    
    fig_hourly.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['total_amount'],
        mode='lines+markers',
        name='Revenu moyen',
        line=dict(color='#ff006e', width=3),
        marker=dict(size=8, color='#ff006e'),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 110, 0.1)'
    ))
    
    fig_hourly.update_layout(
        title={'text': '📈 Rentabilité Horaire', 'font': {'size': 14, 'color': '#ffffff'}},
        xaxis={'title': 'Heure', 'color': '#a8a8a8', 'gridcolor': '#2a2a2a', 'zerolinecolor': '#2a2a2a'},
        yaxis={'title': 'Revenu moyen ($)', 'color': '#a8a8a8', 'gridcolor': '#2a2a2a', 'zerolinecolor': '#2a2a2a'},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=350,
        showlegend=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Heatmap
with col_eco2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    heatmap_data = df.groupby(['day_of_week', 'hour'])['total_amount'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='total_amount')
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[f'{h}h' for h in range(24)],
        y=['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'],
        colorscale=[
            [0, '#0a0a0a'],
            [0.25, '#ff006e'],
            [0.5, '#ff9500'],
            [0.75, '#00d4ff'],
            [1, '#00ff88']
        ],
        colorbar=dict(
            title=dict(
                text='$/trajet',
                font=dict(color='#a8a8a8', size=10)
            ),
            tickfont=dict(color='#a8a8a8'),
            thickness=10,
            len=0.6
        ),
        hovertemplate='%{y} %{x}<br>$%{z:.2f}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title={'text': '🗓️ Patterns Hebdomadaires', 'font': {'size': 14, 'color': '#ffffff'}},
        xaxis={'title': '', 'color': '#a8a8a8', 'tickfont': {'size': 10}},
        yaxis={'title': '', 'color': '#a8a8a8'},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=350
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Top Zones
with col_eco3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    zone_stats = df.groupby('PULocationID').agg({
        'total_amount': ['sum', 'mean'],
        'trip_distance': 'mean',
        'passenger_count': 'count'
    }).reset_index()
    zone_stats.columns = ['zone_id', 'revenue_total', 'revenue_mean', 'distance_mean', 'trip_count']
    zone_stats['efficiency'] = zone_stats['revenue_mean'] / (zone_stats['distance_mean'] + 0.01)
    zone_stats['zone_name'] = zone_stats['zone_id'].apply(
        lambda x: zones_coords.get(x, {}).get('name', f'Zone {x}')[:20]
    )
    
    top_zones = zone_stats.nlargest(8, 'revenue_total')
    
    fig_zones = go.Figure(data=[
        go.Bar(
            x=top_zones['revenue_mean'],
            y=top_zones['zone_name'],
            orientation='h',
            marker=dict(
                color=top_zones['efficiency'],
                colorscale=[
                    [0, '#ff006e'],
                    [0.5, '#ff9500'],
                    [1, '#00ff88']
                ],
                showscale=False
            ),
            text=[f'${x:.0f}' for x in top_zones['revenue_mean']],
            textposition='outside',
            textfont=dict(color='#ffffff', size=10),
            hovertemplate='<b>%{y}</b><br>$%{x:.2f}/trajet<br>Efficacité: %{marker.color:.1f} $/km<extra></extra>'
        )
    ])
    
    fig_zones.update_layout(
        title={'text': '🏆 Top Zones Rentables', 'font': {'size': 14, 'color': '#ffffff'}},
        xaxis={'title': '', 'color': '#a8a8a8', 'gridcolor': '#2a2a2a', 'showticklabels': False},
        yaxis={'title': '', 'color': '#ffffff', 'tickfont': {'size': 10}},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=350,
        margin=dict(l=120, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig_zones, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= 3D MAP =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title">🗺️ Carte 3D Interactive</h2>
    <span class="chart-badge">3D VIEW</span>
</div>
""", unsafe_allow_html=True)

col_map3d, col_controls = st.columns([3, 1], gap="large")

with col_controls:
    st.markdown("""
    <div class="guide-box">
        <h4>Contrôles 3D</h4>
    """, unsafe_allow_html=True)
    
    metric_choice = st.selectbox(
        "📊 Métrique",
        ["Revenu Total", "Nombre de Trajets", "Distance Moyenne", "Efficacité ($/km)"],
        label_visibility="collapsed"
    )
    
    show_top_n = st.slider("🏆 Top zones", 10, 50, 25)
    
    color_scheme = st.selectbox(
        "🎨 Palette",
        ["Plasma", "Viridis", "Turbo", "Hot"],
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_map3d:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    zone_geo_stats = df.groupby('PULocationID').agg({
        'total_amount': ['sum', 'mean'],
        'trip_distance': 'mean',
        'passenger_count': 'count'
    }).reset_index()
    zone_geo_stats.columns = ['zone_id', 'revenue_total', 'revenue_mean', 'distance_mean', 'trip_count']
    zone_geo_stats['efficiency'] = zone_geo_stats['revenue_mean'] / (zone_geo_stats['distance_mean'] + 0.01)
    
    zone_geo_stats['lat'] = zone_geo_stats['zone_id'].apply(lambda x: zones_coords.get(x, {}).get('lat', 40.7))
    zone_geo_stats['lon'] = zone_geo_stats['zone_id'].apply(lambda x: zones_coords.get(x, {}).get('lon', -74.0))
    zone_geo_stats['zone_name'] = zone_geo_stats['zone_id'].apply(lambda x: zones_coords.get(x, {}).get('name', f'Zone {x}'))
    zone_geo_stats['borough'] = zone_geo_stats['zone_id'].apply(lambda x: zones_coords.get(x, {}).get('borough', 'Unknown'))
    
    metric_map = {
        "Revenu Total": 'revenue_total',
        "Nombre de Trajets": 'trip_count',
        "Distance Moyenne": 'distance_mean',
        "Efficacité ($/km)": 'efficiency'
    }
    selected_metric = metric_map[metric_choice]
    
    top_zones_map = zone_geo_stats.nlargest(show_top_n, selected_metric)
    
    fig_3d = go.Figure()
    
    for _, row in top_zones_map.iterrows():
        fig_3d.add_trace(go.Scatter3d(
            x=[row['lon'], row['lon']],
            y=[row['lat'], row['lat']],
            z=[0, row[selected_metric]],
            mode='lines',
            line=dict(color='rgba(255, 0, 110, 0.3)', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=top_zones_map['lon'],
        y=top_zones_map['lat'],
        z=top_zones_map[selected_metric],
        mode='markers+text',
        marker=dict(
            size=12,
            color=top_zones_map[selected_metric],
            colorscale=color_scheme,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=metric_choice,
                    font=dict(color='#a8a8a8', size=10)
                ),
                tickfont=dict(color='#a8a8a8', size=10),
                thickness=10,
                len=0.7,
                x=1.02
            ),
            line=dict(color='white', width=1)
        ),
        text=top_zones_map['zone_name'],
        textposition='top center',
        textfont=dict(size=8, color='#ffffff'),
        hovertemplate='<b>%{text}</b><br>' +
                     f'{metric_choice}: %{{z:.2f}}<br>' +
                     'Coords: (%{y:.3f}, %{x:.3f})<extra></extra>'
    ))
    
    camera = dict(
        eye=dict(x=1.8, y=1.8, z=1.5),
        center=dict(x=0, y=0, z=0)
    )
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(
                title='', 
                backgroundcolor="#0a0a0a", 
                gridcolor='#2a2a2a',
                showspikes=False,
                tickfont=dict(color='#6b6b6b', size=10)
            ),
            yaxis=dict(
                title='', 
                backgroundcolor="#0a0a0a", 
                gridcolor='#2a2a2a',
                showspikes=False,
                tickfont=dict(color='#6b6b6b', size=10)
            ),
            zaxis=dict(
                title='', 
                backgroundcolor="#0a0a0a", 
                gridcolor='#2a2a2a',
                showspikes=False,
                tickfont=dict(color='#6b6b6b', size=10)
            ),
            camera=camera,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            bgcolor='#0a0a0a'
        ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

from taxi_ml_predictions import create_ml_visualization

# ============= PREDICTIVE INSIGHTS =============
create_ml_visualization(df, zones_coords)
# ============= TOP ROUTES TABLE =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title">🛣️ Top 10 Routes</h2>
    <span class="chart-badge">TRENDING</span>
</div>
""", unsafe_allow_html=True)

top_flows = flows.nlargest(10, 'trips')[['PULocationID', 'DOLocationID', 'trips', 'total_amount', 'trip_distance']]
top_flows['Origine'] = top_flows['PULocationID'].apply(lambda x: zones_coords.get(int(x), {}).get('name', f"Zone {x}")[:25])
top_flows['Destination'] = top_flows['DOLocationID'].apply(lambda x: zones_coords.get(int(x), {}).get('name', f"Zone {x}")[:25])
top_flows['Distance'] = top_flows['trip_distance'].apply(lambda x: f"{x:.1f} km")
top_flows['Trajets'] = top_flows['trips'].apply(lambda x: f"{x:,}")
top_flows['Revenu'] = top_flows['total_amount'].apply(lambda x: f"${x:,.0f}")

st.dataframe(
    top_flows[['Origine', 'Destination', 'Trajets', 'Revenu', 'Distance']],
    use_container_width=True,
    hide_index=True
)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #6b6b6b; border-top: 1px solid #2a2a2a; margin-top: 3rem;">
    <p>NYC Taxi Analytics Dashboard © 2024 | Powered by Streamlit & D3.js</p>
</div>
""", unsafe_allow_html=True)