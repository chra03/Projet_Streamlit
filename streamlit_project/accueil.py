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

# ============= CUSTOM CSS - PROFESSIONAL DARK THEME (LIGHT VERSION) =============
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
        --border-color: #252525;
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
    
    /* Main Container - Reduced padding */
    .main .block-container {
        padding: 1.25rem 1.5rem 2rem;
        max-width: none;
    }
    
    /* Header Banner - More compact */
    .header-banner {
        background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-pink) 0%, var(--accent-blue) 33%, var(--accent-green) 66%, var(--accent-orange) 100%);
    }
    
    .header-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-pink) 0%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling - More compact */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
        width: 280px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 1.25rem 1rem;
    }
    
    /* Sidebar Logo Area - Smaller */
    .sidebar-logo {
        text-align: center;
        padding: 0.75rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .sidebar-logo-text {
        font-size: 1.125rem;
        font-weight: 700;
        background: var(--gradient-pink);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Filter Section Headers - Compact */
    .filter-section {
        background: var(--bg-tertiary);
        border-radius: 8px;
        padding: 0.65rem;
        margin-bottom: 0.85rem;
        border: 1px solid var(--border-color);
    }
    
    .filter-header {
        color: var(--text-primary);
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
        padding-left: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .filter-header::before {
        content: '';
        width: 2px;
        height: 0.75rem;
        background: var(--gradient-pink);
        border-radius: 1px;
    }
    
    /* KPI Cards - Slim version */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }
    
    .kpi-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        position: relative;
        transition: all 0.25s ease;
        overflow: hidden;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-pink);
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.12);
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
        transition: opacity 0.25s ease;
    }
    
    .kpi-card:hover::before {
        opacity: 1;
    }
    
    .kpi-card:nth-child(2)::before { background: var(--gradient-blue); }
    .kpi-card:nth-child(2):hover { border-color: var(--accent-blue); box-shadow: 0 6px 20px rgba(0, 212, 255, 0.12); }
    .kpi-card:nth-child(3)::before { background: var(--gradient-green); }
    .kpi-card:nth-child(3):hover { border-color: var(--accent-green); box-shadow: 0 6px 20px rgba(0, 255, 136, 0.12); }
    .kpi-card:nth-child(4)::before { background: var(--gradient-orange); }
    .kpi-card:nth-child(4):hover { border-color: var(--accent-orange); box-shadow: 0 6px 20px rgba(255, 149, 0, 0.12); }
    
    .kpi-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    
    .kpi-icon-pink { background: rgba(255, 0, 110, 0.1); color: var(--accent-pink); }
    .kpi-icon-blue { background: rgba(0, 212, 255, 0.1); color: var(--accent-blue); }
    .kpi-icon-green { background: rgba(0, 255, 136, 0.1); color: var(--accent-green); }
    .kpi-icon-orange { background: rgba(255, 149, 0, 0.1); color: var(--accent-orange); }
    
    .kpi-content {
        flex: 1;
        min-width: 0;
    }
    
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }
    
    .kpi-label {
        font-size: 0.65rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.1rem;
    }
    
    /* Chart Containers - Compact */
    .chart-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.85rem;
        margin-bottom: 1rem;
        transition: all 0.25s ease;
    }
    
    .chart-container:hover {
        border-color: #333;
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .chart-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .chart-badge {
        font-size: 0.6rem;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        background: rgba(139, 92, 246, 0.1);
        color: var(--accent-purple);
        font-weight: 600;
    }
    
    /* Buttons - Slim */
    .stButton > button {
        background: var(--gradient-pink);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.02em;
        transition: all 0.25s ease;
        width: 100%;
        box-shadow: 0 2px 8px rgba(255, 0, 110, 0.15);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 0, 110, 0.25);
    }
    
    /* Sliders - Compact */
    .stSlider > div > div {
        background: var(--bg-tertiary);
        border-radius: 6px;
        padding: 0.35rem;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: transparent;
    }
    
    div[data-baseweb="slider"] > div {
        background: linear-gradient(to right, var(--accent-pink), var(--accent-blue)) !important;
    }
    
    div[data-baseweb="slider"] > div > div {
        background: white !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Select Boxes - Compact */
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
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
        border-radius: 6px;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--gradient-pink);
        color: white;
    }
    
    /* Sections - Reduced spacing */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1.5rem 0;
    }
    
    /* Guide Box - Compact */
    .guide-box {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.85rem;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .guide-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 2px;
        height: 100%;
        background: var(--gradient-blue);
    }
    
    .guide-box h4 {
        color: var(--accent-blue);
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .guide-box p {
        color: var(--text-secondary);
        font-size: 0.7rem;
        line-height: 1.4;
        margin-bottom: 0.4rem;
    }
    
    .guide-box p strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Metrics - Compact */
    [data-testid="metric-container"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.6rem;
    }
    
    [data-testid="metric-container"] label {
        color: var(--text-secondary);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    /* Dataframes - Compact */
    .dataframe {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe thead th {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.65rem;
        letter-spacing: 0.04em;
        border-bottom: 1px solid var(--accent-pink) !important;
    }
    
    .dataframe tbody td {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
        font-size: 0.75rem;
    }
    
    .dataframe tbody tr:hover td {
        background: var(--bg-tertiary) !important;
    }
    
    /* Scrollbar - Thinner */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3a3a3a;
    }
    
    /* Tooltips - Compact */
    .tooltip {
        background: rgba(10, 10, 10, 0.95) !important;
        border: 1px solid var(--accent-pink) !important;
        border-radius: 6px;
        padding: 0.5rem !important;
        font-size: 0.75rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .kpi-card, .chart-container {
        animation: fadeIn 0.4s ease-out;
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
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.75rem;
        padding: 0.5rem 1rem;
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
    
    if st.button(" Réinitialiser", use_container_width=True):
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
        <div class="kpi-content">
            <div class="kpi-value">{len(G.nodes())}</div>
            <div class="kpi-label">Zones Actives</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-blue">🔗</div>
        <div class="kpi-content">
            <div class="kpi-value">{len(G.edges())}</div>
            <div class="kpi-label">Connexions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-green">🚖</div>
        <div class="kpi-content">
            <div class="kpi-value">{flows['trips'].sum():,.0f}</div>
            <div class="kpi-label">Total Trajets</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-orange">💰</div>
        <div class="kpi-content">
            <div class="kpi-value">${flows['total_amount'].sum()/1000:.0f}K</div>
            <div class="kpi-label">Revenu Total</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= NETWORK VISUALIZATION =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title"> Visualisation du Réseau de Taxis</h2>
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

col_graph, col_guide = st.columns([4, 1], gap="medium")

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
                height: 380px;
                background: radial-gradient(circle at center, #1a1a1a 0%, #0a0a0a 100%);
                border-radius: 10px;
                position: relative;
            }}
            .node {{
                stroke: white;
                stroke-width: 1.5px;
                cursor: pointer;
                filter: drop-shadow(0 0 3px rgba(255, 255, 255, 0.1));
                transition: all 0.25s ease;
            }}
            .node:hover {{
                stroke: #00d4ff;
                stroke-width: 2px;
                filter: drop-shadow(0 0 8px rgba(0, 212, 255, 0.5));
            }}
            .node.highlighted {{
                stroke: #ff006e;
                stroke-width: 2px;
                filter: drop-shadow(0 0 10px rgba(255, 0, 110, 0.7));
            }}
            .link {{
                stroke-opacity: 0.5;
                pointer-events: none;
            }}
            .link.highlighted {{
                stroke: #ff9500;
                stroke-width: 2px;
                stroke-opacity: 1;
                filter: drop-shadow(0 0 6px rgba(255, 149, 0, 0.4));
            }}
            .label {{
                font-size: 9px;
                font-weight: 600;
                text-anchor: middle;
                pointer-events: none;
                user-select: none;
                fill: #ffffff;
                text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
            }}
            .tooltip {{
                position: absolute;
                padding: 8px 12px;
                background: rgba(20, 20, 20, 0.95);
                color: #ffffff;
                border: 1px solid #ff006e;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 500;
                pointer-events: none;
                display: none;
                z-index: 1000;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
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
            const height = 380;
            
            const svg = d3.select('#graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            const g = svg.append('g');
            
            const simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.links)
                    .id(d => d.id)
                    .distance(80)
                    .strength(0.5))
                .force('charge', d3.forceManyBody()
                    .strength(-350)
                    .distanceMax(400))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => d.size + 8));
            
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
                .attr('opacity', 0.5);
            
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
                        .html(`<strong>${{d.label}}</strong><br>📍 ${{d.borough}}<br>🔗 ${{d.size.toFixed(0)}} conn.`)
                        .style('left', (event.pageX + 8) + 'px')
                        .style('top', (event.pageY - 20) + 'px');
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
                .attr('dy', -10)
                .text(d => d.label.length > 15 ? d.label.substring(0, 15) + '...' : d.label);
            
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

    st.components.v1.html(html_code, height=400, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

with col_guide:
    st.markdown("""
    <div class="guide-box">
        <h4>Légende</h4>
        <p>🔴 : Zones standards</p>
        <p>🟢 : Départ</p>
        <p>🔵 : Arrivée</p>
        <p>🟣 : Violet: Mixte</p>
        <p style="margin-top: 0.5rem;"><strong>Interactions:</strong></p>
        <p> Glisser /  Zoom</p>
        <p>🟠: Flux filtrés</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= SANKEY DIAGRAM =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title"> Analyse des Flux - Diagramme Sankey</h2>
    <span class="chart-badge">INTERACTIVE</span>
</div>
""", unsafe_allow_html=True)

col_sankey, col_sankey_stats = st.columns([4, 1], gap="medium")

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
        node_labels.append(f"{zone_name[:18]}")
        
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
                link_colors.append('rgba(255, 149, 0, 0.5)')
            else:
                link_colors.append('rgba(74, 74, 74, 0.25)')
        elif depart_borough and pu_borough == depart_borough:
            link_colors.append('rgba(0, 255, 136, 0.4)')
        elif arrivee_borough and do_borough == arrivee_borough:
            link_colors.append('rgba(0, 212, 255, 0.4)')
        else:
            link_colors.append('rgba(255, 0, 110, 0.35)')
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=18,
            line=dict(color="rgba(255, 255, 255, 0.08)", width=0.5),
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
        font={'size': 10, 'color': '#ffffff', 'family': 'Inter'},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        height=420,
        margin=dict(l=0, r=0, t=10, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_sankey_stats:
    total_sankey_trips = top_flows_sankey['trips'].sum()
    avg_sankey_trips = top_flows_sankey['trips'].mean()
    max_flow = top_flows_sankey.nlargest(1, 'trips').iloc[0]
    
    st.markdown(f"""
    <div class="guide-box">
        <h4>Stats Flux</h4>
        <p><strong>Total:</strong> {total_sankey_trips:,.0f}</p>
        <p><strong>Moyenne:</strong> {avg_sankey_trips:.0f}</p>
        <p><strong>Routes:</strong> {len(top_flows_sankey)}</p>
        <p style="margin-top: 0.5rem;"><strong> Top:</strong></p>
        <p>{zones_coords[max_flow['PULocationID']]['name'][:15]}</p>
        <p>→ {zones_coords[max_flow['DOLocationID']]['name'][:15]}</p>
        <p><strong>{max_flow['trips']:,.0f}</strong> trajets</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= ECONOMIC ANALYSIS =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title"> Analyse Économique Spatio-Temporelle</h2>
    <span class="chart-badge">ANALYTICS</span>
</div>
""", unsafe_allow_html=True)

df['hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
df['day_of_week'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.dayofweek
df['revenue_per_km'] = df['total_amount'] / (df['trip_distance'] + 0.01)

days_map = {0: 'Lun', 1: 'Mar', 2: 'Mer', 3: 'Jeu', 4: 'Ven', 5: 'Sam', 6: 'Dim'}
df['day_name'] = df['day_of_week'].map(days_map)

col_eco1, col_eco2, col_eco3 = st.columns(3, gap="small")

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
        line=dict(color='#ff006e', width=2),
        marker=dict(size=5, color='#ff006e'),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 110, 0.08)'
    ))
    
    fig_hourly.update_layout(
        title={'text': ' Rentabilité Horaire', 'font': {'size': 12, 'color': '#ffffff'}},
        xaxis={'title': '', 'color': '#a8a8a8', 'gridcolor': '#252525', 'zerolinecolor': '#252525', 'tickfont': {'size': 9}},
        yaxis={'title': '', 'color': '#a8a8a8', 'gridcolor': '#252525', 'zerolinecolor': '#252525', 'tickfont': {'size': 9}},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=260,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=30, r=10, t=35, b=25)
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
            title=dict(text='$/trajet', font=dict(color='#a8a8a8', size=9)),
            tickfont=dict(color='#a8a8a8', size=8),
            thickness=8,
            len=0.5
        ),
        hovertemplate='%{y} %{x}<br>$%{z:.2f}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title={'text': ' Patterns Hebdo', 'font': {'size': 12, 'color': '#ffffff'}},
        xaxis={'title': '', 'color': '#a8a8a8', 'tickfont': {'size': 8}},
        yaxis={'title': '', 'color': '#a8a8a8', 'tickfont': {'size': 9}},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=260,
        margin=dict(l=35, r=10, t=35, b=25)
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
        lambda x: zones_coords.get(x, {}).get('name', f'Zone {x}')[:16]
    )
    
    top_zones = zone_stats.nlargest(6, 'revenue_total')
    
    fig_zones = go.Figure(data=[
        go.Bar(
            x=top_zones['revenue_mean'],
            y=top_zones['zone_name'],
            orientation='h',
            marker=dict(
                color=top_zones['efficiency'],
                colorscale=[[0, '#ff006e'], [0.5, '#ff9500'], [1, '#00ff88']],
                showscale=False
            ),
            text=[f'${x:.0f}' for x in top_zones['revenue_mean']],
            textposition='outside',
            textfont=dict(color='#ffffff', size=9),
            hovertemplate='<b>%{y}</b><br>$%{x:.2f}/trajet<extra></extra>'
        )
    ])
    
    fig_zones.update_layout(
        title={'text': ' Top Zones', 'font': {'size': 12, 'color': '#ffffff'}},
        xaxis={'title': '', 'color': '#a8a8a8', 'gridcolor': '#252525', 'showticklabels': False},
        yaxis={'title': '', 'color': '#ffffff', 'tickfont': {'size': 9}},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=260,
        margin=dict(l=90, r=35, t=35, b=25)
    )
    
    st.plotly_chart(fig_zones, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============= 3D MAP =============
st.markdown("""
<div class="chart-header">
    <h2 class="chart-title"> Carte 3D Interactive</h2>
    <span class="chart-badge">3D VIEW</span>
</div>
""", unsafe_allow_html=True)

col_map3d, col_controls = st.columns([4, 1], gap="medium")

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
    
    show_top_n = st.slider(" Top zones", 10, 50, 25)
    
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
            line=dict(color='rgba(255, 0, 110, 0.25)', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=top_zones_map['lon'],
        y=top_zones_map['lat'],
        z=top_zones_map[selected_metric],
        mode='markers+text',
        marker=dict(
            size=9,
            color=top_zones_map[selected_metric],
            colorscale=color_scheme,
            showscale=True,
            colorbar=dict(
                title=dict(text=metric_choice[:12], font=dict(color='#a8a8a8', size=9)),
                tickfont=dict(color='#a8a8a8', size=8),
                thickness=8,
                len=0.6,
                x=1.02
            ),
            line=dict(color='white', width=0.5)
        ),
        text=top_zones_map['zone_name'],
        textposition='top center',
        textfont=dict(size=7, color='#ffffff'),
        hovertemplate='<b>%{text}</b><br>' +
                     f'{metric_choice}: %{{z:.2f}}<extra></extra>'
    ))
    
    camera = dict(
        eye=dict(x=1.8, y=1.8, z=1.5),
        center=dict(x=0, y=0, z=0)
    )
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='', backgroundcolor="#0a0a0a", gridcolor='#252525', showspikes=False, tickfont=dict(color='#6b6b6b', size=8)),
            yaxis=dict(title='', backgroundcolor="#0a0a0a", gridcolor='#252525', showspikes=False, tickfont=dict(color='#6b6b6b', size=8)),
            zaxis=dict(title='', backgroundcolor="#0a0a0a", gridcolor='#252525', showspikes=False, tickfont=dict(color='#6b6b6b', size=8)),
            camera=camera,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            bgcolor='#0a0a0a'
        ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#ffffff'},
        height=380,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

from taxi_ml_predictions import create_ml_visualization

# ============= PREDICTIVE INSIGHTS =============
create_ml_visualization(df, zones_coords)


# Footer
st.markdown("""
<div style="text-align: center; padding: 1rem 0; color: #6b6b6b; border-top: 1px solid #252525; margin-top: 1.5rem; font-size: 0.75rem;">
    <p>NYC Taxi Analytics Dashboard © 2024 | Powered by Streamlit & D3.js</p>
</div>
""", unsafe_allow_html=True)
