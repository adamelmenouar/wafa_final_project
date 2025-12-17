import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Wafa Risk Monitor Pro",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# CSS Custom pour le look "Bloomberg / PowerBI"
st.markdown("""
<style>
    .stApp {background-color: #f0f2f6;}
    div.stMetric {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif;}
</style>
""", unsafe_allow_html=True)

# --- 2. FONCTIONS FINANCIÈRES AVANCÉES ---

def calculate_rsi(data, window=14):
    """Calcul du Relative Strength Index (RSI)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20):
    """Calcul des Bandes de Bollinger"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return upper, lower

def calculate_max_drawdown(series):
    """Calcul de la perte maximale historique"""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def monte_carlo_simulation(last_price, mu, sigma, days=252, simulations=100):
    """Simulation de Monte Carlo pour la projection de prix"""
    simulation_df = pd.DataFrame()
    dt = 1
    for x in range(simulations):
        price_series = [last_price]
        price = last_price
        for _ in range(days):
            shock = np.random.normal(0, 1)
            price = price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * shock * np.sqrt(dt))
            price_series.append(price)
        simulation_df[f'Sim_{x}'] = price_series
    return simulation_df

# --- 3. DATA LOAD & PREP ---
@st.cache_data
def load_data():
    file_path = "donnees_bourse_pro.csv"
    dates = pd.date_range(start="2022-01-01", end=pd.Timestamp.today(), freq='B')
    
    if not os.path.exists(file_path):
        data = pd.DataFrame(index=dates)
        np.random.seed(42)
        # Simulation plus réaliste (Mouvement Brownien Géométrique)
        configs = [('IAM.MA', 100, 0.15), ('ATW.MA', 450, 0.22), ('BCP.MA', 260, 0.18)]
        for ticker, start, vol in configs:
            dt = 1/252
            mu = 0.05 # Tendance haussière légère
            prices = [start]
            for _ in range(len(dates)-1):
                shock = np.random.normal(0, 1)
                new_price = prices[-1] * np.exp((mu - 0.5 * vol**2) * dt + vol * shock * np.sqrt(dt))
                prices.append(new_price)
            data[ticker] = prices
        data.to_csv(file_path)
        return data
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

df = load_data()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ec/Attijariwafa_bank_logo.svg", width=150)
    st.markdown("### ⚙️ Configuration")
    selected_asset = st.selectbox("Actif Financier", df.columns)
    
    st.markdown("### 🧠 Modèle IA")
    contamination = st.slider("Sensibilité (Outliers)", 0.01, 0.10, 0.03)
    
    st.markdown("---")
    st.info("Dashboard v2.0 - Developed for Wafa Gestion Challenge")

# --- 5. CALCULS BACKEND ---
asset_data = df[[selected_asset]].copy()
asset_data['Returns'] = asset_data[selected_asset].pct_change().fillna(0)

# Indicateurs Techniques
asset_data['RSI'] = calculate_rsi(asset_data[selected_asset])
asset_data['BB_Upper'], asset_data['BB_Lower'] = calculate_bollinger_bands(asset_data[selected_asset])
asset_data['MA50'] = asset_data[selected_asset].rolling(50).mean()

# IA : Isolation Forest
model = IsolationForest(contamination=contamination, random_state=42)
asset_data['Anomaly_Score'] = model.fit_predict(asset_data['Returns'].values.reshape(-1, 1))
anomalies = asset_data[asset_data['Anomaly_Score'] == -1]

# Métriques Financières
last_price = asset_data[selected_asset].iloc[-1]
returns_mean = asset_data['Returns'].mean()
volatility = asset_data['Returns'].std() * np.sqrt(252) # Annualisée
max_dd = calculate_max_drawdown(asset_data[selected_asset])
var_95 = np.percentile(asset_data['Returns'], 5) # VaR historique 95%

# --- 6. INTERFACE ---
st.title(f"🦅 Risk Monitor Pro : {selected_asset}")

# ONGLETS
tab1, tab2, tab3 = st.tabs(["📈 Market Cockpit", "📊 Risk Analytics", "🔮 Monte Carlo Lab"])

# --- ONGLET 1 : MARKET COCKPIT ---
with tab1:
    # KPIs Top Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix Actuel", f"{last_price:.2f} MAD", f"{asset_data['Returns'].iloc[-1]:.2%}")
    c2.metric("Volatilité (An)", f"{volatility:.2%}", delta_color="inverse")
    c3.metric("RSI (14j)", f"{asset_data['RSI'].iloc[-1]:.1f}", delta="Surachat" if asset_data['RSI'].iloc[-1] > 70 else ("Survente" if asset_data['RSI'].iloc[-1] < 30 else "Neutre"))
    c4.metric("Alertes IA", f"{len(anomalies)}", delta_color="inverse")

    st.markdown("---")

    # GRAPHIQUE COMBINÉ (Prix + Bandes Bollinger + Anomalies)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Row 1 : Prix & Bandes
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['BB_Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['BB_Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='Bandes Bollinger'), row=1, col=1)
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data[selected_asset], name='Cours', line=dict(color='#004b8d', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[selected_asset], mode='markers', name='Anomalie IA', marker=dict(color='red', size=8, symbol='x')), row=1, col=1)

    # Row 2 : RSI
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, layer="below", row=2, col=1)

    fig.update_layout(height=600, template="plotly_white", title="Analyse Technique & Comportementale")
    st.plotly_chart(fig, use_container_width=True)

# --- ONGLET 2 : RISK ANALYTICS ---
with tab2:
    st.subheader("Analyse Quantitative du Risque")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("##### Distribution des Rendements & VaR")
        fig_hist = px.histogram(asset_data, x="Returns", nbins=50, title="Distribution")
        fig_hist.add_vline(x=var_95, line_color="red", line_dash="dash", annotation_text=f"VaR 95%: {var_95:.2%}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.error(f"⚠️ **Value at Risk (95%)** : Une perte supérieure à **{abs(var_95):.2%}** ne devrait se produire qu'une fois tous les 20 jours.")

    with col_r2:
        st.markdown("##### Tableau des Pires Performances (Max Drawdown)")
        st.metric("Max Drawdown Historique", f"{max_dd:.2%}", help="La baisse maximale observée du sommet au creux.")
        
        # Tableau des anomalies
        st.markdown("###### Dernières Anomalies Détectées")
        anomalies_disp = anomalies[[selected_asset, 'Returns', 'RSI']].tail(10).sort_index(ascending=False)
        st.dataframe(anomalies_disp.style.format({selected_asset: "{:.2f}", 'Returns': "{:.2%}", 'RSI': "{:.1f}"}), height=200, use_container_width=True)

# --- ONGLET 3 : MONTE CARLO LAB ---
with tab3:
    st.subheader("🔮 Simulation Monte Carlo (Projection 1 an)")
    st.caption("Projection de 50 scénarios futurs basés sur la volatilité historique de l'actif.")

    col_sim1, col_sim2 = st.columns([3, 1])

    with col_sim1:
        # Lancer la simulation
        sim_data = monte_carlo_simulation(last_price, returns_mean, volatility/np.sqrt(252), days=252, simulations=50)
        
        fig_sim = go.Figure()
        for col in sim_data.columns:
            fig_sim.add_trace(go.Scatter(y=sim_data[col], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        
        # Ajouter la moyenne
        fig_sim.add_trace(go.Scatter(y=sim_data.mean(axis=1), mode='lines', line=dict(color='red', width=3), name='Moyenne Scénarios'))
        fig_sim.add_hline(y=last_price, line_dash="dash", line_color="black", annotation_text="Prix Départ")
        
        fig_sim.update_layout(height=500, title="Projection de Prix (252 Jours)", template="plotly_white", xaxis_title="Jours Futurs", yaxis_title="Prix Projeté")
        st.plotly_chart(fig_sim, use_container_width=True)

    with col_sim2:
        st.info("Résultats Probabilistes")
        final_prices = sim_data.iloc[-1]
        
        worst_case = np.percentile(final_prices, 5)
        best_case = np.percentile(final_prices, 95)
        avg_case = np.mean(final_prices)
        
        st.metric("Scénario Optimiste (95%)", f"{best_case:.0f} MAD", delta=f"{(best_case/last_price)-1:.1%}")
        st.metric("Scénario Central", f"{avg_case:.0f} MAD", delta=f"{(avg_case/last_price)-1:.1%}")
        st.metric("Scénario Pessimiste (5%)", f"{worst_case:.0f} MAD", delta=f"{(worst_case/last_price)-1:.1%}", delta_color="inverse")