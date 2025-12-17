import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Wafa Gestion - Risk Monitor Pro",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# --- 2. CSS CUSTOM (STYLE "BLOOMBERG / POWER BI") ---
st.markdown("""
<style>
    /* Fond global */
    .stApp {background-color: #f0f2f6;}
    
    /* Style des Cartes (KPIs) */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Titres */
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif; color: #0f172a;}
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 5px;
        color: #0f172a;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #004b8d; /* Bleu Wafa */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FONCTIONS FINANCIÈRES & OPTIMISATION (CACHE) ---

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

@st.cache_data
def run_markowitz_optimization(df_selection):
    """
    Fonction OPTIMISÉE ET CACHÉE.
    Elle ne se recalcule que si les données d'entrée (df_selection) changent.
    """
    # Fixer la graine aléatoire pour la reproductibilité (Stabilité visuelle)
    np.random.seed(42)
    
    returns_df = df_selection.pct_change().dropna()
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_portfolios = 5000 # Nombre de simulations
    risk_free_rate = 0.03

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(df_selection.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
    return results, weights_record

# --- 4. CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    file_path = "donnees_bourse_pro.csv"
    dates = pd.date_range(start="2022-01-01", end=pd.Timestamp.today(), freq='B')
    
    # Si le fichier n'existe pas, on simule des données réalistes
    if not os.path.exists(file_path):
        data = pd.DataFrame(index=dates)
        np.random.seed(42)
        configs = [('IAM.MA', 100, 0.15), ('ATW.MA', 450, 0.22), ('BCP.MA', 260, 0.18), ('Lafarge', 1800, 0.12), ('MarsaMaroc', 280, 0.16)]
        for ticker, start, vol in configs:
            dt = 1/252
            mu = 0.05
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

# --- 5. SIDEBAR (PARAMÈTRES) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ec/Attijariwafa_bank_logo.svg", width=180)
    
    st.markdown("### ⚙️ Paramètres")
    # Ce sélecteur ne doit influencer QUE les onglets 1, 2, 3
    selected_asset = st.selectbox("Actif Financier (Vue Détail)", df.columns)
    
    st.markdown("### 🧠 Modèle IA")
    contamination = st.slider("Sensibilité (Outliers)", 0.01, 0.10, 0.03, help="Plus le taux est élevé, plus l'IA est sévère.")
    
    st.markdown("---")
    st.caption("⚠️ **Avertissement :** Prototype éducatif. Ne constitue pas un conseil financier.")
    st.write("By **Adam El Menouar** | PFE 2026")

# --- 6. CALCULS BACKEND (POUR L'ACTIF SÉLECTIONNÉ) ---
asset_data = df[[selected_asset]].copy()
asset_data['Returns'] = asset_data[selected_asset].pct_change().fillna(0)

# Calculs Indicateurs
asset_data['RSI'] = calculate_rsi(asset_data[selected_asset])
asset_data['BB_Upper'], asset_data['BB_Lower'] = calculate_bollinger_bands(asset_data[selected_asset])
asset_data['MA50'] = asset_data[selected_asset].rolling(50).mean()

# IA : Isolation Forest
model = IsolationForest(contamination=contamination, random_state=42)
asset_data['Anomaly_Score'] = model.fit_predict(asset_data['Returns'].values.reshape(-1, 1))
anomalies = asset_data[asset_data['Anomaly_Score'] == -1]

# Métriques Clés
last_price = asset_data[selected_asset].iloc[-1]
returns_mean = asset_data['Returns'].mean()
volatility = asset_data['Returns'].std() * np.sqrt(252) # Annualisée
max_dd = calculate_max_drawdown(asset_data[selected_asset])
var_95 = np.percentile(asset_data['Returns'], 5) # VaR 95%

# --- 7. INTERFACE PRINCIPALE (DASHBOARD) ---
st.title(f"🦅 Risk Monitor Pro : {selected_asset}")

# Création des 4 Onglets
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Cockpit", "📊 Risk Analytics", "🔮 Monte Carlo", "⚖️ Optimiseur Markowitz"])

# === ONGLET 1 : MARKET COCKPIT ===
with tab1:
    # KPIs Top Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix Actuel", f"{last_price:.2f} MAD", f"{asset_data['Returns'].iloc[-1]:.2%}", help="Dernier cours de clôture")
    c2.metric("Volatilité (An)", f"{volatility:.2%}", delta_color="inverse", help="Mesure de l'incertitude sur 1 an")
    c3.metric("RSI (14j)", f"{asset_data['RSI'].iloc[-1]:.1f}", delta="Surachat" if asset_data['RSI'].iloc[-1] > 70 else ("Survente" if asset_data['RSI'].iloc[-1] < 30 else "Neutre"), help=">70 = Cher, <30 = Pas cher")
    c4.metric("Alertes IA", f"{len(anomalies)}", delta_color="inverse", help="Jours anormaux détectés par l'algorithme")

    st.markdown("---")

    # Graphique Combiné
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Row 1 : Prix & Bandes Bollinger
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['BB_Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['BB_Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='Bandes Bollinger'), row=1, col=1)
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data[selected_asset], name='Cours', line=dict(color='#004b8d', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[selected_asset], mode='markers', name='Anomalie IA', marker=dict(color='red', size=8, symbol='x')), row=1, col=1)

    # Row 2 : RSI
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, layer="below", row=2, col=1)

    fig.update_layout(height=600, template="plotly_white", title="Analyse Technique & Détection d'Anomalies", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# === ONGLET 2 : RISK ANALYTICS ===
with tab2:
    st.subheader("Analyse Quantitative du Risque")
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("##### Distribution des Rendements & VaR")
        fig_hist = px.histogram(asset_data, x="Returns", nbins=50, title="Distribution Statistique", color_discrete_sequence=['#004b8d'])
        fig_hist.add_vline(x=var_95, line_color="red", line_dash="dash", annotation_text=f"VaR 95%: {var_95:.2%}")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.error(f"⚠️ **Value at Risk (95%)** : Une perte supérieure à **{abs(var_95):.2%}** ne devrait se produire qu'un jour sur 20.")

    with col_r2:
        st.markdown("##### Analyse des Pires Performances")
        st.metric("Max Drawdown Historique", f"{max_dd:.2%}", help="La pire chute observée du sommet au creux.")
        
        st.markdown("###### Journal des Anomalies (IA)")
        anomalies_disp = anomalies[[selected_asset, 'Returns', 'RSI']].tail(10).sort_index(ascending=False)
        st.dataframe(anomalies_disp.style.format({selected_asset: "{:.2f}", 'Returns': "{:.2%}", 'RSI': "{:.1f}"}), height=200, use_container_width=True)

# === ONGLET 3 : MONTE CARLO ===
with tab3:
    st.subheader("🔮 Simulation Monte Carlo (Projection 1 an)")
    st.caption("Projection de 50 scénarios futurs possibles basés sur la volatilité historique.")

    col_sim1, col_sim2 = st.columns([3, 1])

    with col_sim1:
        # Lancer la simulation
        sim_data = monte_carlo_simulation(last_price, returns_mean, volatility/np.sqrt(252), days=252, simulations=50)
        
        fig_sim = go.Figure()
        for col in sim_data.columns:
            fig_sim.add_trace(go.Scatter(y=sim_data[col], mode='lines', line=dict(width=1), opacity=0.2, showlegend=False))
        
        # Ajouter la moyenne
        fig_sim.add_trace(go.Scatter(y=sim_data.mean(axis=1), mode='lines', line=dict(color='red', width=3), name='Moyenne Scénarios'))
        fig_sim.add_hline(y=last_price, line_dash="dash", line_color="black", annotation_text="Prix Départ")
        
        fig_sim.update_layout(height=500, title="Projection Stochastique (252 Jours)", template="plotly_white", xaxis_title="Jours Futurs", yaxis_title="Prix Projeté")
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

# === ONGLET 4 : OPTIMISEUR DE PORTEFEUILLE (STABILISÉ AVEC CACHE) ===
with tab4:
    st.subheader("⚖️ Allocation Optimale d'Actifs (Frontière Efficiente)")
    st.caption("L'optimisation ne se recalcule que si vous changez l'univers d'investissement ci-dessous (Stabilité Garantie).")

    # SÉLECTEUR MULTIPLE
    # Ce sélecteur est INDÉPENDANT de la sidebar
    selected_assets_opt = st.multiselect(
        "Univers d'Investissement :",
        options=df.columns,
        default=df.columns.tolist()
    )

    if len(selected_assets_opt) < 2:
        st.warning("⚠️ Veuillez sélectionner au moins 2 actifs pour effectuer une optimisation de diversification.")
    else:
        st.markdown("---")
        
        # 1. APPEL DE LA FONCTION CACHÉE
        # Le calcul lourd ne se lance que si 'selected_assets_opt' a changé
        try:
            # On prépare le sous-dataframe à envoyer à la fonction
            df_input = df[selected_assets_opt].copy()
            results, weights_record = run_markowitz_optimization(df_input)
            
            # 2. Identification du Meilleur Portefeuille (Max Sharpe)
            max_sharpe_idx = np.argmax(results[2])
            sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
            best_weights = weights_record[max_sharpe_idx]

            col_opt1, col_opt2 = st.columns([2, 1])

            # 3. Affichage Graphique
            with col_opt1:
                fig_eff = go.Figure()
                
                # Nuage de points
                fig_eff.add_trace(go.Scatter(
                    x=results[0,:], 
                    y=results[1,:], 
                    mode='markers',
                    marker=dict(
                        color=results[2,:], 
                        colorscale='Viridis', 
                        showscale=True, 
                        size=5,
                        colorbar=dict(title="Ratio de Sharpe")
                    ),
                    name='Simulations'
                ))
                
                # L'étoile rouge
                fig_eff.add_trace(go.Scatter(
                    x=[sdp], y=[rp], mode='markers',
                    marker=dict(color='red', size=15, symbol='star'),
                    name='Portfolio Optimal'
                ))

                fig_eff.update_layout(
                    title="Frontière Efficiente (Risque vs Rendement)",
                    xaxis_title="Risque (Volatilité Annuelle)",
                    yaxis_title="Rendement Espéré (Annuel)",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig_eff, use_container_width=True)

            # 4. Résultats Allocation
            with col_opt2:
                st.success("✅ **Allocation Recommandée**")
                st.write("Répartition optimale pour maximiser le ratio de Sharpe :")
                
                # Camembert de répartition
                fig_pie = px.pie(
                    names=selected_assets_opt, 
                    values=best_weights, 
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.metric("Rendement Espéré", f"{rp:.2%}")
                st.metric("Volatilité Attendue", f"{sdp:.2%}")
                st.metric("Ratio de Sharpe", f"{results[2,max_sharpe_idx]:.2f}", help="Indicateur de performance. >1 est excellent.")
        
        except Exception as e:
            st.error(f"Erreur lors de l'optimisation : {e}")