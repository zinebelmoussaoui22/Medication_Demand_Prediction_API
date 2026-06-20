import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

from prophet import Prophet

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Gestion de Stock Pharmacie", layout="centered")

st.title("🏥 Assistant IA : Prévision & Commande de Stock")
st.write("Cet outil utilise le modèle **Meta Prophet** pour anticiper intelligemment les besoins logistiques de votre officine.")

@st.cache_data
def load_clean_data():
    df = pd.read_csv('salesdaily.csv')
    df['datum'] = pd.to_datetime(df['datum'])
    df = df.sort_values('datum').reset_index(drop=True)
    return df

df_source = load_clean_data()

dict_medicaments = {
    'M01AB': 'M01AB - Anti-inflammatoires (Acide acétique)',
    'M01AE': 'M01AE - Anti-inflammatoires (Acide propionique)',
    'N02BA': 'N02BA - Analgésiques (Acide salicylique)',
    'N02BE': 'N02BE - Analgésiques & Antipyrétiques (Paracétamol)',
    'N05B': 'N05B - Anxiolytiques',
    'N05C': 'N05C - Hypnotiques & Sédatifs',
    'R03': 'R03 - Médicaments de l\'asthme / Voies respiratoires',
    'R06': 'R06 - Antihistaminiques (Allergies)'
}

st.sidebar.header("⚙️ Paramètres Logistiques")

choix_affichage = st.sidebar.selectbox("Sélectionner la catégorie ATC :", list(dict_medicaments.values()))
code_atc = [k for k, v in dict_medicaments.items() if v == choix_affichage][0]

jours_prevision = st.sidebar.slider("Période de prévision (en jours) :", min_value=1, max_value=30, value=7, step=1)
stock_actuel = st.sidebar.number_input("Stock actuel en rayon (Boîtes) :", min_value=0, value=20, step=5)

if st.sidebar.button("Calculer les prévisions", type="primary"):
    
    with st.spinner("L'IA Meta Prophet ajuste les calculs temporels..."):
        # Formatage strict pour Prophet (ds et y)
        df_prophet = pd.DataFrame()
        df_prophet['ds'] = df_source['datum']
        df_prophet['y'] = df_source[code_atc]
        
        # Entraînement sur 100% de la dataset pour le pharmacien
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(df_prophet)
        
        # Prévision des jours demandés
        future = model.make_future_dataframe(periods=jours_prevision, freq='D')
        forecast = model.predict(future)
        
        # Récupération des jours futurs uniquement
        predictions_futures = forecast['yhat'].tail(jours_prevision)
        
        # Somme de la demande à venir
        demande_prevue = predictions_futures.sum()
        # Sécurité : si la prédiction globale tombe sous 0, on la remet à 0
        boites_prevues = math.ceil(max(0, demande_prevue))
        
    # Affichage des indicateurs clés
    st.subheader(f"Analyse Prophet pour : {choix_affichage}")
    st.info(f"Calcul des besoins estimé pour les **{jours_prevision} prochains jours**.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Stock Actuel", f"{stock_actuel} bte(s)")
    col2.metric(f"Besoin ({jours_prevision}j)", f"{boites_prevues} bte(s)")
    
    if stock_actuel < boites_prevues:
        quantite_a_commander = boites_prevues - stock_actuel
        col3.metric("Action Requise", " Commander", delta=f"+{quantite_a_commander} boîtes", delta_color="inverse")
        st.error(f" **Alerte Rupture :** Votre stock ({stock_actuel} bte) ne tiendra pas {jours_prevision} jours. Commandez **{quantite_a_commander} boîtes**.")
    else:
        col3.metric("Action Requise", "✅ Stock OK", delta=None)
        st.success(f"🟢 **Situation Optimale :** Votre stock couvre largement les {jours_prevision} prochains jours.")
        
    st.subheader(" Projections des ventes futures")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Historique récent (les 45 derniers jours pour la lisibilité)
    ax.plot(df_source['datum'].tail(45), df_source[code_atc].tail(45), label='Historique récent', color='#1f77b4')
    
    # Traçage des prédictions futures Prophet
    dates_futures = forecast['ds'].tail(jours_prevision)
    ax.plot(dates_futures, predictions_futures, label=f'Prédictions Prophet ({jours_prevision}j)', color='red', linestyle='--', marker='o')
    
    ax.set_title(f"Planning de stock - Prophet")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=25)
    
    st.pyplot(fig)

else:
    st.info(" Ajustez la durée à gauche et cliquez sur **Calculer les prévisions**.")