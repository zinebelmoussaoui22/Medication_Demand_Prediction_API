import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import math

print("=== 1. Chargement et Nettoyage de la donnée ===")
df = pd.read_csv('salesdaily.csv')

# Conversion de la date
df['datum'] = pd.to_datetime(df['datum'])

# Choix du médicament
medicament = 'M01AB'

# Formatage spécifique pour Prophet : colonnes obligatoires 'ds' (date) et 'y' (valeur)
df_prophet = pd.DataFrame()
df_prophet['ds'] = df['datum']
df_prophet['y'] = df[medicament]

# Tri par date
df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

print("=== 2. Entraînement du modèle Meta Prophet ===")
# On initialise Prophet avec les saisonnalités (Annuelle et Hebdomadaire)
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(df_prophet)

# Création des dates futures (30 jours)
future = model.make_future_dataframe(periods=30, freq='D')

# Calcul des prédictions
forecast = model.predict(future)

# On récupère uniquement les 30 derniers jours prédits
predictions_futures = forecast.tail(30)

print("=== 3. Logique métier : Recommandation de Commande ===")
stock_actuel = 50
# La colonne de prédiction s'appelle 'yhat' dans Prophet
demande_prevue = predictions_futures['yhat'].sum()
# Sécurité : on arrondit à l'entier supérieur (math.ceil)
boites_prevues = math.ceil(max(0, demande_prevue)) 

print(f"\n[RÉSULTATS PROPHET POUR LA CATÉGORIE {medicament}]")
print(f"-> Demande brute calculée par l'IA : {demande_prevue:.2f}")
print(f"-> Besoins réels en boîtes entières : {boites_prevues} boîtes")

if stock_actuel < boites_prevues:
    print(f"-> ACTION CONSEILLÉE : Commander {boites_prevues - stock_actuel} boîtes rapidement.")
else:
    print("-> ACTION CONSEILLÉE : Stock suffisant, pas besoin de commander.")

print("\n=== 4. Génération du graphique ===")
plt.figure(figsize=(10, 5))

# On affiche les 90 derniers jours de l'historique pour le contexte
plt.plot(df_prophet['ds'].tail(90), df_prophet['y'].tail(90), label='Historique des ventes', color='#1f77b4')

# On affiche les 30 jours de prédiction
plt.plot(predictions_futures['ds'], predictions_futures['yhat'], label='Prédictions Meta Prophet (30j)', color='red', linestyle='--')

# Optionnel : Ajout de la zone d'incertitude (marge d'erreur de l'IA)
plt.fill_between(predictions_futures['ds'], predictions_futures['yhat_lower'], predictions_futures['yhat_upper'], color='red', alpha=0.1)

plt.title(f'Prévision de stock - Modèle Meta Prophet ({medicament})')
plt.xlabel('Date')
plt.ylabel('Nombre de boîtes')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('prediction_prophet_M01AB.png')

print("Graphique 'prediction_prophet_M01AB.png' enregistré avec succès !")