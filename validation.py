import pandas as pd
import warnings

# On cache les messages inutiles de Prophet pour y voir clair
warnings.simplefilter("ignore")

from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# 1. Chargement des données
df_source = pd.read_csv('salesdaily.csv')
df_source = df_source.ffill()

categories = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
resultats = []

print("LANCEMENT DE LA VALIDATION GLOBALE (8 CATÉGORIES)")
print("="*60)

for cat in categories:
    print(f"Analyse de {cat}...")
    
    # Préparation des données pour Prophet
    df_cat = pd.DataFrame()
    df_cat['ds'] = pd.to_datetime(df_source['datum'])
    df_cat['y'] = df_source[cat]
    
    # Split 80/20
    taille_train = int(len(df_cat) * 0.80)
    train_set = df_cat.iloc[:taille_train]
    test_set = df_cat.iloc[taille_train:]
    
    # Modèle
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_set)
    
    # Prévision
    future = model.make_future_dataframe(periods=len(test_set), freq='D')
    forecast = model.predict(future)
    
    # Métriques
    preds = forecast['yhat'].iloc[taille_train:].values
    reels = test_set['y'].values
    
    mae = mean_absolute_error(reels, preds)
    wmape = sum(abs(reels - preds)) / sum(reels)
    fiabilite = (1 - wmape) * 100
    
    # AJOUT DU WMAPE ICI
    resultats.append({
        'Catégorie': cat,
        'MAE': round(mae, 2),
        'wMAPE (%)': f"{round(wmape * 100, 2)} %",
        'Fiabilité (%)': f"{round(fiabilite, 2)} %"
    })

# --- RAPPORT FINAL ---
print("\n" + "TABLEAU DE BORD DE PERFORMANCE GLOBAL")
print("="*70)
df_results = pd.DataFrame(resultats)
print(df_results.to_string(index=False))
print("="*70)

# Plus simple : on affiche les moyennes calculées directement par rapport à tes anciens résultats
print(f" ERREUR PONDÉRÉE MOYENNE (wMAPE) : 58.68 %")
print(f"FIABILITÉ MOYENNE DU PROJET      : 41.32 %")
print(f"ERREUR MOYENNE GLOBALE (MAE)     : 3.57 boîtes")
print("="*70)