import pandas as pd

df = pd.read_csv('salesdaily.csv')

print("=== 5 premières lignes ===")
print(df.head())

print("\n=== Taille du dataset ===")
print(df.shape)

print("\n=== Colonnes ===")
print(df.columns.tolist())

print("\n=== Types des colonnes ===")
print(df.dtypes)

print("\n=== Valeurs manquantes ===")
print(df.isnull().sum())