#  Assistant IA : Prévision de Demande & Optimisation des Stocks en Pharmacie

Ce projet propose un outil d'aide à la décision logistique destiné aux officines de pharmacie. Grâce au modèle prédictif **Meta Prophet**, l'application anticipe les besoins réels sur 8 catégories de médicaments clés afin d'éviter les ruptures de stock tout en limitant le surstockage.

---

## Structure du Projet

* `app.py` : Interface utilisateur web développée sous Streamlit pour le pharmacien.
* `validation.py` / `train_model.py` : Scripts de traitement des données, d'évaluation des performances et de calcul des métriques ($MAE$, $wMAPE$).
* `explore_data.py` : Analyse exploratoire initiale de la série temporelle.
* `salesdaily.csv` : Jeu de données contenant l'historique quotidien des ventes de l'officine.
* `requirements.txt` : Liste des dépendances logicielles indispensables.

---

##  Installation et Lancement

### 1. Cloner le projet et installer les dépendances
```bash
pip install -r requirements.txt