
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

# Connexion au serveur MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Recuperer le modele en production
NOM, ALIAS = 'fraude', 'production'

VERSION = MlflowClient().get_model_version_by_alias(
    NOM, ALIAS
)

# Charger le modele Random Forest du TP2
m = mlflow.sklearn.load_model(
    f'models:/{NOM}@{ALIAS}'
)

# Charger le dataset
df = pd.read_csv('data/transactions.csv')

X = df.drop(columns='fraude')
y = df['fraude']

# Meme decoupage que dans le TP2
Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Transaction de Mme Benali
CAS = pd.DataFrame([{
    'montant': 1800.0,
    'anciennete': 30,
    'freq_24h': 4,
    'nuit': 1
}])
