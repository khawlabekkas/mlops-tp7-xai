
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from xai_commun import m, Xte

# Selectionner 400 transactions de test
Xs = Xte.sample(400, random_state=0)

# Creer l'explicateur SHAP
ex = shap.TreeExplainer(m)

# Recuperer les contributions pour la classe fraude
sv = ex(Xs)[:, :, 1]

# Afficher la valeur de base
print(
    'valeur de base :',
    round(float(sv.base_values[0]), 4)
)

# Calculer l'importance globale
imp = pd.Series(
    np.abs(sv.values).mean(0),
    index=Xs.columns
)

print(
    imp.sort_values(ascending=False)
       .round(4)
       .to_string()
)

# Sauvegarder les importances
imp.round(4).to_json(
    'reports/shap_importance.json'
)

# Graphique 1 : SHAP Beeswarm
shap.plots.beeswarm(sv, show=False)

plt.tight_layout()
plt.savefig(
    'reports/shap_global.png',
    dpi=120
)
plt.close('all')

# Graphique 2 : Effet du montant
plt.scatter(
    Xs['montant'],
    sv[:, 'montant'].values,
    c=Xs['nuit'],
    cmap='coolwarm',
    s=12
)

plt.xscale('log')
plt.xlabel('montant (MAD)')
plt.ylabel('contribution SHAP du montant')

plt.colorbar(label='nuit')
plt.tight_layout()

plt.savefig(
    'reports/shap_montant.png',
    dpi=120
)

print('graphiques ecrits')
