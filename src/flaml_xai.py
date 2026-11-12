
import json
import numpy as np
import pandas as pd

from flaml import AutoML
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from xai_commun import m, X, Xtr, Xte, ytr, yte

# 1. Recherche automatique du meilleur modele
automl = AutoML()

automl.fit(
    Xtr,
    ytr,
    task='classification',
    metric='f1',
    time_budget=60,
    seed=42,
    verbose=0
)

# 2. Recuperer le meilleur modele
best = automl.model.estimator

print(
    'famille retenue :',
    automl.best_estimator,
    '|',
    type(best).__name__
)

# 3. Comparer les performances
print(
    'F1 FLAML :',
    round(f1_score(yte, best.predict(Xte)), 4)
)

print(
    'F1 foret :',
    round(f1_score(yte, m.predict(Xte)), 4)
)

# 4. Importance par permutation
r = permutation_importance(
    best,
    Xte,
    yte,
    scoring='f1',
    n_repeats=10,
    random_state=0
)

imp = pd.DataFrame({
    'FLAML': r.importances_mean,
    'foret (activite 2)': (
        pd.read_json('reports/importance.json')['permutation']
        .reindex(X.columns)
        .values
    )
}, index=X.columns).round(3)

print(
    imp.sort_values('FLAML', ascending=False).to_string()
)

# Sauvegarder les importances
imp.to_json('reports/flaml_importance.json')

# 5. Explication SHAP du modele FLAML
try:
    import shap

    v = shap.TreeExplainer(best)(
        Xte.sample(400, random_state=0)
    ).values

    if v.ndim == 3:
        v = v[:, :, 1]

    print(
        'SHAP moyen |valeur| :',
        dict(
            zip(
                X.columns,
                np.abs(v).mean(0).round(3).tolist()
            )
        )
    )

except Exception as err:
    print(
        'SHAP TreeExplainer indisponible pour cette famille :',
        type(err).__name__
    )
