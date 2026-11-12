
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import shap

from xai_commun import m, Xte, CAS

# Creer l'explicateur SHAP
ex = shap.TreeExplainer(m)

# Probabilites de fraude sur le jeu de test
proba = m.predict_proba(Xte)[:, 1]

# Selectionner la transaction la plus suspecte
idx = Xte.index[proba.argmax()]

# Expliquer les deux transactions
for nom, x in [
    ('benali', CAS),
    (f'test_{idx}', Xte.loc[[idx]])
]:

    # Contributions SHAP pour la classe fraude
    e = ex(x)[0, :, 1]

    print(f'--- {nom} :', x.iloc[0].to_dict())

    print(
        'valeur de base :',
        round(float(e.base_values), 4)
    )

    # Afficher les contributions par importance
    for col, c in sorted(
        zip(x.columns, e.values),
        key=lambda t: -abs(t[1])
    ):
        print(f' {col:11s} {c:+.4f}')

    # Verifier la propriete d'efficacite
    print(
        'base + somme   :',
        round(float(e.base_values + e.values.sum()), 4),
        '| prediction :',
        round(m.predict_proba(x)[0, 1], 4)
    )

    # Generer le graphique Waterfall
    shap.plots.waterfall(e, show=False)

    plt.tight_layout()

    plt.savefig(
        f'reports/shap_{nom}.png',
        dpi=120
    )

    plt.close('all')
