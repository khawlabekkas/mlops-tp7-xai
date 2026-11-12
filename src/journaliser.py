
import hashlib
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score

from xai_commun import NOM, ALIAS, VERSION, m, Xte, yte


# Calculer l'empreinte MD5 du fichier de donnees
def empreinte(chemin):
    return hashlib.md5(Path(chemin).read_bytes()).hexdigest()


# Predictions du modele
pred = m.predict(Xte)

# Creer l'experience MLflow
mlflow.set_experiment('tp7-explicabilite')

# Demarrer un run pour les explications
with mlflow.start_run(
    run_name=f'xai-{NOM}-v{VERSION.version}'
) as run:

    # Identifier le modele et les donnees
    mlflow.set_tags({
        'modele': NOM,
        'alias': ALIAS,
        'modele_version': VERSION.version,
        'modele_run_id': VERSION.run_id,
        'donnees': 'data/transactions.csv',
        'donnees_md5': empreinte('data/transactions.csv'),
        'methodes': 'permutation ; SHAP TreeExplainer ; LIME'
    })

    # Enregistrer les parametres des explications
    mlflow.log_params({
        'shap_echantillon': 400,
        'lime_num_samples': 2000,
        'lime_random_state': 0,
        'permutation_n_repeats': 10
    })

    # Enregistrer les metriques
    mlflow.log_metrics({
        'f1': f1_score(yte, pred),
        'precision': precision_score(yte, pred),
        'recall': recall_score(yte, pred)
    })

    # Archiver les graphiques et les fichiers
    for f in sorted(Path('reports').glob('*.*')):

        if f.suffix not in ('.png', '.html', '.json'):
            continue

        mlflow.log_artifact(
            str(f),
            artifact_path='explications'
        )

    # Sauvegarder l'identifiant du run
    Path('reports/run_xai.txt').write_text(
        run.info.run_id
    )

# Relier le run XAI a la version du modele
MlflowClient().set_model_version_tag(
    NOM,
    VERSION.version,
    'explications_run',
    run.info.run_id
)

print(
    'run',
    run.info.run_id[:8],
    ': explications du modele',
    NOM,
    'v' + VERSION.version,
    'archivees'
)
