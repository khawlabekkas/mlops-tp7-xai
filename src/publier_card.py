
from pathlib import Path
import mlflow
from mlflow import MlflowClient

# Connexion MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')

c = MlflowClient()

# Recuperer la version en production
v = c.get_model_version_by_alias(
    'fraude',
    'production'
)

# Recuperer le run XAI de l'activite 6
run_id = Path(
    'reports/run_xai.txt'
).read_text().strip()

# Ajouter la Model Card aux artefacts du run
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact('MODEL_CARD.md')

# Lire le contenu de la Model Card
card = Path(
    'MODEL_CARD.md'
).read_text(encoding='utf-8')

# Ajouter la description au registre
c.update_model_version(
    'fraude',
    v.version,
    description=card
)

# Ajouter un tag de tracabilite
c.set_model_version_tag(
    'fraude',
    v.version,
    'model_card',
    'MODEL_CARD.md, run ' + run_id[:8]
)

print(
    'Model Card attachee a fraude v'
    + v.version,
    'et au run',
    run_id[:8]
)
