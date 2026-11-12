# Model Card — fraude (version servie par l'alias production)

## Modèle
**Nom et version dans le registre MLflow :** fraude, version 1, alias production.

**Famille et réglages :** RandomForestClassifier, 500 arbres, profondeur maximale 16.

Run d'origine : b361fd7718fe4aebaa0da37e089272c8

## Objectif et usage prévu
Détecter les transactions bancaires potentiellement frauduleuses à partir de leurs caractéristiques afin de faciliter leur vérification par les équipes de lutte contre la fraude.

## Usages exclus
Le modèle ne doit pas être utilisé pour bloquer définitivement un compte bancaire ou sanctionner un client sans vérification humaine.

## Données
Fichier : data/transactions.csv
Nombre de lignes : 5000
Part de fraudes : 11.36 %
Empreinte MD5 : cf9f6f86c141e7df32c61715c493483f
Origine : donnees synthetiques du TP.


**Variables :** montant, anciennete, freq_24h et nuit.

**Variable cible :** fraude (0 = normal, 1 = fraude).

## Performances
Sur le jeu de test de 1 250 transactions :

- F1-score : 0.8669
- Précision : 0.8837
- Rappel : 0.8507
- AUC : 0.9872

## Méthodes d'explicabilité
**Globale :** importance par permutation et SHAP TreeExplainer.

**Locale :** SHAP Waterfall et LIME.

**Variables les plus influentes :** montant, anciennete, nuit et freq_24h.

L'explication SHAP de Mme Benali montre une probabilité de fraude prédite de 92,4 %, principalement influencée par l'horaire nocturne et le montant de la transaction.

## Comparaison avec un modèle interprétable
La régression logistique avec transformation log10(montant) atteint un F1-score de 0.8593 contre 0.8669 pour le Random Forest.

Elle détecte 113 fraudes sur 134, contre 114 pour la forêt, tout en offrant des coefficients plus faciles à interpréter.

## Limites et risques connus
Les données sont synthétiques et ne représentent pas nécessairement les comportements bancaires réels.

Les explications LIME peuvent varier selon la graine aléatoire et le nombre de voisins utilisés.

Les contributions SHAP expliquent le comportement du modèle, mais ne démontrent pas de relations causales.

Les faux positifs peuvent entraîner des vérifications inutiles pour les clients.

## Recommandations d'utilisation et de surveillance
Maintenir une validation humaine pour les décisions sensibles.

Surveiller régulièrement le F1-score, la précision, le rappel et les faux positifs.

Contrôler les dérives des données et réévaluer périodiquement les explications SHAP et LIME.

## Traçabilité
Run des explications : aa9a064edbb745dbb27bbd8428e05ef6

**Expérience MLflow :** tp7-explicabilite.

**Version du modèle :** fraude v1, alias production.

Empreinte MD5 : cf9f6f86c141e7df32c61715c493483f

Auteur : khawla bekkas

Date : 12/11/2026

**Prochaine revue :** 12/12/2026
