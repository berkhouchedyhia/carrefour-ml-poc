# POC Machine Learning pour Carrefour Market

## Description

Ce prototype interactif démontre comment le Machine Learning peut adresser trois besoins métiers clés pour Carrefour Market :

1. **Mieux anticiper les comportements clients** : Prédire si un client va acheter un produit donné (classification)
2. **Optimiser la logistique** : Prévoir la demande journalière d'un produit (séries temporelles)
3. **Personnaliser les recommandations produit** : Recommander des produits à un client selon ses achats (recommandation)

## Fonctionnalités

### 1. Prédiction du comportement client
- Modèle de classification utilisant RandomForest
- Interface interactive pour tester différents profils clients
- Visualisation des données clients simulées
- Prédiction de probabilité d'achat

### 2. Prévision de la demande
- Modèle de régression linéaire pour les séries temporelles
- Graphique interactif montrant l'historique et les prévisions
- Possibilité de prévoir la demande pour une date future
- Données simulées sur une année complète

### 3. Système de recommandation
- Algorithme K-Nearest Neighbors pour la recommandation collaborative
- Matrice client-produit simulée
- Recommandations basées sur la similarité entre clients
- Interface de test pour différents clients

## Installation et lancement

### Prérequis
- Python 3.7+
- pip

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Lancement de l'application
```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : http://localhost:8501

## Structure du projet

```
carrefour_ml_poc/
├── app.py              # Application Streamlit principale
├── requirements.txt    # Dépendances Python
└── README.md          # Ce fichier
```

## Technologies utilisées

- **Streamlit** : Interface utilisateur interactive
- **Scikit-learn** : Modèles de Machine Learning
- **Pandas** : Manipulation des données
- **NumPy** : Calculs numériques
- **Matplotlib/Seaborn** : Visualisations

## Cas d'usage métier

### Pour les équipes marketing
- Identifier les clients les plus susceptibles d'acheter un produit
- Optimiser les campagnes publicitaires ciblées
- Personnaliser les offres promotionnelles

### Pour les équipes logistiques
- Anticiper les besoins en stock
- Optimiser les commandes fournisseurs
- Réduire le gaspillage et les ruptures de stock

### Pour l'expérience client
- Proposer des recommandations personnalisées
- Améliorer la satisfaction client
- Augmenter le panier moyen

## Limitations et améliorations possibles

### Limitations actuelles
- Données simulées (non réelles)
- Modèles simplifiés pour la démonstration
- Pas de persistance des données
- Interface basique

### Améliorations possibles
- Intégration avec des données réelles de Carrefour
- Modèles plus sophistiqués (Deep Learning, ensembles)
- Interface plus avancée avec tableaux de bord
- API REST pour intégration avec d'autres systèmes
- Monitoring et métriques de performance
- Tests A/B pour valider l'efficacité des recommandations

## Contact

Ce POC a été développé comme démonstration pédagogique pour illustrer le potentiel du Machine Learning dans le retail.

