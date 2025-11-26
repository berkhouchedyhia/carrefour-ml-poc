# POC Machine Learning - Carrefour Market 🧠

## 📌 Objectif
Ce projet interactif démontre comment le Machine Learning peut répondre à trois besoins métiers stratégiques de Carrefour Market :
1. **Anticiper les comportements clients** : Prédiction d’achat
2. **Optimiser la logistique** : Prévision de la demande
3. **Personnaliser les recommandations produit** : Recommandation basée sur les achats

---

## Fonctionnalités principales

### 🔹 Prédiction d'achat (Classification)
- Modèle : `RandomForestClassifier`
- Entrées : âge, revenu, fréquence d'achat, panier moyen
- Sortie : probabilité d’achat d’un produit

### 🔹 Prévision de la demande (Séries temporelles)
- Modèle : `LinearRegression`
- Données : demande journalière simulée (365 jours)
- Visualisation interactive + prévision d’une date future

### 🔹 Recommandation produit (Collaborative Filtering)
- Modèle : `K-Nearest Neighbors`
- Données : matrice binaire client-produit
- Recommandation par similarité entre clients

---

## Structure du projet

```
carrefour-ml-poc/
├── app.py                  # Application Streamlit
├── requirements.txt        # Dépendances Python
├── lancer_app_ml.bat       # Script de lancement automatique
├── README.md               # Ce fichier
└── POC_Carrefour_Market_ML.pdf  # Rapport complet
```

---

## Installation

### Prérequis
- Python 3.7+
- pip

### Étapes

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Accès local via : [http://localhost:8501](http://localhost:8501)

---

### Dépendances Python utilisées

Le projet utilise les bibliothèques suivantes :

- `streamlit` : interface utilisateur interactive
- `pandas` : manipulation des données tabulaires
- `numpy` : calculs numériques
- `scikit-learn` : modèles de Machine Learning
- `matplotlib` : graphiques standards
- `seaborn` : visualisation statistique avancée


---

## Cas d’usage métier

### Marketing
- Ciblage comportemental
- Campagnes personnalisées

### Logistique
- Prévision de stock
- Réduction des ruptures

### Expérience client
- Recommandations pertinentes
- Augmentation du panier moyen

---

## Améliorations possibles
- Connexion à des données réelles Carrefour
- Intégration d’algorithmes avancés (XGBoost, LSTM)
- Tableau de bord complet
- API REST / Monitoring
- Validation via tests A/B

---

## Créatrice
Développé par **Dyhia BERKHOUCHE** dans un cadre pédagogique pour illustrer l’usage opérationnel du Machine Learning dans le secteur du retail.
