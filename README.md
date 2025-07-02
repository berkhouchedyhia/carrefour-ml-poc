# 🧠 POC Machine Learning - Carrefour Market

## 📌 Objectif
Ce projet démontre comment le Machine Learning peut répondre à trois besoins métiers clés pour Carrefour Market :
1. **Anticiper les comportements clients** : Prédire la probabilité d’achat
2. **Optimiser la logistique** : Prévoir la demande journalière
3. **Personnaliser les recommandations produit** : Recommander des articles pertinents aux clients

---

## 🚀 Fonctionnalités

### 🔹 Prédiction d'achat (Classification)
- Modèle : `RandomForestClassifier`
- Données : âge, revenu, fréquence d'achat, panier moyen
- Affichage de la probabilité d’achat via Streamlit

### 🔹 Prévision de la demande (Séries temporelles)
- Modèle : `LinearRegression`
- Données : demande quotidienne simulée (365 jours)
- Prévision de la demande future

### 🔹 Recommandation produit (Collaborative Filtering)
- Modèle : `K-Nearest Neighbors`
- Données : matrice client-produit binaire
- Suggestions de produits non encore achetés par client

---

## 🏗️ Structure du projet

```
carrefour_ml_poc/
├── app.py              # Interface Streamlit
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

---

## ⚙️ Installation

### Prérequis
- Python 3.7+
- pip

### Étapes

```bash
pip install -r requirements.txt
streamlit run app.py
```

L’application sera disponible sur [http://localhost:8501](http://localhost:8501)

---

## 📊 Technologies utilisées
- `Python`, `Streamlit`
- `Scikit-learn`, `Pandas`, `NumPy`
- `Matplotlib`, `Seaborn`

---

## ✅ Cas d’usage métier

### Marketing
- Ciblage client, campagnes intelligentes, recommandations personnalisées

### Logistique
- Prévision de stock, optimisation de la chaîne d’approvisionnement

### Expérience client
- Suggestions personnalisées, meilleure satisfaction, panier moyen augmenté

---

## ⚠️ Limitations actuelles
- Données simulées
- Modèles simplifiés
- Pas de base de données persistante
- Interface basique

## 💡 Pistes d'amélioration
- Intégration de données Carrefour réelles
- Modèles avancés (XGBoost, LSTM)
- Tableau de bord Streamlit avec KPI
- Intégration API / REST
- Tests A/B, logs utilisateurs

---

## 🧾 A propos
Ce projet est une **démonstration pédagogique** du potentiel du Machine Learning dans le retail.

