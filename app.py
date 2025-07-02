# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="POC ML Carrefour Market")

st.title("POC Machine Learning pour Carrefour Market")

st.markdown("""
Ce prototype interactif démontre comment le Machine Learning peut adresser trois besoins métiers clés pour Carrefour Market :

1.  **Mieux anticiper les comportements clients** : Prédire si un client va acheter un produit donné.
2.  **Optimiser la logistique** : Prévoir la demande journalière d'un produit.
3.  **Personnaliser les recommandations produit** : Recommander des produits à un client selon ses achats.

Ce projet est une démo pédagogique, utilisant des données simulées et des modèles ML simplifiés pour illustrer les concepts.
""")

# --- 1. Anticiper les comportements clients (Classification) ---
st.header("1. Anticiper les comportements clients")
st.subheader("Prédiction d'achat d'un produit")

st.markdown("""
Cette section illustre comment un modèle de Machine Learning peut prédire la probabilité qu'un client achète un produit spécifique, basé sur son historique d'achat et ses caractéristiques.
""")

# Génération de données simulées pour la classification
@st.cache_data
def generate_customer_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'revenu_annuel': np.random.randint(20000, 80000, n_samples),
        'frequence_achats_mois': np.random.randint(1, 10, n_samples),
        'montant_moyen_panier': np.random.uniform(10, 100, n_samples),
        'achete_produit_cible': np.random.randint(0, 2, n_samples) # 0 ou 1
    }
    df = pd.DataFrame(data)
    # Rendre la prédiction un peu plus logique
    df['achete_produit_cible'] = ((df['frequence_achats_mois'] * 0.5 + df['montant_moyen_panier'] * 0.1 + df['revenu_annuel'] * 0.0001) > np.random.uniform(5, 15, n_samples)).astype(int)
    return df

customer_df = generate_customer_data()

# Modèle ML pour la classification
X = customer_df[['age', 'revenu_annuel', 'frequence_achats_mois', 'montant_moyen_panier']]
y = customer_df['achete_produit_cible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_classification = RandomForestClassifier(n_estimators=100, random_state=42)
model_classification.fit(X_train, y_train)

st.write("Aperçu des données clients simulées:")
st.dataframe(customer_df.head())

st.write(f"Précision du modèle de classification (sur données test): {model_classification.score(X_test, y_test):.2f}")

with st.expander("Tester la prédiction d'achat"):    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age_input = st.slider("Âge", 18, 70, 30)
    with col2:
        revenu_input = st.number_input("Revenu Annuel (€)", 20000, 80000, 45000, step=1000)
    with col3:
        frequence_input = st.slider("Fréquence Achats/Mois", 1, 10, 5)
    with col4:
        montant_input = st.number_input("Montant Moyen Panier (€)", 10.0, 100.0, 50.0, step=5.0)

    if st.button("Prédire l'achat"):        
        input_data = pd.DataFrame([[age_input, revenu_input, frequence_input, montant_input]], 
                                  columns=['age', 'revenu_annuel', 'frequence_achats_mois', 'montant_moyen_panier'])
        prediction = model_classification.predict(input_data)[0]
        prediction_proba = model_classification.predict_proba(input_data)[0][1] # Probabilité d'acheter

        if prediction == 1:
            st.success(f"Le client est susceptible d'acheter le produit cible (Probabilité: {prediction_proba:.2f})")
        else:
            st.warning(f"Le client n'est probablement pas susceptible d'acheter le produit cible (Probabilité: {prediction_proba:.2f})")

# --- 2. Optimiser la logistique (Séries Temporelles) ---
st.header("2. Optimiser la logistique")
st.subheader("Prévision de la demande journalière d'un produit")

st.markdown("""
Cette section démontre comment un modèle de séries temporelles peut prévoir la demande future d'un produit, aidant ainsi à optimiser les stocks et la logistique.
""")

# Génération de données simulées pour les séries temporelles
@st.cache_data
def generate_demand_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    demand = 50 + 10 * np.sin(np.linspace(0, 3 * np.pi, 365)) + np.random.normal(0, 5, 365)
    demand[demand < 0] = 0 # Pas de demande négative
    df = pd.DataFrame({'Date': dates, 'Demande': demand.astype(int)})
    return df

demand_df = generate_demand_data()

# Modèle ML pour les séries temporelles (régression linéaire simple)
demand_df['Jour_Index'] = (demand_df['Date'] - demand_df['Date'].min()).dt.days

X_demand = demand_df[['Jour_Index']]
y_demand = demand_df['Demande']

X_train_demand, X_test_demand, y_train_demand, y_test_demand = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)

model_demand = LinearRegression()
model_demand.fit(X_train_demand, y_train_demand)

st.write("Aperçu des données de demande simulées:")
st.dataframe(demand_df.head())

st.write(f"Score R² du modèle de prévision de demande (sur données test): {model_demand.score(X_test_demand, y_test_demand):.2f}")

# Visualisation de la demande
st.subheader("Historique et prévisions de la demande")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(demand_df['Date'], demand_df['Demande'], label='Demande réelle')

# Prévisions sur la période existante pour visualisation
predictions_demand = model_demand.predict(X_demand)
ax.plot(demand_df['Date'], predictions_demand, label='Prévisions du modèle', linestyle='--')

ax.set_xlabel("Date")
ax.set_ylabel("Demande")
ax.set_title("Demande journalière simulée et prévisions")
ax.legend()
st.pyplot(fig)

with st.expander("Prévoir la demande pour une date future"):    
    future_date = st.date_input("Sélectionnez une date future pour la prévision", value=pd.to_datetime('2024-01-01'))
    
    if st.button("Prédire la demande"):        
        future_day_index = (pd.to_datetime(future_date) - demand_df['Date'].min()).days
        future_demand_prediction = model_demand.predict(pd.DataFrame([[future_day_index]], columns=['Jour_Index']))[0]
        st.info(f"La demande prévue pour le {future_date.strftime('%d/%m/%Y')} est de **{max(0, int(future_demand_prediction))} unités**.")

# --- 3. Personnaliser les recommandations produit (Recommandation) ---
st.header("3. Personnaliser les recommandations produit")
st.subheader("Recommandation de produits basée sur les achats")

st.markdown("""
Cette section montre comment un système de recommandation peut suggérer des produits pertinents à un client, en se basant sur les produits qu'il a déjà achetés et les comportements d'achat similaires d'autres clients.
""")

# Génération de données simulées pour la recommandation
@st.cache_data
def generate_recommendation_data():
    np.random.seed(42)
    n_customers = 100
    n_products = 50
    
    # Créer une matrice client-produit (achats binaires)
    data = np.random.randint(0, 2, size=(n_customers, n_products)) # 0: non acheté, 1: acheté
    customer_product_df = pd.DataFrame(data, 
                                       index=[f'Client_{i+1}' for i in range(n_customers)],
                                       columns=[f'Produit_{j+1}' for j in range(n_products)])
    return customer_product_df

product_purchase_df = generate_recommendation_data()

# Modèle de recommandation (KNN - K-Nearest Neighbors)
# Utilisation de la similarité cosinus pour trouver des clients similaires
model_reco = NearestNeighbors(metric='cosine', algorithm='brute')
model_reco.fit(product_purchase_df)

st.write("Aperçu des données d'achat client-produit simulées:")
st.dataframe(product_purchase_df.head())

with st.expander("Obtenir des recommandations pour un client"):    
    customer_id_input = st.selectbox("Sélectionnez un Client", product_purchase_df.index)
    
    if st.button("Obtenir des recommandations"):        
        customer_index = product_purchase_df.index.get_loc(customer_id_input)
        distances, indices = model_reco.kneighbors(product_purchase_df.iloc[customer_index].values.reshape(1, -1), n_neighbors=5)
        
        # Trouver les produits achetés par les voisins mais pas par le client cible
        customer_purchases = product_purchase_df.iloc[customer_index]
        recommended_products = set()
        
        for i in range(1, len(indices.flatten())):
            neighbor_index = indices.flatten()[i]
            neighbor_purchases = product_purchase_df.iloc[neighbor_index]
            
            # Produits achetés par le voisin et non par le client cible
            new_recommendations = neighbor_purchases[(neighbor_purchases == 1) & (customer_purchases == 0)].index.tolist()
            recommended_products.update(new_recommendations)
            
        if recommended_products:
            st.success(f"Produits recommandés pour {customer_id_input}:")
            for prod in list(recommended_products)[:5]: # Limiter à 5 recommandations pour la démo
                st.write(f"- {prod}")
        else:
            st.info(f"Aucune nouvelle recommandation trouvée pour {customer_id_input} basée sur les voisins.")

st.markdown("""
--- 

### Conclusion

Ce POC démontre le potentiel du Machine Learning pour transformer les opérations et l'expérience client chez Carrefour Market. Chaque section présente un cas d'usage concret, de la prédiction des comportements d'achat à l'optimisation logistique et à la personnalisation des offres. Ces applications, bien que simplifiées ici, peuvent être étendues et affinées pour générer une valeur significative.
By Dyhia BERKHOUCHE""")


