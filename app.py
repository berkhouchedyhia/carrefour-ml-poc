import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

st.set_page_config(layout="wide", page_title="POC ML Carrefour Market")
st.sidebar.image("logo.png", width=200)

# ---------- STYLE GLOBAL CARREFOUR ----------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #002b5c 0%, #ff0000 100%) !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stRadio > div {
    flex-direction: column;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    margin: 6px 0;
    background-color: #003366;
    padding: 10px;
    border-radius: 8px;
    font-weight: bold;
    text-align: center;
    transition: 0.3s ease;
    border: 1px solid transparent;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background-color: #005bac;
    border: 1px solid white;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-selected="true"] {
    background-color: #ff0000;
    color: white;
    border: 2px solid white;
}
body {
    background-color: #eef2f5;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #003366;
}
.stButton>button {
    background-color: #002b5c;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.5em 1em;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #ff0000;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- NAVIGATION ----------
page = st.sidebar.radio("Navigation", [
    "Accueil",
    "1 - Comportement client",
    "2 - Prévision logistique",
    "3 - Recommandation produit"
])

# ---------- ACCUEIL ----------
if page == "Accueil":
    st.title("🧠 POC Machine Learning pour Carrefour Market")
    st.markdown("""
    Ce prototype interactif démontre comment le **Machine Learning** peut adresser trois besoins métiers clés :

    1. **Mieux anticiper les comportements clients**
    2. **Optimiser la logistique**
    3. **Personnaliser les recommandations produit**

    Ce projet est une démo pédagogique, utilisant des données simulées et des modèles ML simplifiés.
    """)

# ---------- SECTION 1 ----------
elif page == "1 - Comportement client":
    st.header("1 – Anticiper les comportements clients")
    st.subheader("Prédiction d'achat d'un produit")

    @st.cache_data
    def generate_customer_data():
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'age': np.random.randint(18, 70, n),
            'revenu_annuel': np.random.randint(20000, 80000, n),
            'frequence_achats_mois': np.random.randint(1, 10, n),
            'montant_moyen_panier': np.random.uniform(10, 100, n)
        })
        df['achete_produit_cible'] = ((df['frequence_achats_mois'] * 0.5 +
                                       df['montant_moyen_panier'] * 0.1 +
                                       df['revenu_annuel'] * 0.0001) >
                                      np.random.uniform(5, 15, n)).astype(int)
        return df

    df = generate_customer_data()
    X = df[['age', 'revenu_annuel', 'frequence_achats_mois', 'montant_moyen_panier']]
    y = df['achete_produit_cible']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    st.dataframe(df.head())
    st.write(f"**Score du modèle :** {clf.score(X_test, y_test):.2f}")

    st.markdown("#### Tester une prédiction")
    col1, col2, col3, col4 = st.columns(4)
    age = col1.slider("Âge", 18, 70, 35)
    revenu = col2.number_input("Revenu (€)", 20000, 80000, 40000, step=1000)
    freq = col3.slider("Achats/Mois", 1, 10, 4)
    panier = col4.number_input("Panier Moyen (€)", 10.0, 100.0, 50.0)

    if st.button("Prédire l'achat"):
        input_data = pd.DataFrame([[age, revenu, freq, panier]], columns=X.columns)
        pred = clf.predict(input_data)[0]
        proba = clf.predict_proba(input_data)[0][1]
        if pred == 1:
            st.success(f"Client susceptible d'acheter (probabilité : {proba:.2f})")
        else:
            st.warning(f"Client peu susceptible (probabilité : {proba:.2f})")

# ---------- SECTION 2 ----------
elif page == "2 - Prévision logistique":
    st.header("2 – Prévision de la demande journalière")

    @st.cache_data
    def generate_demand_data():
        dates = pd.date_range('2023-01-01', periods=365)
        demand = 50 + 10 * np.sin(np.linspace(0, 3 * np.pi, 365)) + np.random.normal(0, 5, 365)
        demand = np.clip(demand, 0, None)
        return pd.DataFrame({'Date': dates, 'Demande': demand.astype(int)})

    df_d = generate_demand_data()
    df_d["Jour"] = (df_d["Date"] - df_d["Date"].min()).dt.days
    X = df_d[['Jour']]
    y = df_d['Demande']
    reg = LinearRegression().fit(X, y)

    st.line_chart(df_d.set_index("Date")["Demande"])
    st.write(f"**Score R² :** {reg.score(X, y):.2f}")

    future_date = st.date_input("Date future", pd.to_datetime("2024-01-01"))
    days_ahead = (pd.to_datetime(future_date) - df_d["Date"].min()).days
    pred = reg.predict([[days_ahead]])[0]
    st.info(f"Demande prévue pour le {future_date}: **{int(pred)} unités**")

# ---------- SECTION 3 ----------
elif page == "3 - Recommandation produit":
    st.header("3 – Recommandation de produits")

    @st.cache_data
    def generate_matrix():
        clients, produits = 100, 30
        mat = np.random.randint(0, 2, (clients, produits))
        return pd.DataFrame(mat, index=[f"Client {i}" for i in range(clients)],
                            columns=[f"Produit {j}" for j in range(produits)])

    matrix = generate_matrix()
    model = NearestNeighbors(metric="cosine").fit(matrix)

    st.dataframe(matrix.head())
    client = st.selectbox("Choisir un client", matrix.index)
    if st.button("Recommander"):
        i = matrix.index.get_loc(client)
        dists, indices = model.kneighbors([matrix.iloc[i]], n_neighbors=5)
        voisins = indices[0][1:]
        recommandations = set()
        for v in voisins:
            voisin_achats = matrix.iloc[v]
            diff = voisin_achats[(voisin_achats == 1) & (matrix.iloc[i] == 0)].index
            recommandations.update(diff)
        if recommandations:
            st.success("Produits recommandés :")
            for p in list(recommandations)[:5]:
                st.write(f"- {p}")
        else:
            st.info("Aucune recommandation nouvelle pour ce client.")

# ---------- FOOTER ----------
st.markdown("<hr><p style='text-align:center;'>By Dyhia BERKHOUCHE</p>", unsafe_allow_html=True)
