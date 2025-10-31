# app.py 
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Demo modèle", layout="wide")
st.title("Démo : prédiction risque")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_pickle("lgbm_model.pkl")
    imputer = load_pickle("imputer.pkl")
    feature_names = load_pickle("feature_names.pkl")
    st.success("Modèle et objets chargés")
except Exception as e:
    st.error("Erreur chargement .pkl : " + str(e))
    st.stop()

# Charge test_data.csv
try:
    df = pd.read_csv("test_data.csv")
except Exception as e:
    st.error("Impossible de lire test_data.csv : " + str(e))
    st.stop()

st.write("Aperçu (5 premières lignes)")
st.dataframe(df.head())

# Vérifie colonnes
missing = [c for c in feature_names if c not in df.columns]
if missing:
    st.error("Colonnes manquantes pour prédiction: " + ", ".join(missing))
    st.stop()

X = df[feature_names].copy()
try:
    X_imp = imputer.transform(X)
except Exception:
    X_imp = X.values

# Prédiction
try:
    probs = model.predict_proba(X_imp)[:,1]
except Exception:
    probs = model.predict(X_imp)

df["score_defaut"] = probs
st.write("Distribution des scores")
st.bar_chart(df["score_defaut"])

st.write("Sélectionne une ligne pour voir le score")
idx = st.number_input("Index (ligne)", min_value=int(df.index.min()), max_value=int(df.index.max()), value=int(df.index.min()))
client = df.loc[int(idx)]
st.write("Score :", float(client["score_defaut"]))
st.download_button("Télécharger scores", df.to_csv(index=False), file_name="scores.csv")