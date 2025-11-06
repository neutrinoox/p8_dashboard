# app.py
import json
import time
import requests
import pandas as pd
import streamlit as st
import altair as alt

# ==============================
# CONFIG & ACCESSIBILITÉ
# ==============================
st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="📊",
    layout="wide",
)

# Taille de texte ajustable (accessibilité WCAG 1.4.4)
with st.sidebar:
    st.header("⚙️ Réglages d’affichage")
    base_font_scale = st.slider(
        "Taille du texte (100% = par défaut)",
        min_value=80, max_value=160, value=110, step=10,
        help="Agrandissez le texte si besoin (critère WCAG 1.4.4)"
    )
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            font-size: {base_font_scale}%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# PARAMÈTRES API
# ==============================
API_BASE = "https://projet7-credit-scoring-api.onrender.com"
ENDPOINTS = ["/predict", "/predict_proba", "/inference", "/score"]

# ==============================
# TITRE & INTRO
# ==============================
st.title("🏦 Dashboard Scoring Crédit")
st.caption("Application interactive connectée à l’API du projet 7 — OpenClassrooms Data Scientist")

st.markdown("""
Ce tableau de bord permet de **tester le modèle de scoring** de crédit développé dans le projet précédent.
L’utilisateur peut saisir les données d’un client ou charger un fichier CSV pour obtenir les prédictions du modèle.
""")

# ==============================
# SIDEBAR : paramètres
# ==============================
st.sidebar.header("🔌 Connexion à l’API")
endpoint_choice = st.sidebar.selectbox("Choisir l’endpoint à tester", ENDPOINTS)
timeout_s = st.sidebar.slider("⏱️ Délai d’attente (sec)", 5, 60, 20)

mode = st.sidebar.radio("Mode", ["Prédiction unitaire", "Batch CSV"])

# ==============================
# APPEL API (corrigé)
# ==============================
def call_api(payload):
    """
    Appelle l'API avec le bon format : {"data": payload}
    Corrige l'erreur 422 ("Field required: data")
    """
    url = f"{API_BASE}{endpoint_choice}"
    headers = {"Content-Type": "application/json"}
    try:
        wrapped = {"data": payload}  # ✅ l’API attend ce format
        t0 = time.time()
        resp = requests.post(url, headers=headers, data=json.dumps(wrapped), timeout=timeout_s)
        dt = time.time() - t0
        return resp, dt
    except requests.exceptions.RequestException as e:
        return e, None

def success_badge(prob, threshold=0.5):
    if prob is None:
        return "Résultat indisponible"
    label = "⚠️ Risque élevé" if prob >= threshold else "✅ Risque modéré/faible"
    return f"{label} — score: {prob:.3f} (seuil={threshold:.2f})"

# ==============================
# INTERFACE
# ==============================
st.markdown("## 🧮 Test de prédiction")

if mode == "Prédiction unitaire":
    col_form, col_result = st.columns([1.1, 1.2])

    with col_form:
        st.subheader("Données d’entrée (exemple simplifié)")
        amt_credit = st.number_input("Montant du crédit", min_value=0.0, value=150000.0)
        amt_annuity = st.number_input("Mensualité", min_value=0.0, value=15000.0)
        amt_income = st.number_input("Revenu annuel", min_value=0.0, value=120000.0)
        days_birth = st.number_input("Âge en jours négatifs (ex: -14000 ≈ 38 ans)", value=-14000)
        days_employed = st.number_input("Ancienneté (ex: -3000)", value=-3000)
        ext1 = st.number_input("EXT_SOURCE_1 (0–1)", 0.0, 1.0, 0.55)
        ext2 = st.number_input("EXT_SOURCE_2 (0–1)", 0.0, 1.0, 0.62)
        ext3 = st.number_input("EXT_SOURCE_3 (0–1)", 0.0, 1.0, 0.58)

        with st.expander("➕ Coller un JSON personnalisé (facultatif)"):
            raw_json = st.text_area(
                "Collez ici un JSON complet (cela remplacera les valeurs ci-dessus)",
                height=160,
                placeholder='{"AMT_CREDIT": 150000, "AMT_ANNUITY": 15000, ...}'
            )

        ask = st.button("🚀 Obtenir la prédiction")

    with col_result:
        st.subheader("Résultat")
        if ask:
            if raw_json.strip():
                try:
                    payload = json.loads(raw_json)
                except Exception as e:
                    st.error(f"JSON invalide : {e}")
                    payload = None
            else:
                payload = {
                    "AMT_CREDIT": amt_credit,
                    "AMT_ANNUITY": amt_annuity,
                    "AMT_INCOME_TOTAL": amt_income,
                    "DAYS_BIRTH": days_birth,
                    "DAYS_EMPLOYED": days_employed,
                    "EXT_SOURCE_1": ext1,
                    "EXT_SOURCE_2": ext2,
                    "EXT_SOURCE_3": ext3,
                }

            if payload:
                resp, dt = call_api(payload)
                if isinstance(resp, Exception):
                    st.error(f"Erreur API : {resp}")
                else:
                    st.write(f"⏱️ Temps de réponse : {dt:.2f} s")
                    if resp.status_code == 200:
                        data = resp.json()
                        prob = None
                        if isinstance(data, dict):
                            for k in ["probability", "proba", "score", "prediction_proba"]:
                                if k in data:
                                    prob = float(data[k])
                                    break
                        elif isinstance(data, list) and len(data) > 0:
                            for k in ["probability", "proba", "score", "prediction_proba"]:
                                if k in data[0]:
                                    prob = float(data[0][k])
                                    break
                        st.success(success_badge(prob))
                        st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                    else:
                        st.error(f"Code HTTP {resp.status_code}")
                        st.text(resp.text[:800])

elif mode == "Batch CSV":
    st.subheader("📂 Prédictions en lot (CSV)")
    file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head(10))
        if st.button("🚀 Envoyer au modèle"):
            records = df.to_dict(orient="records")
            resp, dt = call_api(records)
            if isinstance(resp, Exception):
                st.error(f"Erreur API : {resp}")
            elif resp.status_code == 200:
                data = resp.json()
                st.success("Prédictions reçues ✅")
                try:
                    out_df = pd.DataFrame(data if isinstance(data, list) else data.get("predictions", []))
                    st.dataframe(out_df.head(20))
                    if "score" in out_df.columns:
                        chart = alt.Chart(out_df).mark_bar().encode(
                            x=alt.X("score", bin=alt.Bin(maxbins=30)),
                            y="count()",
                            tooltip=["score"]
                        ).properties(height=300)
                        st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur de lecture des résultats : {e}")
            else:
                st.error(f"Code HTTP {resp.status_code}")
                st.text(resp.text[:800])

# ==============================
# PIED DE PAGE
# ==============================
st.markdown("---")
st.markdown(
    "🔎 **Accessibilité** : titres explicites, texte ajustable, graphiques lisibles et info-bulles détaillées. "
    "Aucune information uniquement transmise par la couleur."
)

