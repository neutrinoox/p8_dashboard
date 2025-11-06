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

# Sidebar: réglages d'affichage (WCAG: redimensionnement)
with st.sidebar:
    st.header("⚙️ Réglages d’affichage")
    base_font_scale = st.slider(
        "Taille du texte (100% = par défaut)",
        min_value=80, max_value=160, value=110, step=10,
        help="Agrandissez le texte si besoin (critère WCAG 1.4.4)."
    )
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
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
ENDPOINTS = ["/predict", "/predict_proba", "/inference", "/score"]  # choisis ce qui marche chez toi

# ==============================
# TITRE & INTRO
# ==============================
st.title("🏦 Dashboard Scoring Crédit")
st.caption("De la donnée brute à une décision éclairée : testez le scoring, explorez les résultats, comprenez les facteurs.")

with st.expander("🎯 Contexte & objectifs (1 min)", expanded=True):
    st.markdown(
        """
        **Objectif métier** : estimer le risque de défaut pour aider à la décision de crédit.  
        **Ce que vous pouvez faire ici :**  
        1) Tester une prédiction unitaire sur un profil type.  
        2) Charger un CSV pour obtenir une série de scores et analyser leur distribution.  
        **Lecture du score** : plus le score est élevé, plus le risque est important.
        """
    )

# ==============================
# SIDEBAR : paramètres de connexion
# ==============================
st.sidebar.header("🔌 Connexion à l’API")
endpoint_choice = st.sidebar.selectbox("Choisir l’endpoint à tester", ENDPOINTS)
timeout_s = st.sidebar.slider("⏱️ Délai d’attente (sec)", 5, 60, 20)
mode = st.sidebar.radio("Mode", ["Prédiction unitaire", "Batch CSV"])

# ==============================
# FONCTIONS UTILITAIRES
# ==============================
def call_api(payload):
    """
    Appelle l'API en POST avec le format attendu: {"data": ...}
    (corrige l'erreur HTTP 422: Field required 'data')
    """
    url = f"{API_BASE}{endpoint_choice}"
    headers = {"Content-Type": "application/json"}

    try:
        wrapped = {"data": payload}  # <— important
        t0 = time.time()
        resp = requests.post(url, headers=headers, data=json.dumps(wrapped), timeout=timeout_s)
        dt = time.time() - t0
        return resp, dt
    except requests.exceptions.RequestException as e:
        return e, None

def extract_probability(obj):
    """
    Récupère le score/proba quelle que soit la clé renvoyée par l'API.
    Ton API renvoie 'default_probability' d'après la capture.
    """
    candidate_keys = ["default_probability", "probability", "proba", "score", "prediction_proba"]
    if isinstance(obj, dict):
        for k in candidate_keys:
            if k in obj:
                try:
                    return float(obj[k])
                except Exception:
                    pass
    return None

def success_badge(prob, threshold=0.5, risk_text=None):
    """
    Fabrique une étiquette claire (pas seulement la couleur).
    Si 'risk_text' (ex: FAIBLE/MOYEN/ÉLEVÉ) est fourni par l'API, on l'affiche aussi.
    """
    if prob is None:
        return "Résultat indisponible"
    human = "⚠️ Risque élevé" if prob >= threshold else "✅ Risque modéré/faible"
    if risk_text:
        return f"{human} — score: {prob:.3f} (seuil={threshold:.2f}) — niveau: {risk_text}"
    return f"{human} — score: {prob:.3f} (seuil={threshold:.2f})"

# ==============================
# INTERFACE UTILISATEUR
# ==============================
st.markdown("## 🧮 Test de prédiction")

if mode == "Prédiction unitaire":
    col_form, col_result = st.columns([1.1, 1.2], gap="large")

    with col_form:
        st.subheader("Profil client (exemple)")
        amt_credit = st.number_input("Montant du crédit (AMT_CREDIT)", min_value=0.0, value=150000.0, step=1000.0)
        amt_annuity = st.number_input("Mensualité estimée (AMT_ANNUITY)", min_value=0.0, value=15000.0, step=500.0)
        amt_income = st.number_input("Revenu annuel du foyer (AMT_INCOME_TOTAL)", min_value=0.0, value=120000.0, step=5000.0)
        days_birth = st.number_input("Âge en jours négatifs (ex: -14000 ≈ 38 ans) [DAYS_BIRTH]", value=-14000)
        days_employed = st.number_input("Ancienneté en jours négatifs (ex: -3000) [DAYS_EMPLOYED]", value=-3000)
        ext1 = st.number_input("EXT_SOURCE_1 (0–1)", 0.0, 1.0, 0.55)
        ext2 = st.number_input("EXT_SOURCE_2 (0–1)", 0.0, 1.0, 0.62)
        ext3 = st.number_input("EXT_SOURCE_3 (0–1)", 0.0, 1.0, 0.58)

        st.caption("📝 Si l’API attend plus de colonnes, utilisez le JSON personnalisé ci-dessous.")

        with st.expander("➕ Coller un JSON personnalisé (facultatif)"):
            raw_json = st.text_area(
                "Collez ici un JSON complet (cela remplace les champs ci-dessus)",
                height=160,
                placeholder='{"AMT_CREDIT": 150000, "AMT_ANNUITY": 15000, "AMT_INCOME_TOTAL": 120000, "DAYS_BIRTH": -14000, ...}'
            )

        ask = st.button("🚀 Obtenir la prédiction")

    with col_result:
        st.subheader("Résultat")
        if ask:
            # 1) Construire le payload
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

            # 2) Appel API
            if payload is not None:
                resp, dt = call_api(payload)
                if isinstance(resp, Exception):
                    st.error(f"Erreur d’appel API : {resp}")
                else:
                    st.write(f"⏱️ Temps de réponse : {dt:.2f} s")
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except Exception:
                            st.warning("Réponse API non JSON.")
                            st.text(resp.text[:1000])
                            data = None

                        if data is not None:
                            # Extraction probabilité et niveau de risque/prediction si présents
                            prob = extract_probability(data)
                            risk_level = data.get("risk_level") if isinstance(data, dict) else None
                            decision = data.get("prediction") if isinstance(data, dict) else None

                            st.success(success_badge(prob, threshold=0.5, risk_text=risk_level))
                            if decision:
                                st.info(f"Interprétation : le modèle recommande **{decision}** pour ce dossier.")

                            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                    else:
                        st.error(f"Code HTTP {resp.status_code}")
                        st.text(resp.text[:1500])
        # Aide à la lecture
        st.info("Lecture : en dessous du seuil, le profil est considéré **modéré** ; au-dessus, **élevé**. "
                "Adaptez le seuil selon l’appétence au risque.")

elif mode == "Batch CSV":
    st.subheader("📂 Prédictions en lot (CSV)")
    file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    show_preview = st.checkbox("Afficher un aperçu du CSV", value=True)

    if file is not None:
        df = pd.read_csv(file)
        if show_preview:
            st.dataframe(df.head(15), use_container_width=True)

        if st.button("🚀 Envoyer au modèle"):
            records = df.to_dict(orient="records")
            resp, dt = call_api(records)
            if isinstance(resp, Exception):
                st.error(f"Erreur d’appel API : {resp}")
            else:
                st.write(f"⏱️ Temps de réponse : {dt:.2f} s")
                if resp.status_code == 200:
                    try:
                        out = resp.json()
                        # Normalisation en DataFrame
                        if isinstance(out, list):
                            out_df = pd.DataFrame(out)
                        elif isinstance(out, dict) and "predictions" in out:
                            out_df = pd.DataFrame(out["predictions"])
                        else:
                            out_df = pd.DataFrame([out])

                        st.success("Prédictions reçues ✅")
                        st.dataframe(out_df.head(30), use_container_width=True)

                        # Trouver une colonne de score pour le graphique
                        score_col = None
                        for c in out_df.columns:
                            if c.lower() in {"default_probability", "probability", "proba", "score", "prediction_proba"}:
                                score_col = c
                                break

                        if score_col:
                            st.markdown("### Distribution des scores")
                            chart = (
                                alt.Chart(out_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X(score_col, bin=alt.Bin(maxbins=30), title="Score / probabilité de défaut"),
                                    y=alt.Y("count()", title="Nombre de dossiers"),
                                    tooltip=[alt.Tooltip(score_col, title="Score")]
                                )
                                .properties(height=300)
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.info("Aucune colonne de score reconnue pour tracer une distribution.")
                    except Exception as e:
                        st.error(f"Erreur de lecture de la réponse : {e}")
                        st.text(resp.text[:1500])
                else:
                    st.error(f"Code HTTP {resp.status_code}")
                    st.text(resp.text[:1500])

# ==============================
# PIED DE PAGE
# ==============================
st.markdown("---")
st.markdown(
    "**À propos** — Modèle LightGBM (P7), API Render, tableau de bord Streamlit (P8). "
    "Pensé pour un public non technique : parcours simple, texte agrandissable, graphiques annotés. "
    "Aucune information transmise par la couleur seule."
)
