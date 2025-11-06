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

# Contraste & taille du texte (WCAG: redimensionnement + titres de page)
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

# Endpoints par défaut (adaptez si vos routes diffèrent)
PREDICT_ENDPOINTS = ["/predict", "/predict_proba", "/inference", "/score"]

# ==============================
# EN-TÊTE & INTRO
# ==============================
st.title("Dashboard Scoring Crédit")
st.caption("Ce tableau de bord interroge l’API déployée (P7) pour obtenir une prédiction de risque et explorer des résultats.")

with st.expander("ℹ️ Comment utiliser ce dashboard (lecture rapide)", expanded=True):
    st.markdown(
        """
        - **Prédiction unitaire** : saisissez quelques caractéristiques d’un client (à gauche), puis cliquez **Obtenir la prédiction**.  
        - **Batch CSV** : chargez un fichier CSV avec vos colonnes, envoyez au modèle et visualisez la distribution des scores.  
        - **Accessibilité** : vous pouvez **agrandir le texte** via le panneau à gauche. Les graphiques ont des **titres** et **info-bulles**,
          et ne dépendent pas uniquement de la couleur.
        """
    )

# ==============================
# PANNEAUX (SIDEBAR)
# ==============================
st.sidebar.header("🔌 Connexion à l’API")
endpoint_choice = st.sidebar.selectbox(
    "Choisir l’endpoint API à tester",
    PREDICT_ENDPOINTS,
    help="Sélectionnez la route correspondant à la prédiction côté API."
)
timeout_s = st.sidebar.slider(
    "Délai d’attente API (secondes)",
    min_value=5, max_value=60, value=20, step=5,
    help="Augmentez si Render est en 'cold start'."
)

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Mode",
    ["Prédiction unitaire", "Batch CSV"],
    help="Sélectionnez un mode de démonstration."
)

# ==============================
# OUTILS
# ==============================
def call_api(payload: dict | list):
    """
    Appelle l'API avec un JSON.
    - payload dict -> prédiction unitaire
    - payload list[dict] -> prédictions multiples
    """
    url = f"{API_BASE}{endpoint_choice}"
    headers = {"Content-Type": "application/json"}
    try:
        t0 = time.time()
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
        dt = time.time() - t0
        return resp, dt
    except requests.exceptions.RequestException as e:
        return e, None

def success_badge(prob, threshold=0.5):
    """
    Retourne une étiquette textuelle claire (pas uniquement couleur).
    """
    if prob is None:
        return "Résultat indisponible"
    label = "⚠️ Risque élevé" if prob >= threshold else "✅ Risque modéré/faible"
    return f"{label} — score: {prob:.3f} (seuil={threshold:.2f})"

# ==============================
# DÉMOS & FORMULAIRES
# ==============================
st.markdown("## 🧪 Démo de prédiction")

if mode == "Prédiction unitaire":
    col_form, col_result = st.columns([1.1, 1.2], gap="large")

    with col_form:
        st.subheader("Données d’entrée (exemple simplifié)")
        st.markdown(
            "Ces champs sont un **extrait minimal** typique du jeu Home Credit. "
            "Adaptez-les selon votre schéma exact de features côté API."
        )

        # Champs simples (exemple minimal réaliste)
        amt_credit = st.number_input("Montant du crédit (AMT_CREDIT)", min_value=0.0, value=150000.0, step=1000.0)
        amt_annuity = st.number_input("Mensualité (AMT_ANNUITY)", min_value=0.0, value=15000.0, step=500.0)
        amt_income = st.number_input("Revenu annuel (AMT_INCOME_TOTAL)", min_value=0.0, value=120000.0, step=5000.0)
        days_birth = st.number_input("Âge en jours négatifs (DAYS_BIRTH, ex: -14000 ≈ 38 ans)", value=-14000)
        days_employed = st.number_input("Ancienneté en jours négatifs (DAYS_EMPLOYED, ex: -3000)", value=-3000)
        ext1 = st.number_input("EXT_SOURCE_1 (0–1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
        ext2 = st.number_input("EXT_SOURCE_2 (0–1)", min_value=0.0, max_value=1.0, value=0.62, step=0.01)
        ext3 = st.number_input("EXT_SOURCE_3 (0–1)", min_value=0.0, max_value=1.0, value=0.58, step=0.01)

        st.caption("📝 Conseil : si votre API attend plus de colonnes, ajoutez un JSON personnalisé ci-dessous.")

        # Zone de JSON libre (optionnel) pour coller un payload exact
        with st.expander("➕ Coller un JSON personnalisé (écrase les champs ci-dessus)"):
            raw_json = st.text_area(
                "Collez ici un JSON conforme à votre schéma de features",
                height=160,
                placeholder='{"AMT_CREDIT": 150000, "AMT_ANNUITY": 15000, "AMT_INCOME_TOTAL": 120000, "DAYS_BIRTH": -14000, ...}'
            )

        ask = st.button("🚀 Obtenir la prédiction")

    with col_result:
        st.subheader("Résultat")
        if 'ask' in locals() and ask:
            # Construire le payload
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

            if payload is not None:
                resp, dt = call_api(payload)
                if isinstance(resp, Exception):
                    st.error(f"Échec d’appel API : {resp}")
                else:
                    st.write(f"⏱️ Temps de réponse : {dt:.2f} s")
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            # On essaie des clés usuelles
                            prob = None
                            for key in ["probability", "proba", "score", "default_proba", "prediction_proba"]:
                                if isinstance(data, dict) and key in data:
                                    prob = float(data[key])
                                    break
                            # Si la réponse est une liste, on tente le premier élément
                            if prob is None and isinstance(data, list) and data:
                                first = data[0]
                                for key in ["probability", "proba", "score", "default_proba", "prediction_proba"]:
                                    if key in first:
                                        prob = float(first[key]); break

                            st.success(success_badge(prob))
                            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                        except Exception:
                            st.warning("Réponse API non JSON ou inattendue.")
                            st.text(resp.text[:1000])
                    else:
                        st.error(f"Code HTTP {resp.status_code}")
                        st.text(resp.text[:1500])

elif mode == "Batch CSV":
    st.subheader("Prédictions par lot (CSV)")
    st.markdown("Chargez un CSV puis envoyez-le à l’API (selon le schéma attendu par votre endpoint).")

    file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    col_a, col_b = st.columns([1,1])

    with col_a:
        send = st.button("🚀 Envoyer au modèle")
    with col_b:
        show_preview = st.checkbox("Afficher un aperçu du CSV", value=True)

    if file is not None:
        df = pd.read_csv(file)
        if show_preview:
            st.write("Aperçu des données :")
            st.dataframe(df.head(20), use_container_width=True)

        if send:
            records = df.to_dict(orient="records")
            resp, dt = call_api(records)
            if isinstance(resp, Exception):
                st.error(f"Échec d’appel API : {resp}")
            else:
                st.write(f"⏱️ Temps de réponse : {dt:.2f} s")
                if resp.status_code == 200:
                    try:
                        out = resp.json()
                        # Normalisons la sortie en DataFrame si possible
                        if isinstance(out, list):
                            out_df = pd.DataFrame(out)
                        elif isinstance(out, dict) and "predictions" in out and isinstance(out["predictions"], list):
                            out_df = pd.DataFrame(out["predictions"])
                        else:
                            out_df = pd.DataFrame([out])

                        st.success("Prédictions reçues.")
                        st.dataframe(out_df, use_container_width=True)

                        # Graphique simple (distribution des scores si trouvés)
                        score_col = None
                        for c in out_df.columns:
                            if c.lower() in {"probability","proba","score","default_proba","prediction_proba"}:
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
                                .properties(height=280)
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.info("Aucune colonne de score reconnue dans la réponse pour tracer une distribution.")
                    except Exception:
                        st.warning("Réponse API non JSON ou inattendue.")
                        st.text(resp.text[:1500])
                else:
                    st.error(f"Code HTTP {resp.status_code}")
                    st.text(resp.text[:1500])

# ==============================
# PIED DE PAGE
# ==============================
st.markdown("---")
st.markdown(
    "🔎 **Accessibilité** : Titres explicites, zoom texte ajustable, graphiques avec info-bulles. "
    "Aucune information transmise par la couleur seule."
)
