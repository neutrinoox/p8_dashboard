# app.py
import io
import json
import time
import requests
import pandas as pd
import streamlit as st
import altair as alt

# ==============================
# CONFIG & ACCESSIBILITÉ (WCAG)
# ==============================
st.set_page_config(
    page_title="Dashboard Scoring Crédit – P8",
    page_icon="📊",
    layout="wide",
)

with st.sidebar:
    st.header("⚙️ Réglages d’affichage")
    font_scale = st.slider(
        "Taille du texte (100% défaut)", 80, 180, 110, 10,
        help="WCAG 1.4.4 : redimensionnement du texte"
    )
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{ font-size: {font_scale}%; }}
        /* Focus visible pour clavier (accessibilité) */
        :focus {{ outline: 3px solid #4F46E5 !important; outline-offset: 2px; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# PARAMÈTRES API (P7)
# ==============================
API_BASE = "https://projet7-credit-scoring-api.onrender.com"
ENDPOINTS = ["/predict", "/predict_proba", "/inference", "/score"]  # choisis celui qui répond
DEFAULT_ENDPOINT = "/predict"

st.sidebar.header("🔌 Connexion à l’API")
endpoint_choice = st.sidebar.selectbox("Endpoint à tester", ENDPOINTS, index=ENDPOINTS.index(DEFAULT_ENDPOINT))
timeout_s = st.sidebar.slider("⏱️ Délai d’attente (sec)", 5, 60, 20)
threshold = st.sidebar.slider("Seuil décision (0–1)", 0.05, 0.95, 0.50, 0.01,
                              help="Seuil métier pour distinguer faible/modéré de élevé")

mode = st.sidebar.radio("Mode", ["Prédiction unitaire", "Batch CSV"], help="CE1 : parcours utilisateur simple")

# ==============================
# EN-TÊTE
# ==============================
st.title("🏦 Dashboard Scoring Crédit (P8)")
st.caption("Connecté à l’API du P7 (LightGBM sur Home Credit). Démo publique et inclusive (WCAG).")

with st.expander("🎯 Contexte & objectifs (1 min)", expanded=True):
    st.markdown("""
    **Objectif métier** : estimer le risque de défaut pour aider à la décision de crédit.  
    **Parcours** (CE1) :
    1) *Prédiction unitaire* : tester un profil type et lire la décision.
    2) *Batch CSV* : charger plusieurs profils, voir la distribution des scores et un scatter métier.
    **Lecture du score** : plus le score est élevé, plus le risque est important. Le **seuil** est réglable (barre latérale).
    """)

st.info("Accessibilité (WCAG) : titres explicites (2.4.2), texte redimensionnable (1.4.4), \
contraste par thème, info-bulles lisibles, aucune information codée uniquement par la couleur (1.4.1), \
contenus non textuels accompagnés d’un texte explicatif (1.1.1).")

# ==============================
# FONCTIONS UTILITAIRES
# ==============================
def call_api(payload):
    """
    Appelle l'API en POST avec le format attendu {"data": ...}
    Corrige l'erreur HTTP 422: Field required 'data'
    """
    url = f"{API_BASE}{endpoint_choice}"
    headers = {"Content-Type": "application/json"}
    wrapped = {"data": payload}
    try:
        t0 = time.time()
        resp = requests.post(url, headers=headers, data=json.dumps(wrapped), timeout=timeout_s)
        dt = time.time() - t0
        return resp, dt
    except requests.exceptions.RequestException as e:
        return e, None

def extract_probability(obj):
    """
    Récupère un score/proba quelle que soit la clé renvoyée par l'API.
    (Ton API renvoie 'default_probability' d'après tes captures.)
    """
    candidate = ["default_probability", "probability", "proba", "score", "prediction_proba"]
    if isinstance(obj, dict):
        for k in candidate:
            if k in obj:
                try:
                    return float(obj[k])
                except Exception:
                    pass
    return None

def label_from_prob(p, thr):
    if p is None:
        return "Résultat indisponible"
    return ("⚠️ Risque élevé" if p >= thr else "✅ Risque modéré/faible") + f" — score: {p:.3f} (seuil={thr:.2f})"

def altair_histogram(df, score_col, title):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(score_col, bin=alt.Bin(maxbins=30), title="Score / probabilité de défaut"),
            y=alt.Y("count()", title="Nombre de dossiers"),
            tooltip=[alt.Tooltip(score_col, title="Score")]
        )
        .properties(height=300, title=title)
    )

def altair_scatter(df, x_col, y_col, title, tooltip_cols):
    return (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            tooltip=[alt.Tooltip(c) for c in tooltip_cols]
        )
        .properties(height=320, title=title)
    )

# ==============================
# PRÉDICTION UNITÉ
# ==============================
st.markdown("## 🧮 Test de prédiction")

if mode == "Prédiction unitaire":
    col_form, col_result = st.columns([1.1, 1.2], gap="large")

    with col_form:
        st.subheader("Profil client (exemple)")
        amt_credit = st.number_input("Montant du crédit (AMT_CREDIT)", 0.0, 5_000_000.0, 150_000.0, 1_000.0)
        amt_annuity = st.number_input("Mensualité (AMT_ANNUITY)", 0.0, 200_000.0, 15_000.0, 500.0)
        amt_income = st.number_input("Revenu annuel du foyer (AMT_INCOME_TOTAL)", 0.0, 3_000_000.0, 120_000.0, 5_000.0)
        days_birth = st.number_input("Âge en jours négatifs (ex: -14000 ≈ 38 ans) [DAYS_BIRTH]", value=-14000)
        days_employed = st.number_input("Ancienneté en jours négatifs (ex: -3000) [DAYS_EMPLOYED]", value=-3000)
        ext1 = st.number_input("EXT_SOURCE_1 (0–1)", 0.0, 1.0, 0.55, 0.01)
        ext2 = st.number_input("EXT_SOURCE_2 (0–1)", 0.0, 1.0, 0.62, 0.01)
        ext3 = st.number_input("EXT_SOURCE_3 (0–1)", 0.0, 1.0, 0.58, 0.01)

        st.caption("📝 Si l’API attend plus de colonnes, utilisez le JSON personnalisé ci-dessous (remplace le formulaire).")
        with st.expander("➕ Coller un JSON personnalisé (facultatif)"):
            raw_json = st.text_area(
                "Collez ici un JSON complet conforme à votre API",
                height=160,
                placeholder='{"AMT_CREDIT": 150000, "AMT_ANNUITY": 15000, "AMT_INCOME_TOTAL": 120000, "DAYS_BIRTH": -14000, ...}'
            )

        ask = st.button("🚀 Obtenir la prédiction")

    with col_result:
        st.subheader("Résultat")
        if ask:
            payload = None
            if raw_json.strip():
                try:
                    payload = json.loads(raw_json)
                except Exception as e:
                    st.error(f"JSON invalide : {e}")
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
                    st.error(f"Erreur d’appel API : {resp}")
                else:
                    st.write(f"⏱️ Temps de réponse : {dt:.2f} s")
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except Exception:
                            st.warning("Réponse non JSON.")
                            st.text(resp.text[:1000])
                            data = None

                        if data is not None:
                            prob = extract_probability(data)
                            risk_text = data.get("risk_level") if isinstance(data, dict) else None
                            decision = data.get("prediction") if isinstance(data, dict) else None
                            st.success(label_from_prob(prob, threshold) + (f" — niveau: {risk_text}" if risk_text else ""))
                            if decision:
                                st.info(f"Interprétation : le modèle recommande **{decision}** pour ce dossier.")

                            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                    else:
                        st.error(f"Code HTTP {resp.status_code}")
                        st.text(resp.text[:1500])

        st.info("Lecture : en dessous du seuil, le profil est **modéré/faible** ; au-dessus, **élevé**. \
Ajustez le **seuil** dans la barre latérale selon l’appétence au risque.")

# ==============================
# BATCH CSV + 2 GRAPHIQUES (CE2)
# ==============================
if mode == "Batch CSV":
    st.subheader("📂 Prédictions en lot (CSV)")
    file = st.file_uploader("Importer un fichier CSV", type=["csv"], help="Colonnes conformes à votre API")
    show_preview = st.checkbox("Afficher un aperçu du CSV", value=True)

    if file is not None:
        df_in = pd.read_csv(file)
        if show_preview:
            st.dataframe(df_in.head(15), use_container_width=True)

        if st.button("🚀 Envoyer au modèle"):
            records = df_in.to_dict(orient="records")
            resp, dt = call_api(records)
            if isinstance(resp, Exception):
                st.error(f"Erreur d’appel API : {resp}")
            else:
                st.write(f"⏱️ Temps de réponse : {dt:.2f} s")
                if resp.status_code == 200:
                    try:
                        out = resp.json()
                        if isinstance(out, list):
                            df_out = pd.DataFrame(out)
                        elif isinstance(out, dict) and "predictions" in out:
                            df_out = pd.DataFrame(out["predictions"])
                        else:
                            df_out = pd.DataFrame([out])
                    except Exception as e:
                        st.error(f"Réponse non lisible : {e}")
                        df_out = None

                    if df_out is not None and len(df_out) > 0:
                        st.success("Prédictions reçues ✅")
                        st.dataframe(df_out.head(30), use_container_width=True)

                        # 1) Histogramme des scores (graphique interactif 1)
                        score_col = None
                        for c in df_out.columns:
                            if c.lower() in {"default_probability", "probability", "proba", "score", "prediction_proba"}:
                                score_col = c
                                break
                        if score_col:
                            st.markdown("### 📈 Distribution des scores (CE2, CE4)")
                            st.altair_chart(altair_histogram(df_out, score_col, "Distribution des probabilités de défaut"),
                                            use_container_width=True)
                        else:
                            st.info("Aucune colonne de score reconnue pour tracer la distribution.")

                        # 2) Scatter métier Score vs Montant crédit (graphique interactif 2)
                        st.markdown("### 🟢 Score vs Montant du crédit (CE2, CE4)")
                        if score_col and ("AMT_CREDIT" in df_in.columns):
                            # Rejoindre entrée et sortie si besoin
                            df_plot = df_in.copy()
                            df_plot[score_col] = df_out[score_col]
                            chart = altair_scatter(df_plot, "AMT_CREDIT", score_col,
                                                   "Relation Score / Montant du crédit",
                                                   ["AMT_CREDIT", score_col])
                            st.altair_chart(chart, use_container_width=True)
                        elif score_col:
                            # Fallback si AMT_CREDIT absent
                            df_tmp = df_out.copy()
                            df_tmp["index"] = range(len(df_tmp))
                            chart = altair_scatter(df_tmp, "index", score_col,
                                                   "Relation Score / Index (exemple)",
                                                   ["index", score_col])
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.info("Ajoutez une colonne de score pour afficher le scatter métier.")
                else:
                    st.error(f"Code HTTP {resp.status_code}")
                    st.text(resp.text[:1500])

# ==============================
# IMPORTANCE DES VARIABLES (CE4)
# ==============================
st.markdown("## 📊 Importance des variables")
st.markdown(
    "Illustration des variables qui contribuent le plus aux prédictions du modèle (P7). "
    "Le dashboard utilise automatiquement `feature_importance.csv` s'il est présent "
    "(colonnes **feature**, **importance**) ; sinon, une version illustrative est affichée."
)

def render_fi_chart(df):
    df = df.sort_values("importance", ascending=False).head(20)
    chart_fi = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("importance", title="Importance moyenne (gain LightGBM)"),
            y=alt.Y("feature", sort='-x', title="Variables"),
            tooltip=["feature", "importance"]
        )
        .properties(height=460, title="Top variables influentes")
    )
    st.altair_chart(chart_fi, use_container_width=True)

try:
    fi_df = pd.read_csv("feature_importance.csv")
    if {"feature", "importance"}.issubset(fi_df.columns):
        render_fi_chart(fi_df)
    else:
        st.info("`feature_importance.csv` n’a pas les colonnes attendues ('feature', 'importance'). \
Affichage d’un exemple illustratif.")
        raise FileNotFoundError
except Exception:
    demo_fi = pd.DataFrame({
        "feature": [
            "EXT_SOURCE_3","PAYMENT_RATE","EXT_SOURCE_2","AMT_CREDIT","DAYS_BIRTH",
            "EXT_SOURCE_1","AMT_ANNUITY","INCOME_CREDIT_PERC","CREDIT_TO_ANNUITY_RATIO","DAYS_EMPLOYED_RATIO"
        ],
        "importance": [520,480,450,430,410,370,340,320,300,280]
    })
    st.caption("Affichage illustratif (sans fichier). Ajoutez un vrai `feature_importance.csv` pour le remplacer.")
    render_fi_chart(demo_fi)

# ==============================
# VEILLE TECHNIQUE & NOTE MÉTHODO (supports livrables)
# ==============================
st.markdown("## 🔎 Veille technique & Note méthodologique (supports)")
st.markdown(
    "Cette section fournit des **modèles à compléter** pour votre livrable 2 (notebook de veille) "
    "et votre **note méthodologique** (livrable 3). Téléchargez, complétez, puis déposez sur la plateforme."
)

veille_md = """# Veille technique – P8
## 1. Sources récentes (3–5)
- [Auteur, année] Titre — source (blog/conference/journal). Lien:
- [Auteur, année] ...
## 2. Points clés (avec détails mathématiques)
- Méthode A : principe, équations, complexité, limites
- Méthode B : ...
## 3. Preuve de concept (PoC)
- Données utilisées:
- Baseline (classique) vs Nouvelle approche (récente):
- Protocole, métriques (AUC/PR/Recall@k...), résultats comparés
## 4. Conclusion
- Apports réels, risques, recommandations d’adoption
"""

note_md = """# Note méthodologique – P8 (10 pages max)
## 1. Démarche de modélisation (synthèse)
- Jeu de données, features, split, pipeline
## 2. Métrique d’évaluation & optimisation
- Métrique retenue (justification métier)
- Stratégie d’optimisation (CV, recherche d’hyperparamètres)
## 3. Interprétabilité globale & locale
- Importance des variables (globale), exemples locaux (ex: LIME/SHAP)
## 4. Limites & améliorations
- Biais potentiels, data drift, axes d’amélioration (features, seuil, calibration)
"""

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "📥 Télécharger le modèle Veille (Markdown)",
        data=veille_md.encode("utf-8"),
        file_name="modele_veille_P8.md",
        mime="text/markdown"
    )
with col_dl2:
    st.download_button(
        "📥 Télécharger la Note méthodologique (Markdown)",
        data=note_md.encode("utf-8"),
        file_name="modele_note_methodo_P8.md",
        mime="text/markdown"
    )

st.markdown("> **Rappel Livrables** : \
**1)** Dashboard déployé ; **2)** Notebook de veille (technique récente texte/image) ; \
**3)** Note méthodo (10 pages) ; **4)** Présentation (≤ 30 slides). \
Nommer : `Nom_Prénom_1_dashboard_mmaaaa`, `Nom_Prénom_2_notebook_veille_mmaaaa`, \
`Nom_Prénom_3_note_méthodologique_mmaaaa`, `Nom_Prénom_4_presentation_mmaaaa`.")

# ==============================
# PIED DE PAGE
# ==============================
st.markdown("---")
st.markdown(
    "**À propos** — Modèle LightGBM (P7), API Render, tableau de bord Streamlit (P8). "
    "Pensé pour un public non technique : parcours simple (CE1), au moins deux graphiques interactifs (CE2), "
    "lisibles et pertinents métier (CE3–CE4), critères WCAG clés (CE5), déployé sur le web (CE6)."
)
