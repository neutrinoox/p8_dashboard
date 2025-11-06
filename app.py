# app.py
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
        :focus {{ outline: 3px solid #4F46E5 !important; outline-offset: 2px; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# PARAMÈTRES API (P7)
# ==============================
API_BASE = "https://projet7-credit-scoring-api.onrender.com"
ENDPOINTS = ["/predict", "/predict_proba", "/inference", "/score"]
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
    **Parcours** :
    1) *Prédiction unitaire* : tester un profil type et lire la décision (avec un graphique Score vs Seuil).  
    2) *Batch CSV* : charger plusieurs profils, voir la distribution des scores et un scatter métier.  
    **Lecture du score** : plus le score est élevé, plus le risque est important. Le **seuil** est réglable (barre latérale).
    """)

st.info("Accessibilité (WCAG) : titres explicites (2.4.2), texte redimensionnable (1.4.4), "
        "contraste par thème, info-bulles lisibles, aucune information codée uniquement par la couleur (1.4.1), "
        "contenus non textuels accompagnés d’un texte explicatif (1.1.1).")

# ==============================
# OUTILS
# ==============================
def call_api(payload):
    """POST {"data": ...} → corrige l'erreur 422 (Field 'data' requis)."""
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
    """Extrait une proba quelle que soit la clé renvoyée par l’API."""
    for k in ["default_probability", "probability", "proba", "score", "prediction_proba"]:
        if isinstance(obj, dict) and k in obj:
            try:
                return float(obj[k])
            except Exception:
                pass
    return None

def label_from_prob(p, thr):
    if p is None:
        return "Résultat indisponible"
    return ("⚠️ Risque élevé" if p >= thr else "✅ Risque modéré/faible") + f" — score: {p:.3f} (seuil={thr:.2f})"

def normalize_predictions(out):
    """
    Convertit la réponse API en DataFrame et crée une colonne '__score__'
    robuste (default_probability / probability / proba / score / prediction_proba).
    """
    if isinstance(out, list):
        df = pd.DataFrame(out)
    elif isinstance(out, dict) and "predictions" in out and isinstance(out["predictions"], list):
        df = pd.DataFrame(out["predictions"])
    else:
        df = pd.DataFrame([out])

    df["__score__"] = None
    for cand in ["default_probability", "probability", "proba", "score", "prediction_proba"]:
        if cand in df.columns:
            df["__score__"] = pd.to_numeric(df[cand], errors="coerce")
            break
    return df

def chart_hist(df, score_col="__score__", title="Distribution des probabilités de défaut"):
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

def chart_scatter(df, x_col, y_col="__score__", title="Relation Score / Montant du crédit", tooltip_cols=None):
    if tooltip_cols is None:
        tooltip_cols = [x_col, y_col]
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

def chart_unit_gauge(prob, thr):
    """Graphique unitaire : barre horizontale [0..1] + règle verticale au seuil."""
    if prob is None:
        return None
    df_bar = pd.DataFrame({"start":[0.0], "end":[float(prob)]})
    base = alt.Chart(df_bar).mark_bar().encode(
        x=alt.X("end:Q", title="Score (0 → 1)", scale=alt.Scale(domain=[0,1])),
        tooltip=[alt.Tooltip("end:Q", title="Score")]
    ).properties(height=60)
    rule = alt.Chart(pd.DataFrame({"x":[float(thr)]})).mark_rule(strokeDash=[6,4]).encode(
        x="x:Q",
        tooltip=[alt.Tooltip("x:Q", title="Seuil")]
    )
    return base + rule

# ==============================
# PRÉDICTION UNITÉ + 1 GRAPHIQUE
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

                            # 🔹 Graphique 1 : Score vs Seuil (toujours visible en mode unitaire)
                            g1 = chart_unit_gauge(prob, threshold)
                            if g1 is not None:
                                st.altair_chart(g1, use_container_width=True)

                            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                    else:
                        st.error(f"Code HTTP {resp.status_code}")
                        st.text(resp.text[:1500])

        st.info("Lecture : en dessous du seuil, le profil est **modéré/faible** ; au-dessus, **élevé**. "
                "Ajustez le **seuil** dans la barre latérale selon l’appétence au risque.")

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
                        df_out = normalize_predictions(out)
                    except Exception as e:
                        st.error(f"Réponse non lisible : {e}")
                        df_out = None

                    if df_out is not None and len(df_out) > 0:
                        st.success("Prédictions reçues ✅")
                        st.dataframe(df_out.head(30), use_container_width=True)

                        # 🔹 Graphique 2a : Histogramme des scores (interactif)
                        if df_out["__score__"].notna().any():
                            st.markdown("### 📈 Distribution des scores (CE2, CE4)")
                            st.altair_chart(chart_hist(df_out, "__score__"), use_container_width=True)
                        else:
                            st.info("Aucune colonne de score reconnue pour tracer la distribution.")

                        # 🔹 Graphique 2b : Scatter Score vs Montant du crédit (ou index)
                        st.markdown("### 🟢 Score vs Montant du crédit (CE2, CE4)")
                        score_ok = df_out["__score__"].notna().any()
                        if score_ok and ("AMT_CREDIT" in df_in.columns):
                            df_plot = df_in.copy()
                            df_plot["__score__"] = df_out["__score__"]
                            st.altair_chart(chart_scatter(df_plot, "AMT_CREDIT"), use_container_width=True)
                        elif score_ok:
                            df_tmp = df_out.copy()
                            df_tmp["index"] = range(len(df_tmp))
                            st.altair_chart(chart_scatter(df_tmp, "index", title="Relation Score / Index (exemple)"),
                                            use_container_width=True)
                        else:
                            st.info("Ajoutez une colonne de score pour afficher le scatter métier.")
                else:
                    st.error(f"Code HTTP {resp.status_code}")
                    st.text(resp.text[:1500])

# ==============================
# PIED DE PAGE
# ==============================
st.markdown("---")
st.markdown(
    "**À propos** — Modèle LightGBM (P7), API Render, tableau de bord Streamlit (P8). "
    "Pensé pour un public non technique : parcours simple (CE1), au moins deux graphiques interactifs (CE2), "
    "lisibles et pertinents métier (CE3–CE4), critères WCAG clés (CE5), déployé sur le web (CE6)."
)

