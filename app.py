# app.py
import json
import time
import gc
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from typing import List, Optional

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="P8 – Dashboard Scoring Crédit", page_icon="📊", layout="wide")
alt.data_transformers.disable_max_rows()  # autorise de gros jeux dans les charts

API_MODEL_DEFAULT = "https://projet7-credit-scoring-api.onrender.com"
TRAIN_DATA_URL_DEFAULT = "https://sdz8rwt21sedumxt.public.blob.vercel-storage.com/train_data.csv"

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header(" Réglages & Mémoire")
    MODEL_URL = st.text_input("URL API Modèle (Render)", API_MODEL_DEFAULT)
    DATA_URL = st.text_input("URL TrainData (Vercel Blob)", TRAIN_DATA_URL_DEFAULT)

    # Anti-OOM: taille échantillon et colonnes à charger
    sample_max = st.number_input("Taille max de l’échantillon (lignes)", min_value=5_000, max_value=300_000, value=80_000, step=5_000)
    chunk_size = st.number_input("Taille des chunks (lignes)", min_value=50_000, max_value=500_000, value=150_000, step=10_000)
    usecols_raw = st.text_input("Colonnes à conserver (optionnel, séparées par des virgules)", "")
    threshold = st.slider("Seuil décision (0–1)", 0.05, 0.95, 0.50, 0.01)
    font_scale = st.slider("Taille du texte (%)", 90, 170, 110, 5)

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{ font-size: {font_scale}%; }}
      :focus {{ outline: 3px solid #4F46E5 !important; outline-offset: 2px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("P8 – Dashboard Scoring Crédit")
st.caption("Charge un échantillon mémoire-safe depuis Vercel Blob, interroge l’API Render, compare par ID, explore et explique.")

# ==============================
# OUTILS CHARGEMENT MEMOIRE-SAFE
# ==============================
def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    for c in num:
        col = df[c]
        if pd.api.types.is_float_dtype(col):
            df[c] = pd.to_numeric(col, downcast="float")  # float32
        elif pd.api.types.is_integer_dtype(col):
            df[c] = pd.to_numeric(col, downcast="integer")  # int32/int16
    return df

@st.cache_data(show_spinner=True)
def load_head(url: str, nrows: int = 5000) -> pd.DataFrame:
    dfh = pd.read_csv(url, nrows=nrows)
    if "Unnamed: 0" in dfh.columns:
        dfh = dfh.drop(columns=["Unnamed: 0"])
    return _downcast_numeric(dfh)

@st.cache_data(show_spinner=True)
def count_rows_stream(url: str, chunk_size: int) -> int:
    total = 0
    for chunk in pd.read_csv(url, chunksize=chunk_size):
        total += len(chunk)
        del chunk
        gc.collect()
    return total

@st.cache_data(show_spinner=True)
def load_sample_stream(
    url: str,
    sample_max: int,
    chunk_size: int,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    2 passes mémoire-safe :
    - pass 1 : compte les lignes totales
    - pass 2 : échantillon proportionnel par chunk pour atteindre sample_max
    """
    # Pass 1: count
    total = 0
    for chunk in pd.read_csv(url, chunksize=chunk_size, usecols=usecols):
        total += len(chunk)
        del chunk
        gc.collect()

    if total == 0:
        return pd.DataFrame()

    # Pass 2: proportional sampling
    remain = sample_max
    taken_frames = []
    seen = 0
    for chunk in pd.read_csv(url, chunksize=chunk_size, usecols=usecols):
        n = len(chunk)
        seen += n
        # nombre à prendre dans ce chunk (proportionnel)
        to_take = min(n, max(0, math.ceil(sample_max * (n / total))))
        if to_take > 0:
            sub = chunk.sample(n=to_take, random_state=42)
            if "Unnamed: 0" in sub.columns:
                sub = sub.drop(columns=["Unnamed: 0"])
            taken_frames.append(sub)
        del chunk
        gc.collect()

    if not taken_frames:
        return pd.DataFrame()

    df = pd.concat(taken_frames, ignore_index=True)
    # si on a prélevé un peu trop, on recadre
    if len(df) > sample_max:
        df = df.sample(sample_max, random_state=42).reset_index(drop=True)

    df = _downcast_numeric(df)
    return df

# ==============================
# CHARGEMENT DATA
# ==============================
with st.spinner(" Préparation des métadonnées…"):
    head = load_head(DATA_URL)
    all_columns = head.columns.tolist()

# ID heuristique
ID_CANDIDATES = ["SK_ID_CURR", "client_id", "CLIENT_ID", "ID", "id"]
id_col_default = next((c for c in ID_CANDIDATES if c in all_columns), all_columns[0])

with st.expander(" Pré-sélection des colonnes (optionnel)"):
    st.markdown("Restriction possible des colonnes pour alléger la mémoire. laisse vider pour tout charger.")
    st.code("Exemple : SK_ID_CURR,AMT_CREDIT,AMT_ANNUITY,AMT_INCOME_TOTAL,EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3")
    id_col = st.selectbox("Colonne ID", options=all_columns, index=all_columns.index(id_col_default))
    if usecols_raw.strip():
        usecols = [c.strip() for c in usecols_raw.split(",") if c.strip() in all_columns]
        if id_col not in usecols:
            usecols = [id_col] + usecols
    else:
        usecols = None  # toutes colonnes

with st.spinner("Chargement mémoire-safe de l’échantillon…"):
    try:
        df = load_sample_stream(DATA_URL, sample_max=int(sample_max), chunk_size=int(chunk_size), usecols=usecols)
    except Exception as e:
        st.error(f"Échec du chargement échantillonné : {e}")
        st.stop()

if df.empty:
    st.error("Le dataset est vide après chargement. Vérifiez l’URL ou les colonnes sélectionnées.")
    st.stop()

st.success(f"Échantillon chargé : {df.shape[0]:,} lignes × {df.shape[1]:,} colonnes")
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# ==============================
# OUTILS API (modèle)
# ==============================
def call_predict(model_url: str, features: dict, cid: Optional[str] = None, timeout_s: int = 25):
    url = model_url.rstrip("/") + "/predict"
    payload = {"data": features}
    if cid is not None:
        payload["client_id"] = str(cid)
    headers = {"Content-Type": "application/json"}
    t0 = time.time()
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    dt = time.time() - t0
    return r, dt

def try_fetch_shap(model_url: str, cid: str, features: dict, timeout_s: int = 25):
    # GET /shap?client_id=
    try:
        url = model_url.rstrip("/") + "/shap"
        r = requests.get(url, params={"client_id": cid}, timeout=timeout_s)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # POST /shap
    try:
        url = model_url.rstrip("/") + "/shap"
        payload = {"data": features, "client_id": str(cid)}
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ==============================
# RECHERCHE CLIENT & PRÉDICTION
# ==============================
st.markdown("##Rechercher un client")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    query_id = st.text_input(f"ID client ({id_col})", "")
with c2:
    pick = st.selectbox("…ou sélectionner un ID existant", options=["—"] + list(map(str, df[id_col].head(2000))), index=0)
with c3:
    cargar = st.button("Charger le client", use_container_width=True)

client_row = None
if cargar:
    use_id = query_id.strip() or (pick if pick != "—" else "")
    if not use_id:
        st.warning("Saisis un ID client ou choisis-en un dans la liste.")
    else:
        sub = df[df[id_col].astype(str) == str(use_id)]
        if sub.empty:
            st.error(f"Aucun client avec {id_col} = {use_id}")
        else:
            client_row = sub.iloc[0]
            st.success(f"Client {use_id} chargé.")
            st.dataframe(client_row.to_frame().T, use_container_width=True)

# ==============================
# ACTION : PRÉDIRE
# ==============================
st.markdown("---")
st.markdown("## 🧮 Prédiction du risque")

colp1, colp2 = st.columns([1, 1])
with colp1:
    st.caption("Le payload enverra **les colonnes numériques** disponibles pour ce client (les autres sont ignorées).")
with colp2:
    predict_now = st.button("Calculer le risque (API Render)", use_container_width=True)

pred_result = None
pred_time = None
features_used = {}
if predict_now:
    if client_row is None:
        st.warning("Charge d’abord un client.")
    else:
        # construit un payload compact
        feats = {}
        for c in num_cols:
            val = client_row[c]
            if pd.isna(val):
                continue
            feats[c] = float(val)
        features_used = feats

        try:
            r, dt = call_predict(MODEL_URL, feats, cid=str(client_row[id_col]))
            pred_time = dt
            if r.status_code == 200:
                pred_result = r.json()
            else:
                st.error(f"Erreur API ({r.status_code}) : {r.text[:500]}")
        except Exception as e:
            st.error(f"Erreur d’appel API : {e}")

if pred_result:
    colr1, colr2, colr3 = st.columns(3)
    prob = pred_result.get("default_probability")
    risk = pred_result.get("risk_level")
    decision = pred_result.get("prediction")

    with colr1:
        st.success(f"**Score** : {prob:.4f}" if isinstance(prob, (int,float)) else f"Score indisponible")
        if pred_time is not None:
            st.caption(f"⏱️ {pred_time:.2f}s")
    with colr2:
        st.info(f"**Décision** : {decision}")
    with colr3:
        st.warning(f"**Niveau** : {risk}")

    # Jauge linéaire vs seuil
    if isinstance(prob, (int,float)):
        df_bar = pd.DataFrame({"score":[float(prob)]})
        base = alt.Chart(df_bar).mark_bar().encode(
            x=alt.X("score:Q", title="Score (0 → 1)", scale=alt.Scale(domain=[0,1])),
            tooltip=[alt.Tooltip("score:Q", title="Score")]
        ).properties(height=50)
        rule = alt.Chart(pd.DataFrame({"x":[float(threshold)]})).mark_rule(strokeDash=[6,4]).encode(x="x:Q")
        st.altair_chart(base + rule, use_container_width=True)

# ==============================
# SHAP (
# ==============================
st.markdown("##  Interprétation locale (SHAP)")
if pred_result and client_row is not None:
    with st.spinner("Tentative de récupération des SHAP…"):
        shap_res = try_fetch_shap(MODEL_URL, str(client_row[id_col]), features_used)
    if shap_res and isinstance(shap_res, dict) and "shap_values" in shap_res:
        shap_series = pd.Series(shap_res["shap_values"]).sort_values(key=lambda x: x.abs(), ascending=False)
        topk = st.slider("Top features à afficher", 5, min(30, len(shap_series)), 12)
        st.bar_chart(shap_series.head(topk).rename("contribution"))
        st.caption("Astuce : passe la souris pour voir la contribution de chaque feature. (Si indisponible, expose un endpoint `/shap` côté API.)")
    else:
        st.info("SHAP non disponible (aucun endpoint `/shap` accessible).")

# ==============================
# EXPLORATION : 3 graphiques interactifs
# ==============================
st.markdown("---")
st.markdown("## 📊 Exploration interactive (client vs population)")

if len(num_cols) < 2:
    st.warning("Il faut au moins 2 colonnes numériques pour les graphiques.")
else:
    # 1) Scatter multi-variables
    st.markdown("### 1) 🌐 Nuage de points configurable")
    cx, cy = st.columns(2)
    with cx:
        x_var = st.selectbox("Axe X", options=num_cols, index=num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 0)
    with cy:
        y_var = st.selectbox("Axe Y", options=num_cols, index=num_cols.index("AMT_CREDIT") if "AMT_CREDIT" in num_cols else 1)

    c2a, c2b = st.columns(2)
    with c2a:
        color_var = st.selectbox("Couleur (optionnel)", options=["(aucune)"] + num_cols, index=0)
    with c2b:
        size_var = st.selectbox("Taille (optionnel)", options=["(aucune)"] + num_cols, index=0)

    base = alt.Chart(df).mark_circle(opacity=0.35).encode(
        x=alt.X(f"{x_var}:Q", title=x_var),
        y=alt.Y(f"{y_var}:Q", title=y_var),
        tooltip=[x_var, y_var] + ([color_var] if color_var != "(aucune)" else [])
    )
    if color_var != "(aucune)":
        base = base.encode(color=alt.Color(f"{color_var}:Q", scale=alt.Scale(scheme="turbo")))
    if size_var != "(aucune)":
        base = base.encode(size=alt.Size(f"{size_var}:Q", legend=None))

    zoom = alt.selection_interval(bind="scales")
    chart_scatter = base.add_selection(zoom).properties(height=380)

    # point client
    if client_row is not None and pd.notna(client_row.get(x_var)) and pd.notna(client_row.get(y_var)):
        client_point = pd.DataFrame([{x_var: float(client_row[x_var]), y_var: float(client_row[y_var])}])
        highlight = alt.Chart(client_point).mark_point(size=220, filled=True).encode(
            x=f"{x_var}:Q", y=f"{y_var}:Q"
        )
        chart_scatter = chart_scatter + highlight

    st.altair_chart(chart_scatter, use_container_width=True)

    # 2) Histogramme + percentile + z-score
    st.markdown("### 2) 📈 Distribution d’une variable (avec position du client)")
    hv1, hv2, hv3 = st.columns([2,1,1])
    with hv1:
        hist_var = st.selectbox("Variable à explorer", options=num_cols, index=num_cols.index("AMT_CREDIT") if "AMT_CREDIT" in num_cols else 0)
    client_val = None
    if client_row is not None and hist_var in df.columns:
        try:
            client_val = float(client_row[hist_var])
        except Exception:
            client_val = None

    base_hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{hist_var}:Q", bin=alt.Bin(maxbins=40), title=hist_var),
        y=alt.Y("count()", title="Fréquence"),
        tooltip=[alt.Tooltip(f"{hist_var}:Q", title=hist_var)]
    ).properties(height=240)

    if client_val is not None and not np.isnan(client_val):
        rule = alt.Chart(pd.DataFrame({hist_var: [client_val]})).mark_rule(strokeDash=[6,4]).encode(
            x=f"{hist_var}:Q", tooltip=[alt.Tooltip(f"{hist_var}:Q", title="Valeur client")]
        )
        st.altair_chart(base_hist + rule, use_container_width=True)
    else:
        st.altair_chart(base_hist, use_container_width=True)

    with hv2:
        if client_val is not None and not np.isnan(client_val):
            s = df[hist_var].dropna().sort_values()
            pct = (s.searchsorted(client_val, side="right") / len(s)) * 100
            st.metric("Percentile du client", f"{pct:.1f} %")
        else:
            st.metric("Percentile du client", "—")
    with hv3:
        if client_val is not None and not np.isnan(client_val):
            m, sd = df[hist_var].mean(), df[hist_var].std()
            z = (client_val - m) / sd if sd and sd > 0 else np.nan
            st.metric("z-score du client", f"{z:+.2f}" if not np.isnan(z) else "—")
        else:
            st.metric("z-score du client", "—")

    # 3) Heatmap corrélations
    st.markdown("### 3) Carte de corrélations (sélection)")
    sel_vars = st.multiselect(
        "Variables numériques",
        options=num_cols,
        default=[v for v in ["AMT_CREDIT","AMT_ANNUITY","AMT_INCOME_TOTAL","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"] if v in num_cols][:6],
    )
    if len(sel_vars) >= 2:
        corr = df[sel_vars].corr(numeric_only=True)
        corr_df = corr.reset_index().melt("index")
        corr_df.columns = ["x","y","value"]
        heat = alt.Chart(corr_df).mark_rect().encode(
            x=alt.X("x:N", sort=sel_vars, title=""),
            y=alt.Y("y:N", sort=sel_vars, title=""),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue"), title="corr"),
            tooltip=["x:N","y:N", alt.Tooltip("value:Q", format=".2f")]
        ).properties(height=360)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Sélectionne au moins 2 variables pour la heatmap.")

# ==============================
# NOTES
# ==============================
with st.expander(" Notes perf & mémoire"):
    st.markdown(
        """
- **Chargement en 2 passes** : compte les lignes, puis échantillonne proportionnellement par chunk → **pas de pic mémoire**.
- **Downcast** des numériques en *float32/int32* pour réduire l’empreinte RAM.
- **Réduis les colonnes** (champ *Colonnes à conserver*) si besoin (ID + variables clés).
- **Augmente/réduis** *Taille des chunks* et *Taille de l’échantillon* selon ta machine.
- **Altair** est configuré pour accepter de gros échantillons sans couper les données.
        """
    )

