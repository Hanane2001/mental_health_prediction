import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.exploration import SOURCE, exporter_data

st.set_page_config(
    page_title="Mental Health Risk — Predictor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

    /* ---------- RESET / BASE ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-feature-settings: 'cv02','cv03','cv04','cv11','tnum';
    }
    .stApp {
        background: #0E1117;
        color: #E6EDF3;
    }
    .main .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem;
        max-width: 1280px;
    }

    /* Cacher branding Streamlit */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Icône Material réutilisable */
    .material-symbols-rounded {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
        line-height: 1;
    }

    /* ---------- TOP NAV ---------- */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem;
        background: #161B22;
        border-radius: 12px;
        border: 1px solid #21262D;
        margin-bottom: 2rem;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 700;
        font-size: 1.05rem;
        color: #E6EDF3;
        letter-spacing: -0.01em;
    }
    .nav-logo {
        width: 36px; height: 36px;
        border-radius: 9px;
        background: linear-gradient(135deg, #6366F1 0%, #A78BFA 100%);
        display: flex; align-items: center; justify-content: center;
        color: white;
    }
    .nav-logo .material-symbols-rounded { font-size: 1.3rem; }
    .nav-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.3rem 0.7rem;
        background: rgba(99, 102, 241, 0.12);
        color: #A78BFA;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 6px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ---------- PAGE HEADER ---------- */
    .page-header { margin-bottom: 2rem; }
    .page-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8B949E;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #E6EDF3;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 0.5rem 0;
    }
    .page-desc {
        font-size: 1rem;
        color: #8B949E;
        line-height: 1.6;
        max-width: 680px;
        margin: 0;
    }

    /* ---------- KPI CARDS ---------- */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #30363D;
        background: #1A2029;
    }
    .kpi-icon {
        width: 38px; height: 38px;
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 0.9rem;
    }
    .kpi-icon .material-symbols-rounded { font-size: 1.25rem; }
    .kpi-icon-indigo { background: rgba(99, 102, 241, 0.15); color: #818CF8; }
    .kpi-icon-violet { background: rgba(167, 139, 250, 0.15); color: #A78BFA; }
    .kpi-icon-teal   { background: rgba(45, 212, 191, 0.12); color: #2DD4BF; }
    .kpi-icon-amber  { background: rgba(251, 191, 36, 0.12); color: #FBBF24; }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #E6EDF3;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #8B949E;
        font-weight: 500;
        margin: 0.35rem 0 0 0;
    }

    /* ---------- SECTION ---------- */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #8B949E;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 2rem 0 1rem 0;
    }

    /* ---------- FORM SECTION ---------- */
    .form-section-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #A78BFA;
        margin: 1.25rem 0 0.75rem 0;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid #21262D;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .form-section-title .material-symbols-rounded { font-size: 1.1rem; }

    /* ---------- INPUTS ---------- */
    .stNumberInput input, .stTextInput input {
        border-radius: 8px !important;
        border: 1px solid #30363D !important;
        background: #0D1117 !important;
        padding: 0.55rem 0.75rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #E6EDF3 !important;
        transition: all 0.15s ease !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #30363D !important;
        background: #0D1117 !important;
        font-size: 0.9rem !important;
        color: #E6EDF3 !important;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #B0B8C1 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        margin-bottom: 0.25rem !important;
    }
    .stSlider > div > div > div > div {
        background: #6366F1 !important;
    }
    /* Streamlit dark overrides */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #6366F1 !important;
        border: 2px solid #0D1117 !important;
    }

    /* ---------- BUTTON ---------- */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white !important;
        border: none;
        border-radius: 9px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: -0.01em;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid #21262D;
        padding: 0;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 0.75rem 1.15rem;
        font-weight: 600;
        font-size: 0.88rem;
        color: #8B949E;
        background: transparent;
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #A78BFA !important;
        background: transparent !important;
        border-bottom: 2px solid #6366F1 !important;
    }

    /* ---------- RESULT CARD ---------- */
    .result-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 14px;
        padding: 2rem;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
    }
    .result-card.risk-low::before    { background: #2DD4BF; }
    .result-card.risk-medium::before { background: #FBBF24; }
    .result-card.risk-high::before   { background: #F87171; }

    .result-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        color: #8B949E;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .result-eyebrow .material-symbols-rounded { font-size: 1rem; }
    .result-value {
        font-size: 2.6rem;
        font-weight: 800;
        color: #E6EDF3;
        letter-spacing: -0.04em;
        line-height: 1;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .result-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-low    { background: #2DD4BF; box-shadow: 0 0 12px rgba(45, 212, 191, 0.5); }
    .dot-medium { background: #FBBF24; box-shadow: 0 0 12px rgba(251, 191, 36, 0.5); }
    .dot-high   { background: #F87171; box-shadow: 0 0 12px rgba(248, 113, 113, 0.5); }

    .result-desc {
        font-size: 0.92rem;
        color: #8B949E;
        line-height: 1.65;
        max-width: 560px;
    }

    /* ---------- PROBABILITY BARS ---------- */
    .prob-item { margin-bottom: 1rem; }
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
    }
    .prob-name {
        font-weight: 500;
        color: #B0B8C1;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .prob-value {
        font-weight: 700;
        color: #E6EDF3;
        font-variant-numeric: tabular-nums;
    }
    .prob-track {
        height: 6px;
        background: #21262D;
        border-radius: 3px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .fill-low    { background: #2DD4BF; }
    .fill-medium { background: #FBBF24; }
    .fill-high   { background: #F87171; }

    /* ---------- INFO BANNER ---------- */
    .info-banner {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        display: flex;
        gap: 0.85rem;
        align-items: flex-start;
        margin-bottom: 1.5rem;
    }
    .info-banner .material-symbols-rounded {
        color: #818CF8;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .info-text {
        font-size: 0.88rem;
        color: #B0B8C1;
        line-height: 1.6;
    }

    /* ---------- EMPTY STATE ---------- */
    .empty-state {
        background: #161B22;
        border: 1px dashed #30363D;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
    }
    .empty-state .material-symbols-rounded {
        font-size: 2.5rem;
        color: #484F58;
        margin-bottom: 1rem;
    }
    .empty-title {
        font-size: 1rem;
        font-weight: 600;
        color: #E6EDF3;
        margin: 0 0 0.5rem 0;
    }
    .empty-desc {
        font-size: 0.88rem;
        color: #8B949E;
        line-height: 1.6;
        max-width: 420px;
        margin: 0 auto;
    }

    /* ---------- FEATURE CARD ---------- */
    .feature-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    .feature-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #E6EDF3;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .feature-title .material-symbols-rounded {
        color: #A78BFA;
        font-size: 1.15rem;
    }
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .feature-list li {
        font-size: 0.87rem;
        color: #B0B8C1;
        padding: 0.55rem 0;
        border-bottom: 1px solid #21262D;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .feature-list li:last-child { border-bottom: none; }
    .feature-list strong { color: #E6EDF3; font-weight: 600; }
    .feature-list .material-symbols-rounded {
        font-size: 0.95rem;
        color: #2DD4BF;
    }

    /* ---------- WARNING ---------- */
    .warning-banner {
        background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 0.85rem;
        color: #FBBF24;
        line-height: 1.6;
        margin-top: 1.5rem;
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
    }
    .warning-banner .material-symbols-rounded {
        flex-shrink: 0;
        font-size: 1.2rem;
    }

    /* ---------- PLOTLY DARK TUNING ---------- */
    .js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def icon(name, size="1rem", color=None):
    style = f"font-size:{size};"
    if color:
        style += f"color:{color};"
    return f'<span class="material-symbols-rounded" style="{style}">{name}</span>'

MODEL_DIR  = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
NAME_PATH  = MODEL_DIR / "best_model_name.pkl"

@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None, None
    pipe = joblib.load(MODEL_PATH)
    name = joblib.load(NAME_PATH) if NAME_PATH.exists() else "Best Model"
    return pipe, name


@st.cache_data
def load_raw_data():
    return exporter_data(SOURCE)

def predict_from_input(pipe, row: dict):
    df = pd.DataFrame([row])
    pred = pipe.predict(df)[0]

    if hasattr(pipe, "predict_proba"):
        probs_arr = pipe.predict_proba(df)[0]
        classes = pipe.named_steps["model"].classes_
    else:
        scores = pipe.decision_function(df)[0]
        exp = np.exp(scores - np.max(scores))
        probs_arr = exp / exp.sum()
        classes = pipe.named_steps["model"].classes_

    return str(pred), {str(c): float(p) for c, p in zip(classes, probs_arr)}

st.markdown(
    f"""
    <div class="top-nav">
        <div class="nav-brand">
            <div class="nav-logo">{icon('psychology', '1.3rem')}</div>
            <span>MindRisk</span>
        </div>
        <div class="nav-badge">ML Predictor</div>
    </div>
    """,
    unsafe_allow_html=True,
)

pipe, model_name = load_pipeline()

if pipe is None:
    st.error(
        "Modèle non trouvé dans `models/`. "
        "Exécutez `save_best_model(models, res_df)` dans le notebook `04_train.ipynb`."
    )
    st.stop()

try:
    df_raw = load_raw_data()
except Exception as e:
    st.warning(f"Données brutes indisponibles pour l'EDA : {e}")
    df_raw = None

st.markdown(
    """
    <div class="page-header">
        <div class="page-eyebrow">Dashboard</div>
        <h1 class="page-title">Mental Health Risk Predictor</h1>
        <p class="page-desc">
            Évaluez le niveau de risque de santé mentale d'un étudiant
            à partir de 16 indicateurs comportementaux, académiques et psychosociaux.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

n_samples = len(df_raw) if df_raw is not None else 2000
st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-icon kpi-icon-indigo">{icon('groups', '1.25rem')}</div>
            <p class="kpi-value">{n_samples:,}</p>
            <p class="kpi-label">Étudiants analysés</p>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon kpi-icon-violet">{icon('category', '1.25rem')}</div>
            <p class="kpi-value">16</p>
            <p class="kpi-label">Features du modèle</p>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon kpi-icon-teal">{icon('stacked_bar_chart', '1.25rem')}</div>
            <p class="kpi-value">3</p>
            <p class="kpi-label">Classes de risque</p>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon kpi-icon-amber">{icon('model_training', '1.25rem')}</div>
            <p class="kpi-value" style="font-size:1.15rem;">{model_name}</p>
            <p class="kpi-label">Modèle actif</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_eda, tab_about = st.tabs(
    ["Prédiction", "Analyse des données", "À propos"]
)

with tab_pred:
    col_form, col_result = st.columns([1.15, 1], gap="large")

    with col_form:
        with st.form("pred_form", clear_on_submit=False):
            st.markdown(
                f'<div class="form-section-title">{icon("person")} Profil de l\'étudiant</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Âge", 17, 30, 21, 1)
            with c2:
                gender = st.selectbox("Genre", ["Male", "Female", "Non-binary"])

            c1, c2 = st.columns(2)
            with c1:
                year = st.number_input("Année d'étude", 1, 5, 2, 1)
            with c2:
                part_time = st.selectbox("Job temps partiel", ["No", "Yes"])

            c1, c2 = st.columns(2)
            with c1:
                prev = st.selectbox("Antécédents mentaux", ["No", "Yes"])
            with c2:
                phys = st.selectbox("Activité physique", ["Low", "Moderate", "High"])

            st.markdown(
                f'<div class="form-section-title">{icon("schedule")} Habitudes & lifestyle</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                sleep_h = st.slider("Sommeil (h/nuit)", 3.0, 10.0, 7.0, 0.1)
                study_h = st.slider("Étude (h/semaine)", 2.0, 40.0, 20.0, 0.5)
                gpa = st.slider("GPA", 1.5, 4.0, 2.6, 0.01)
                att_rate = st.slider("Présence (%)", 54.0, 100.0, 85.0, 0.1)
            with c2:
                social = st.slider("Réseaux sociaux (h/j)", 2.0, 10.0, 6.0, 0.1)
                screen = st.slider("Temps écran (h/j)", 3.0, 14.0, 8.5, 0.1)
                fin_stress = st.slider("Stress financier", 1.0, 10.0, 6.0, 0.1)
                fam_supp = st.slider("Soutien familial", 1.0, 10.0, 6.5, 0.1)

            diet = st.selectbox("Qualité alimentaire", ["Poor", "Average", "Good"])

            submitted = st.form_submit_button(
                "Analyser le risque", use_container_width=True
            )

    with col_result:
        if not submitted:
            st.markdown(
                f"""
                <div class="empty-state">
                    {icon('insights', '2.5rem', '#484F58')}
                    <p class="empty-title">En attente d'analyse</p>
                    <p class="empty-desc">
                        Remplissez le formulaire ci-contre puis cliquez sur
                        <strong>Analyser le risque</strong> pour obtenir une prédiction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            input_row = {
                "Age": age, "Gender": gender, "Year_of_Study": year,
                "Sleep_Hours": sleep_h, "Study_Hours_Per_Week": study_h,
                "GPA": gpa, "Social_Media_Hours": social,
                "Screen_Time_Hours": screen, "Physical_Activity": phys,
                "Financial_Stress": fin_stress, "Family_Support": fam_supp,
                "Attendance_Rate": att_rate, "Part_Time_Job": part_time,
                "Diet_Quality": diet, "Previous_Mental_Health_History": prev,
            }

            pred, probs = predict_from_input(pipe, input_row)
            p_low = pred.lower()

            if p_low == "low":
                cls, dot, ic = "risk-low", "dot-low", "verified_user"
                desc = ("Les indicateurs suggèrent un risque faible. "
                        "Continuez à maintenir de bonnes habitudes.")
            elif p_low == "medium":
                cls, dot, ic = "risk-medium", "dot-medium", "warning"
                desc = ("Risque modéré détecté. "
                        "Une attention préventive est recommandée.")
            else:
                cls, dot, ic = "risk-high", "dot-high", "error"
                desc = ("Risque élevé détecté. "
                        "Une intervention professionnelle est recommandée.")

            st.markdown(
                f"""
                <div class="result-card {cls}">
                    <div class="result-eyebrow">{icon(ic, '1rem')} Niveau de risque prédit</div>
                    <div class="result-value">
                        <span class="result-dot {dot}"></span>
                        {pred.capitalize()}
                    </div>
                    <p class="result-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-label">Probabilités par classe</div>',
                unsafe_allow_html=True,
            )

            order = ["Low", "Medium", "High"]
            proba = {}
            for k, v in probs.items():
                key = str(k)
                if key in ("0", "1", "2"):
                    key = {"0": "Low", "1": "Medium", "2": "High"}[key]
                proba[key] = v

            html = ""
            for label in order:
                pct = proba.get(label, 0) * 100
                fill_cls = f"fill-{label.lower()}"
                html += f"""
                <div class="prob-item">
                    <div class="prob-header">
                        <span class="prob-name">{label}</span>
                        <span class="prob-value">{pct:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill {fill_cls}" style="width:{pct}%;"></div>
                    </div>
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label">Profil de l\'étudiant</div>',
                unsafe_allow_html=True,
            )

            radar_feats = ["Sleep_Hours", "Study_Hours_Per_Week", "GPA",
                           "Attendance_Rate", "Family_Support", "Physical_Activity"]
            normalizers = {
                "Sleep_Hours": (3, 10),
                "Study_Hours_Per_Week": (2, 40),
                "GPA": (1.5, 4.0),
                "Attendance_Rate": (54, 100),
                "Family_Support": (1, 10),
                "Physical_Activity": (0, 2),
            }
            phys_map = {"Low": 0, "Moderate": 1, "High": 2}
            vals = []
            for f in radar_feats:
                v = input_row[f]
                if f == "Physical_Activity":
                    v = phys_map[v]
                lo, hi = normalizers[f]
                vals.append((v - lo) / (hi - lo))

            labels_radar = ["Sommeil", "Étude", "GPA", "Présence", "Soutien", "Activité"]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels_radar + [labels_radar[0]],
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.18)",
                line=dict(color="#818CF8", width=2),
                marker=dict(size=6, color="#A78BFA"),
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        gridcolor="rgba(255,255,255,0.08)",
                        tickfont=dict(size=9, color="#8B949E"),
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=10, color="#B0B8C1"),
                        gridcolor="rgba(255,255,255,0.06)",
                    ),
                    bgcolor="rgba(22,27,34,0.5)",
                ),
                showlegend=False,
                height=300,
                margin=dict(l=50, r=50, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_eda:
    if df_raw is None:
        st.warning("Données brutes indisponibles — vérifiez `data/raw/student_mental_health.csv`.")
    else:
        colormap = {"Low": "#2DD4BF", "Medium": "#FBBF24", "High": "#F87171"}

        LAYOUT_DARK = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B0B8C1", family="Inter"),
            xaxis=dict(tickfont=dict(color="#8B949E"),
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(color="#8B949E"),
                       gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(font=dict(size=11, color="#B0B8C1")),
        )

        st.markdown(
            '<div class="section-label">Vue d\'ensemble</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            tc = df_raw["Mental_Health_Risk"].value_counts().reset_index()
            tc.columns = ["Risque", "Nombre"]
            fig = px.bar(
                tc, x="Risque", y="Nombre",
                color="Risque", color_discrete_map=colormap, text="Nombre",
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(
                title=dict(text="Distribution de la cible",
                           font=dict(size=13, color="#E6EDF3")),
                showlegend=False, height=340,
                margin=dict(l=10, r=10, t=40, b=10),
                **LAYOUT_DARK,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            gc = df_raw["Gender"].value_counts().reset_index()
            gc.columns = ["Genre", "Nombre"]
            fig = px.pie(
                gc, names="Genre", values="Nombre", hole=.6,
                color_discrete_sequence=["#6366F1", "#A78BFA", "#2DD4BF"],
            )
            fig.update_layout(
                title=dict(text="Répartition par genre",
                           font=dict(size=13, color="#E6EDF3")),
                height=340,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=11, color="#B0B8C1")),
                font=dict(color="#B0B8C1", family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="section-label">Analyse par variable</div>',
            unsafe_allow_html=True,
        )

        num_cols = ["Sleep_Hours", "Study_Hours_Per_Week", "GPA",
                    "Social_Media_Hours", "Financial_Stress", "Attendance_Rate",
                    "Screen_Time_Hours", "Family_Support"]
        selected = st.selectbox("Choisir une variable :", num_cols)

        fig = px.histogram(
            df_raw, x=selected, color="Mental_Health_Risk",
            nbins=30, barmode="overlay", opacity=.8,
            color_discrete_map=colormap,
        )
        fig.update_layout(
            title=dict(text=f"Distribution de {selected}",
                       font=dict(size=13, color="#E6EDF3")),
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            **LAYOUT_DARK,
        )
        fig.update_layout(legend_title="Risque")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            df_raw, x="Mental_Health_Risk", y=selected,
            color="Mental_Health_Risk", color_discrete_map=colormap,
        )
        fig.update_layout(
            title=dict(text=f"{selected} par classe de risque",
                       font=dict(size=13, color="#E6EDF3")),
            height=340, showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            **LAYOUT_DARK,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_about:
    st.markdown(
        f"""
        <div class="info-banner">
            {icon('lightbulb', '1.3rem', '#818CF8')}
            <div class="info-text">
                Ce dashboard illustre un pipeline ML complet : exploration, preprocessing,
                entraînement de 3 classifieurs et sélection du meilleur modèle (F1-macro).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            f"""
            <div class="feature-card">
                <p class="feature-title">{icon('dataset')} Dataset</p>
                <ul class="feature-list">
                    <li>{icon('check_circle')}<span><strong>2 000</strong> étudiants simulés</span></li>
                    <li>{icon('check_circle')}<span><strong>16</strong> features + 1 cible</span></li>
                    <li>{icon('check_circle')}<span><strong>3</strong> classes équilibrées</span></li>
                    <li>{icon('check_circle')}<span>Valeurs manquantes (~3-5%)</span></li>
                    <li>{icon('check_circle')}<span>Numériques + catégorielles</span></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="feature-card">
                <p class="feature-title">{icon('account_tree')} Pipeline ML</p>
                <ul class="feature-list">
                    <li>{icon('check_circle')}<span>Imputation (médiane / mode)</span></li>
                    <li>{icon('check_circle')}<span>StandardScaler (numériques)</span></li>
                    <li>{icon('check_circle')}<span>OneHotEncoder (nominales)</span></li>
                    <li>{icon('check_circle')}<span>OrdinalEncoder (ordinales)</span></li>
                    <li>{icon('check_circle')}<span>LogReg / RandomForest / LinearSVC</span></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="warning-banner">
            {icon('warning', '1.2rem', '#FBBF24')}
            <span>
                <strong>Avertissement</strong> — Ce projet est un prototype éducatif.
                Il ne remplace en aucun cas l'évaluation d'un professionnel de santé mentale.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )