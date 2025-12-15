# =========================
# Survey Statistics Web App
# BILINGUAL | FINAL VERSION
# =========================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Survey Statistics Web App",
    layout="wide"
)

# =========================
# LANGUAGE SELECTOR
# =========================
language = st.sidebar.selectbox(
    "Language / Bahasa",
    ["English", "Bahasa Indonesia"]
)

TEXT = {
    "English": {
        "menu": ["Team Members", "Statistics"],
        "team_title": "Team Members",
        "team_desc": "Project development team",
        "sid": "Student ID",
        "role": "Role",

        "stat_title": "Survey Statistics",
        "upload": "Upload CSV file",
        "upload_hint": "Upload your survey CSV file (exported from Google Forms).",

        "preview": "Dataset Preview",
        "col_type": "Column Types",

        "composite": "Composite Scores (X_total & Y_total)",
        "x_select": "Select items for X_total",
        "y_select": "Select items for Y_total",
        "after_comp": "Columns after adding composite scores",

        "desc": "Descriptive Statistics",
        "freq": "Frequency & Percentage Table",
        "freq_select": "Select variable",

        "hist_box": "Histogram & Boxplot",
        "choose_num": "Choose numeric variable",

        "corr": "Pearson Correlation & Scatterplot",
        "x_var": "X variable",
        "y_var": "Y variable",
        "why_corr": "Why Pearson Correlation?",
        "why_corr_text": (
            "Pearson correlation is used because both variables are numerical "
            "and assumed to have a linear relationship."
        ),

        "no_numeric": "No numeric columns available yet."
    },

    "Bahasa Indonesia": {
        "menu": ["Anggota Tim", "Statistik"],
        "team_title": "Anggota Tim",
        "team_desc": "Tim pengembang proyek",
        "sid": "NIM",
        "role": "Peran",

        "stat_title": "Statistik Survei",
        "upload": "Unggah file CSV",
        "upload_hint": "Unggah file CSV survei (hasil ekspor Google Forms).",

        "preview": "Pratinjau Data",
        "col_type": "Tipe Kolom",

        "composite": "Skor Komposit (X_total & Y_total)",
        "x_select": "Pilih item untuk X_total",
        "y_select": "Pilih item untuk Y_total",
        "after_comp": "Kolom setelah penambahan skor komposit",

        "desc": "Statistik Deskriptif",
        "freq": "Tabel Frekuensi & Persentase",
        "freq_select": "Pilih variabel",

        "hist_box": "Histogram & Boxplot",
        "choose_num": "Pilih variabel numerik",

        "corr": "Korelasi Pearson & Scatterplot",
        "x_var": "Variabel X",
        "y_var": "Variabel Y",
        "why_corr": "Alasan Menggunakan Korelasi Pearson",
        "why_corr_text": (
            "Korelasi Pearson digunakan karena kedua variabel bersifat numerik "
            "dan diasumsikan memiliki hubungan linier."
        ),

        "no_numeric": "Belum ada kolom numerik yang tersedia."
    }
}

T = TEXT[language]

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")

# =========================
# TEAM MEMBERS
# =========================
TEAM_MEMBERS = [
    {"name": "Dyastiti Eka Marlinda", "sid": "004202400047", "role": "Project Manager, Front-End & Application Structure", "image": "dyas.jpg"},
    {"name": "Lovyta Amelia", "sid": "004202400108", "role": "Data Handling & Preprocessing", "image": "amel.jpg"},
    {"name": "Mutiara Rahemi Putri", "sid": "004202400131", "role": "Localization & Reporting", "image": "muti.jpg"},
    {"name": "Putri Wulan Sari", "sid": "004202400036", "role": "Statistical Analysis", "image": "putri.jpg"}
]
# =========================
# FUNCTIONS
# =========================
def compute_descriptive_stats(df):
    data = df.select_dtypes(include=[np.number])
    desc = pd.DataFrame(index=data.columns)
    desc["Mean"] = data.mean()
    desc["Median"] = data.median()
    desc["Mode"] = data.mode().iloc[0] if not data.mode().empty else np.nan
    desc["Min"] = data.min()
    desc["Max"] = data.max()
    desc["Std Dev"] = data.std()
    return desc

def compute_pearson(df, x, y):
    valid = df[[x, y]].dropna()
    if len(valid) <3:
        return None, None

    if valid[x].nunique() <= 1 or valid[y].nunique() <= 1:
        return None, None

    r, p = stats.pearsonr(valid[x], valid[y])
    return float(r), float(p)


# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio("Menu", T["menu"])

# =========================
# PAGE: TEAM MEMBERS
# =========================
if page == T["menu"][0]:
    st.title(T["team_title"])
    st.write(T["team_desc"])

    for m in TEAM_MEMBERS:
        c1, c2 = st.columns([1, 3])
        img_path = os.path.join(ASSET_DIR, m["image"])

        if os.path.exists(img_path):
            c1.image(Image.open(img_path), use_container_width=True)
        else:
            c1.info("Image not found")

        c2.subheader(m["name"])
        c2.write(f"**{T['sid']}:** {m['sid']}")
        c2.write(f"**{T['role']}:** {m['role']}")
        st.divider()

# =========================
# PAGE: STATISTICS
# =========================
else:
    st.title(T["stat_title"])
    st.write(T["upload_hint"])

    uploaded_file = st.file_uploader(T["upload"], type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)

    st.subheader(T["preview"])
    st.dataframe(df.head())

    st.markdown(f"### {T['col_type']}")
    st.write(df.dtypes)

    # Composite scores
    st.markdown(f"## {T['composite']}")
    c1, c2 = st.columns(2)
    with c1:
        x_items = st.multiselect(T["x_select"], df.columns)
    with c2:
        y_items = st.multiselect(T["y_select"], df.columns)

    df_comp = df.copy()
    df_comp[x_items + y_items] = df_comp[x_items + y_items].apply(pd.to_numeric, errors="coerce")

    df_comp["X_total"] = df_comp[x_items].sum(axis=1) if x_items else np.nan
    df_comp["Y_total"] = df_comp[y_items].sum(axis=1) if y_items else np.nan

    st.write(T["after_comp"])
    st.dataframe(df_comp.head())

    # Descriptive stats
    st.markdown(f"## {T['desc']}")
    desc = compute_descriptive_stats(df_comp)
    st.dataframe(desc)

    # Histogram & Boxplot
    st.markdown(f"## {T['hist_box']}")
    numeric_cols = df_comp.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        sel = st.selectbox(T["choose_num"], numeric_cols)
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            ax.hist(df_comp[sel].dropna())
            st.pyplot(fig)

        with c2:
            fig2, ax2 = plt.subplots()
            ax2.boxplot(df_comp[sel].dropna())
            st.pyplot(fig2)
    else:
        st.info(T["no_numeric"])

    # Correlation
    st.markdown(f"## {T['corr']}")
    x_var = st.selectbox(T["x_var"], numeric_cols)
    y_var = st.selectbox(T["y_var"], numeric_cols)

    r, p = compute_pearson(df_comp, x_var, y_var)
    st.subheader(T["why_corr"])
    st.write(T["why_corr_text"])
    if r is None or p is None:
        st.warning("Correlation cannot be computed (data insufficient or constant).")
    else:
        st.write(f"r = {r:.4f}, p = {p:.4f}")


    fig, ax = plt.subplots()
    ax.scatter(df_comp[x_var], df_comp[y_var])
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    st.pyplot(fig)
