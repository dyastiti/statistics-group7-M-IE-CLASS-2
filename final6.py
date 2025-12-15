# =========================
# Survey Statistics Web App
# FINAL + SCATTER & BOX PLOT
# =========================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from fpdf import FPDF
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
        "preview": "Dataset Preview",
        "desc": "Descriptive Statistics",
        "corr": "Pearson Correlation",
        "scatter": "Scatter Plot",
        "box": "Box Plot",
        "pdf": "Export PDF",
        "pdf_btn": "Download PDF Report"
    },
    "Bahasa Indonesia": {
        "menu": ["Anggota Tim", "Statistik"],
        "team_title": "Anggota Tim",
        "team_desc": "Tim pengembang proyek",
        "sid": "NIM",
        "role": "Peran",
        "stat_title": "Statistik Survei",
        "upload": "Unggah file CSV",
        "preview": "Pratinjau Data",
        "desc": "Statistik Deskriptif",
        "corr": "Korelasi Pearson",
        "scatter": "Diagram Pencar",
        "box": "Diagram Boxplot",
        "pdf": "Ekspor PDF",
        "pdf_btn": "Unduh Laporan PDF"
    }
}

T = TEXT[language]

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")

# =========================
# TEAM MEMBERS DATA
# =========================
TEAM_MEMBERS = [
    {
        "name": "Dyastiti Eka Marlinda",
        "sid": "004202400047",
        "role": "Front-End & Application Structure",
        "image": "dyas.jpg"
    },
    {
        "name": "Lovyta Amelia",
        "sid": "004202400108",
        "role": "Data Handling & Preprocessing",
        "image": "amel.jpg"
    },
    {
        "name": "Mutiara Rahemi Putri",
        "sid": "004202400131",
        "role": "Localization & Reporting",
        "image": "muti.jpg"
    },
    {
        "name": "Putri Wulan Sari",
        "sid": "004202400036",
        "role": "Statistical Analysis",
        "image": "putri.jpg"
    }
]

# =========================
# UTILITY (UNICODE SAFE)
# =========================
def safe_text(text):
    replacements = {
        "“": '"', "”": '"',
        "’": "'", "‘": "'",
        "–": "-", "—": "-",
        "…": "..."
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def compute_descriptive_stats(df, numeric_only=True):
    if numeric_only:
        data = df.select_dtypes(include=[np.number])
    else:
        data = df.copy()

    desc = pd.DataFrame(index=data.columns)
    desc["Mean"] = data.mean(numeric_only=True)
    desc["Median"] = data.median(numeric_only=True)

    modes = data.mode()
    desc["Mode"] = modes.iloc[0] if not modes.empty else np.nan
    desc["Min"] = data.min(numeric_only=True)
    desc["Max"] = data.max(numeric_only=True)
    desc["Std Dev"] = data.std(numeric_only=True)
    return desc

def frequency_tables(df):
    freq_dict = {}
    n = len(df)
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        perc = (counts / n * 100).round(2)
        freq_table = pd.DataFrame({
            "Value": counts.index,
            "Frequency": counts.values,
            "Percentage": perc.values
        })
        freq_dict[col] = freq_table
    return freq_dict

def compute_pearson_correlation(df, x_var, y_var):
    x = df[x_var]
    y = df[y_var]
    valid = ~(x.isna() | y.isna())
    x_clean = x[valid].astype(float)
    y_clean = y[valid].astype(float)

    if len(x_clean) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x_clean, y_clean)
    return r, p

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
        col1, col2 = st.columns([1, 3])
        img_path = os.path.join(ASSET_DIR, m["image"])

        if os.path.exists(img_path):
            col1.image(Image.open(img_path), use_container_width=True)
        else:
            col1.warning("Image not found")

        col2.subheader(m["name"])
        col2.write(f"**{T['sid']}:** {m['sid']}")
        col2.write(f"**{T['role']}:** {m['role']}")
        st.divider()

# =========================
# PAGE: STATISTICS
# =========================
else:
        st.markdown('<div class="top-row"><h2>Survey Statistics</h2></div>', unsafe_allow_html=True)
        st.write("Upload your survey CSV (export from Google Forms). Then select items to build composites and run analyses.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        # read csv
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Dataset")
        st.dataframe(df.head())
        st.markdown("### Column types")
        st.write(df.dtypes)

        # Composite selection
        st.markdown("## Composite Scores (X_total & Y_total)")
        col1, col2 = st.columns(2)
        with col1:
            x_items = st.multiselect("Select items for X_total", options=list(df.columns))
        with col2:
            y_items = st.multiselect("Select items for Y_total", options=list(df.columns))

        df_comp = df.copy()
        if x_items:
            # try to convert to numeric when possible
            df_comp[x_items] = df_comp[x_items].apply(pd.to_numeric, errors="coerce")
            df_comp["X_total"] = df_comp[x_items].sum(axis=1, numeric_only=True)
        else:
            df_comp["X_total"] = np.nan

        if y_items:
            df_comp[y_items] = df_comp[y_items].apply(pd.to_numeric, errors="coerce")
            df_comp["Y_total"] = df_comp[y_items].sum(axis=1, numeric_only=True)
        else:
            df_comp["Y_total"] = np.nan

        st.write("Columns after adding composite scores:")
        st.dataframe(df_comp.head())

        # Descriptive stats
        st.markdown("## Descriptive Statistics")
        desc_stats = compute_descriptive_stats(df_comp.select_dtypes(include=[np.number]))
        st.dataframe(desc_stats)

        # Frequency & percentage
        st.markdown("## Frequency & Percentage Tables")
        freq_col = st.selectbox("Select variable for frequency table", options=list(df_comp.columns))
        freq_dict = frequency_tables(df_comp[[freq_col]])
        st.dataframe(freq_dict[freq_col])

        # Hist & boxplot
        st.markdown("## Histograms & Boxplots")
        numeric_cols_with_comp = list(df_comp.select_dtypes(include=[np.number]).columns)
        if numeric_cols_with_comp:
            sel_num = st.selectbox("Choose numeric variable", options=numeric_cols_with_comp)
            if sel_num:
                cA, cB = st.columns(2)
                with cA:
                    st.write(f"Histogram of {sel_num}")
                    fig, ax = plt.subplots()
                    ax.hist(df_comp[sel_num].dropna())
                    ax.set_xlabel(sel_num)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                with cB:
                    st.write(f"Boxplot of {sel_num}")
                    fig2, ax2 = plt.subplots()
                    ax2.boxplot(df_comp[sel_num].dropna())
                    st.pyplot(fig2)
        else:
            st.info("No numeric columns available yet (try building composite scores).")

        # Correlation
        st.markdown("## Pearson Correlation & Scatterplot")
        numeric_cols_with_comp = list(df_comp.select_dtypes(include=[np.number]).columns)
        if numeric_cols_with_comp:
            colC, colD = st.columns(2)
            with colC:
                x_var_corr = st.selectbox("X variable", options=numeric_cols_with_comp, index=numeric_cols_with_comp.index("X_total") if "X_total" in numeric_cols_with_comp else 0)
            with colD:
                y_var_corr = st.selectbox("Y variable", options=numeric_cols_with_comp, index=numeric_cols_with_comp.index("Y_total") if "Y_total" in numeric_cols_with_comp else 0)

            corr_results = []
            if x_var_corr and y_var_corr:
                r, p = compute_pearson_correlation(df_comp, x_var_corr, y_var_corr)
                st.write(f"*Pearson Correlation between {x_var_corr} and {y_var_corr}:*")
                st.write("### Why Pearson Correlation?")
                st.write(
                    "Pearson correlation is used because both variables are numerical and "
                    "assumed to have a linear relationship. Likert-scale items converted "
                    "into numeric values behave as continuous measures suitable for Pearson."
                )
                st.write(f"- r = {r:.4f}" if not np.isnan(r) else "- r: not enough data")
                st.write(f"- p = {p:.6f}" if not np.isnan(p) else "- p: not enough data")
                corr_results.append({"x": x_var_corr, "y": y_var_corr, "r": r, "p": p})

                st.write("### Scatterplot")
                fig_sc, ax_sc = plt.subplots()
                ax_sc.scatter(df_comp[x_var_corr], df_comp[y_var_corr])
                ax_sc.set_xlabel(x_var_corr)
                ax_sc.set_ylabel(y_var_corr)
                st.pyplot(fig_sc)
        else:
            st.info("No numeric columns available for correlation (build composites first).")


