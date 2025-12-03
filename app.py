import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

st.set_page_config(
    page_title="Statistics Dashboard",
    layout="wide"
)
GROUP_MEMBERS = [
    "DYASTITI EKA MARLINDA - SID 004202400047",
    "LOVYTA AMELIA - SID 004202400108",
    "MUTIARA RAHEMI PUTRI - SID 004202400131 ",
     "PUTRI WULAN SARI - SID 004202400036",
]

def compute_descriptive_stats(df, numeric_only=True):
    """
    Returns a dataframe with:
    mean, median, mode, min, max, std for each column.
    """
    if numeric_only:
        data = df.select_dtypes(include=[np.number])
    else:
        data = df.copy()

    desc = pd.DataFrame(index=data.columns)
    desc["Mean"] = data.mean(numeric_only=True)
    desc["Median"] = data.median(numeric_only=True)

    modes = data.mode()
    if not modes.empty:
        desc["Mode"] = modes.iloc[0]
    else:
        desc["Mode"] = np.nan

    desc["Min"] = data.min(numeric_only=True)
    desc["Max"] = data.max(numeric_only=True)
    desc["Std Dev"] = data.std(numeric_only=True)

    return desc

def frequency_tables(df):
    """
    Returns a dict: {column_name: frequency_table_df}
    Each freq table has columns: 'Value', 'Frequency', 'Percentage'
    """
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
    """
    Returns r, p for selected numeric variables.
    Automatically drops NaN.
    """
    x = df[x_var]
    y = df[y_var]
    valid = ~(x.isna() | y.isna())
    x_clean = x[valid].astype(float)
    y_clean = y[valid].astype(float)

    if len(x_clean) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x_clean, y_clean)
    return r, p


def create_pdf_report(title, desc_stats, corr_results, group_members):
    """
    Create a simple PDF summary using fpdf.
    Returns bytes of the PDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Group Members:", ln=True)
    pdf.set_font("Arial", "", 11)
    for m in group_members:
        pdf.cell(0, 6, f"- {m}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Descriptive Statistics (preview):", ln=True)
    pdf.set_font("Arial", "", 9)
  
    desc_preview = desc_stats.copy()
    max_rows = 15
    max_cols = 6
    desc_preview = desc_preview.iloc[:max_rows, :max_cols]

# Print as simple text table
    col_names = ["Variable"] + list(desc_preview.columns)
    col_widths = [35] + [25] * len(desc_preview.columns)

    for col_name, w in zip(col_names, col_widths):
        pdf.cell(w, 6, str(col_name)[0:12], border=1)
    pdf.ln()

    for idx, row in desc_preview.iterrows():
        pdf.cell(col_widths[0], 6, str(idx)[0:18], border=1)
        for val, w in zip(row.values, col_widths[1:]):
            text = f"{val:.2f}" if isinstance(val, (int, float, np.floating)) else str(val)
            pdf.cell(w, 6, text[0:10], border=1)
        pdf.ln()

    pdf.ln(5)

# Correlations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Correlation Results:", ln=True)
    pdf.set_font("Arial", "", 11)
    def create_pdf_report(title, desc_stats, corr_results, group_members):
                pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(5)

    # group section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Group Members:", ln=True)
    pdf.set_font("Arial", "", 11)
    for m in group_members:
        pdf.cell(0, 6, f"- {m}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Descriptive Statistics (preview):", ln=True)
    pdf.set_font("Arial", "", 9)

    desc_preview = desc_stats.copy()
    max_rows = 15
    max_cols = 6
    desc_preview = desc_preview.iloc[:max_rows, :max_cols]

    col_names = ["Variable"] + list(desc_preview.columns)
    col_widths = [35] + [25] * len(desc_preview.columns)

    for col_name, w in zip(col_names, col_widths):
        pdf.cell(w, 6, str(col_name)[0:12], border=1)
    pdf.ln()

    for idx, row in desc_preview.iterrows():
        pdf.cell(col_widths[0], 6, str(idx)[0:18], border=1)
        for val, w in zip(row.values, col_widths[1:]):
            text = f"{val:.2f}" if isinstance(val, (int, float, np.floating)) else str(val)
            pdf.cell(w, 6, text[0:10], border=1)
        pdf.ln()

    pdf.ln(5)

    # Correlation section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Correlation Results:", ln=True)
    pdf.set_font("Arial", "", 11)

    if corr_results:
        for item in corr_results:
            x, y, r, p = item["x"], item["y"], item["r"], item["p"]
            pdf.multi_cell(0, 6, f"{x} vs {y}: r = {r:.3f}, p = {p:.4f}")
            pdf.multi_cell(0, 6, text)
    else:
        pdf.multi_cell(0, 6, "No correlation computed.", ln=True)

    output = pdf.output(dest="S")
    pdf_bytes = output if isinstance(output, (bytes, bytearray)) else output.encode("latin1", "replace")
    return pdf_bytes

st.sidebar.title("Settings")
st.sidebar.subheader("Group Members")
st.sidebar.write("This list is hard-coded in the script, edit it there:")
for m in GROUP_MEMBERS:
    st.sidebar.write(f"- {m}")

st.sidebar.markdown("---")
st.sidebar.write("Developed for: Tiktok & Study Performance Survey")
st.title("Survey Statistics Web App")
st.write("Upload your survey dataset (CSV) to begin.")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    st.markdown("### Column Types")
    st.write(df.dtypes)
    st.markdown("## Composite Scores (X_total & Y_total)")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    st.write("Select items to build composite scores. For example:")
    st.write("- X_total: TikTok-related usage items")
    st.write("- Y_total: Learning performance items")

    col1, col2 = st.columns(2)

    with col1:
        x_items = st.multiselect(
            "Select items for X_total",
            options=df.columns,
            default=[c for c in df.columns if "X" in c or "TikTok" in c][:3]
        )
    with col2:
        y_items = st.multiselect(
            "Select items for Y_total",
            options=df.columns,
            default=[c for c in df.columns if "Y" in c or "Belajar" in c][:3]
        )

    df_comp = df.copy()
    if x_items:
        df_comp["X_total"] = df_comp[x_items].sum(axis=1, numeric_only=True)
    if y_items:
        df_comp["Y_total"] = df_comp[y_items].sum(axis=1, numeric_only=True)

    st.write("Columns after adding composite scores:")
    st.dataframe(df_comp.head())

    st.markdown("## Descriptive Statistics")
    desc_stats = compute_descriptive_stats(df_comp.select_dtypes(include=[np.number]))
    st.write("Descriptive statistics for numeric variables (including composite scores):")
    st.dataframe(desc_stats)

    st.markdown("## Frequency & Percentage Tables")
    selected_column_for_freq = st.selectbox(
        "Select a variable for frequency table",
        options=df_comp.columns
    )
    freq_dict = frequency_tables(df_comp[[selected_column_for_freq]])
    st.write(f"Frequency table for: **{selected_column_for_freq}**")
    st.dataframe(freq_dict[selected_column_for_freq])

    st.markdown("## Histograms & Boxplots (Optional)")

    numeric_cols_with_comp = list(df_comp.select_dtypes(include=[np.number]).columns)
    selected_num_col = st.selectbox(
        "Select numeric variable for histogram & boxplot",
        options=numeric_cols_with_comp
    )

    if selected_num_col:
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.write(f"Histogram of {selected_num_col}")
            fig, ax = plt.subplots()
            ax.hist(df_comp[selected_num_col].dropna(), bins=10)
            ax.set_xlabel(selected_num_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with col_h2:
            st.write(f"Boxplot of {selected_num_col}")
            fig2, ax2 = plt.subplots()
            ax2.boxplot(df_comp[selected_num_col].dropna(), vert=True)
            ax2.set_ylabel(selected_num_col)
            st.pyplot(fig2)

    st.markdown("## Correlation (Pearson) & Scatterplot")

    col_corr1, col_corr2 = st.columns(2)
    with col_corr1:
        x_var_corr = st.selectbox(
            "X variable (independent)",
            options=numeric_cols_with_comp,
            index=numeric_cols_with_comp.index("X_total") if "X_total" in numeric_cols_with_comp else 0
        )
    with col_corr2:
        y_var_corr = st.selectbox(
            "Y variable (dependent)",
            options=numeric_cols_with_comp,
            index=numeric_cols_with_comp.index("Y_total") if "Y_total" in numeric_cols_with_comp else 0
        )
    corr_results = []
    if x_var_corr and y_var_corr:
        r, p = compute_pearson_correlation(df_comp, x_var_corr, y_var_corr)

        st.write(f"**Pearson Correlation between {x_var_corr} and {y_var_corr}:**")
        st.write(f"- Correlation coefficient (r): `{r:.4f}`" if not np.isnan(r) else "- r: not enough data")
        st.write(f"- p-value: `{p:.6f}`" if not np.isnan(p) else "- p-value: not enough data")

        corr_results.append({
            "x": x_var_corr,
            "y": y_var_corr,
            "r": r,
            "p": p
        })
        st.write("### Scatterplot")
        fig_sc, ax_sc = plt.subplots()
        ax_sc.scatter(df_comp[x_var_corr], df_comp[y_var_corr])
        ax_sc.set_xlabel(x_var_corr)
        ax_sc.set_ylabel(y_var_corr)
        ax_sc.set_title(f"{x_var_corr} vs {y_var_corr}")
        st.pyplot(fig_sc)

    st.markdown("## Export PDF Report")
    report_title = st.text_input(
        "Report Title",
        value="Descriptive Statistics & Correlation Report"
    )

    if st.button("Generate PDF Report"):
        pdf_bytes = create_pdf_report(
            title=report_title,
            desc_stats=desc_stats,
            corr_results=corr_results,
            group_members=GROUP_MEMBERS
        )
        st.success("PDF report generated.")
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="statistics_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Please upload a CSV file to start the analysis.")