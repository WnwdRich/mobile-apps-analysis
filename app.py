import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mobile Apps Dataset", layout="wide")
st.title("Mobile Apps Dataset – Preview")

FILE_NAME = "GoogleAppData.xlsx"   # make sure this file is in the repo root

@st.cache_data
def load_excel(path: str):
    xls = pd.ExcelFile(path)
    data = pd.read_excel(xls, xls.sheet_names[0])                 # sheet 1 = data
    dictionary = pd.read_excel(xls, xls.sheet_names[1]) if len(xls.sheet_names) > 1 else None  # sheet 2 = column defs
    return data, dictionary, xls.sheet_names

try:
    df, dict_df, sheets = load_excel(FILE_NAME)

    st.caption(f"Found file **{FILE_NAME}** | Sheets: {', '.join(sheets)}")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Rows", len(df))
    with c2: st.metric("Columns", len(df.columns))
    with c3: st.write("First columns:", list(df.columns[:6]))

    tab_data, tab_dict = st.tabs(["📊 Data (Sheet 1)", "📖 Dictionary (Sheet 2)"])

    with tab_data:
        st.subheader("Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Missing values (%) by column")
        na_pct = (df.isna().mean().mul(100)).round(2).sort_values(ascending=False)
        st.dataframe(na_pct.to_frame("NA %"))

        st.subheader("Numeric summary")
        st.dataframe(df.describe(include="number").T)

    with tab_dict:
        if dict_df is not None:
            st.subheader("Column definitions")
            st.dataframe(dict_df, use_container_width=True)
        else:
            st.info("No second sheet with column dictionary was found.")

except FileNotFoundError:
    st.error(f"File `{FILE_NAME}` not found in the repository. Upload it to the repo root and rerun.")
except Exception as e:
    st.error(f"Error reading the Excel file: {e}")
