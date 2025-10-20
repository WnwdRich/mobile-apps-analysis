import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mobile Apps Dataset", layout="wide")
st.title("Mobile Apps Dataset – Data Quality Checks")

FILE_NAME = "GoogleAppData.xlsx"

@st.cache_data
def load_excel(path: str):
    xls = pd.ExcelFile(path)
    data = pd.read_excel(xls, xls.sheet_names[0])
    dictionary = pd.read_excel(xls, xls.sheet_names[1]) if len(xls.sheet_names) > 1 else None
    return data, dictionary, xls.sheet_names

try:
    df, dict_df, sheets = load_excel(FILE_NAME)

    st.caption(f"Loaded **{FILE_NAME}** | Sheets: {', '.join(sheets)}")

    tab_data, tab_duplicates, tab_missing, tab_dict = st.tabs(
        ["📊 Data (Sheet 1)", "🔁 Duplicate Records", "⚠️ Missing Values", "📖 Dictionary (Sheet 2)"]
    )

    # ---- Data preview ----
    with tab_data:
        st.subheader("First 50 rows of dataset")
        st.dataframe(df.head(50), use_container_width=True)

    # ---- Duplicate detection ----
    with tab_duplicates:
        st.subheader("Detect duplicate rows (all columns identical)")
        duplicated_mask = df.duplicated(keep=False)
        duplicates = df[duplicated_mask]

        st.write(f"**Total duplicate rows:** {len(duplicates)} "
                 f"out of {len(df)} total rows "
                 f"({len(duplicates)/len(df)*100:.2f}% duplicates).")

        if not duplicates.empty:
            st.dataframe(duplicates, use_container_width=True)
        else:
            st.success("✅ No duplicate records found — all rows are unique.")

    # ---- Missing values analysis ----
    with tab_missing:
        st.subheader("Empty / Missing Values Analysis")
        missing_count = df.isna().sum()
        missing_percent = (df.isna().mean() * 100).round(2)
        missing_summary = pd.DataFrame({
            "Missing Values": missing_count,
            "Missing %": missing_percent
        }).sort_values(by="Missing %", ascending=False)

        st.write(f"**Columns with at least one missing value:** {(missing_count > 0).sum()} out of {len(df.columns)}")
        st.dataframe(missing_summary, use_container_width=True)

        # Optional: bar chart for visualization
        st.subheader("Visualize missing percentage by column")
        st.bar_chart(missing_summary["Missing %"])

    # ---- Dictionary tab ----
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
