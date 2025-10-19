import streamlit as st
import pandas as pd

st.set_page_config(page_title="Google App Data – Step 1", layout="wide")
st.title("Google App Data – טעינה ותצוגה ראשונית")

FILE_NAME = "GoogleAppData.xlsx"

@st.cache_data
def load_excel(file_name: str):
    # קורא את כל שמות הגיליונות
    xls = pd.ExcelFile(file_name)
    sheet_names = xls.sheet_names

    # לפי ההנחיות: הטאב הראשון = הדאטה, השני = מילון עמודות
    df_data = pd.read_excel(xls, sheet_name=sheet_names[0])
    df_dict = pd.read_excel(xls, sheet_name=sheet_names[1]) if len(sheet_names) > 1 else None
    return df_data, df_dict, sheet_names

try:
    df, dict_df, sheets = load_excel(FILE_NAME)

    st.success(f"נמצא הקובץ {FILE_NAME}. גיליונות: {', '.join(sheets)}")

    # תצוגה מהירה
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("שורות", len(df))
    with c2: st.metric("עמודות", len(df.columns))
    with c3: st.write("עמודות ראשונות:", list(df.columns[:6]))

    # טאבים: נתונים / מילון
    tab1, tab2 = st.tabs(["📊 Data (Sheet1)", "📖 Dictionary (Sheet2)"])
    with tab1:
        st.subheader("הצצה ל־50 השורות הראשונות")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("אחוז חוסרים לכל עמודה")
        na = (df.isna().mean()*100).round(2).sort_values(ascending=False)
        st.dataframe(na.to_frame("NA %"))

        st.subheader("Summary סטטיסטי (לעמודות מספריות)")
        st.dataframe(df.describe(include='number').T)

    with tab2:
        if dict_df is not None:
            st.write("מילון העמודות כפי שמופיע בטאב השני:")
            st.dataframe(dict_df, use_container_width=True)
        else:
            st.info("לא נמצא טאב שני עם מילון עמודות.")

except FileNotFoundError:
    st.error(f"לא נמצא הקובץ `{FILE_NAME}` ברפוזיטורי. העלי אותו ל-GitHub (Add file → Upload files) ושמרי.")
except Exception as e:
    st.error(f"ארעה שגיאה בקריאת הקובץ: {e}")
