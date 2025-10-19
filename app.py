import streamlit as st
import pandas as pd
import numpy as np

# כותרת
st.title("ברוכה הבאה לאפליקציית Streamlit הראשונה שלך 🎉")

# טקסט הסבר
st.write("זהו מבחן שהכול עובד! בהמשך נוסיף ניתוח נתונים אמיתי מתוך קובץ ה-Excel שלך.")

# יצירת טבלה קטנה לדוגמה
data = {
    "קטגוריה": ["Social", "Game", "Education", "Health", "Finance"],
    "הורדות (במיליונים)": [120, 250, 90, 60, 80],
    "דירוג ממוצע": [4.3, 4.1, 4.5, 4.4, 4.2]
}
df = pd.DataFrame(data)

# הצגת טבלה
st.subheader("דוגמת נתונים")
st.dataframe(df)

# גרף פשוט
st.subheader("גרף הורדות לפי קטגוריה")
st.bar_chart(df.set_index("קטגוריה")["הורדות (במיליונים)"])

# אינטראקטיביות
st.subheader("בדקי את הפילטר:")
selected_category = st.selectbox("בחרי קטגוריה", df["קטגוריה"])
filtered = df[df["קטגוריה"] == selected_category]
st.write("תוצאות עבור:", selected_category)
st.dataframe(filtered)

