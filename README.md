# Mobile Apps Analysis – Streamlit

Interactive Streamlit app
It loads an Excel with Google Play–style app data and provides quick “slides” with data quality checks and insights.

## Files
- `app.py` – Streamlit app
- `GoogleAppData.xlsx` – dataset (Sheet1=data, Sheet2=column dictionary)
- `requirements.txt` – Python dependencies

## What’s inside
**Slide 1 – Dataset & Definitions**
- First 50 rows preview
- Column dictionary (from Sheet 2)

**Slide 2 – Data Quality**
- Exact duplicate rows
- Likely misaligned rows (left-shift heuristic via 1-col right shift)
- Missing values (custom rules):
  - **Rating** counted missing only if `Reviews > 0` and `Rating` is NaN
  - **Size**: “Varies with device” treated as missing
  - **Type**: NaN is missing
  - **Content Rating**: “Unrated” is missing

**Slide 3 – Insights**
