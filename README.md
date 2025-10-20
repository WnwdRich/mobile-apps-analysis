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

**Slide 3 — Insights & Findings**

Interactive analyses built on the cleaned dataset (exact duplicates removed; optional misaligned-row exclusion; numeric parsing for Installs/Reviews/Size). Each block includes toggles and concise visuals.

### 1) Popular Categories by Installs
- **What:** Ranks categories by **total installs**; also shows **avg installs/app** and **#apps**.
- **Cleaning:** Drops rows missing *Category* or *Installs*. Installs parsed (e.g., `1,000,000+ → 1000000`).
- **UI:** *Top-N slider*, optional full table.
- **Output:** Bar chart + highlight of the top category.

### 2) Type (Free/Paid) → Reviews & Rating (by Category)
- **What:** Compares **Free vs Paid** for **average rating** and **total reviews**.
- **Cleaning:** Type strictly normalized to **Free/Paid** (anything else dropped). Optional per-category filter.
- **UI:** Category dropdown; bar charts + optional summary tables.
- **Output:** Mean/median **rating** by Type; **total reviews** by Type; counts of apps.

### 3) Rating × Size (bucketed)
- **What:** Relationship between **app size** and **ratings**.
- **Cleaning:** Excludes `Rating = NaN` and `Size = "Varies with device"` (treated as `NaN`).
- **Buckets:** `<20MB`, `20–50MB`, `50–100MB`; rating rounded to 1 decimal.
- **UI:** Optional table, min-count filter.
- **Output:** Stacked bars of **counts per rating** across size buckets.  
  **Conclusion:** Within these buckets, **size does not appear to affect rating**.

### 4) Content Rating (age target) → Installs
- **What:** Which age groups drive **more installs**.
- **Cleaning:** Drops rows missing *Content Rating* or *Installs*; optional **exclude “Unrated”**; optional per-category filter.
- **UI:** Results table (total/avg/median installs, #apps) + **donut chart** of total installs share.
- **Why:** Installs are the key value signal; helps decide **target age segment** for a new app.

> All charts/tables are computed on deduplicated data and respect the app’s parsing/missing-value rules.

