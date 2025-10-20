import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Mobile Apps – Slides", layout="wide")
st.title("Mobile Apps – Concise Slides")

FILE_NAME = "GoogleAppData.xlsx"

# ---------- Load ----------
@st.cache_data
def load_excel(path: str):
    xls = pd.ExcelFile(path)
    data = pd.read_excel(xls, xls.sheet_names[0])                  # Sheet 1: dataset
    dictionary = pd.read_excel(xls, xls.sheet_names[1]) if len(xls.sheet_names) > 1 else None  # Sheet 2: definitions
    return data, dictionary, xls.sheet_names

# ---------- Helpers ----------
def to_int_like(x):
    """Parse numbers like '1,234+' -> 1234 (float)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("+", "")
    return pd.to_numeric(s, errors="coerce")

def size_to_mb(x):
    """Parse '15M', '500k', '1.2G', 'Varies with device' -> MB (float)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(" ", "")
    if "varies" in s:
        return np.nan
    m = re.match(r"^([\d\.]+)([kmg])?b?$", s)
    if not m:
        return pd.to_numeric(s, errors="coerce")
    val = float(m.group(1)); unit = m.group(2)
    if unit == "k": return val / 1024
    if unit == "g": return val * 1024
    return val  # default MB

def find_col(cols, *aliases):
    """Case-insensitive finder with substring fallback."""
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    for c in cols:
        if any(a.lower() in c.lower() for a in aliases):
            return c
    return None

def row_fail_score(row, cols):
    """Simple validity checks used for misalignment heuristic."""
    score = 0
    c_rating  = find_col(cols, "rating", "ratings")
    c_reviews = find_col(cols, "reviews", "review_count", "num_reviews")
    c_inst    = find_col(cols, "installs", "downloads")
    c_size    = find_col(cols, "size", "app_size")

    if c_rating and not pd.isna(row[c_rating]):
        val = pd.to_numeric(row[c_rating], errors="coerce")
        if pd.isna(val) or val < 0 or val > 5:
            score += 1

    if c_reviews and not pd.isna(row[c_reviews]):
        val = pd.to_numeric(row[c_reviews], errors="coerce")
        if pd.isna(val) or val < 0:
            score += 1

    if c_inst and not pd.isna(row[c_inst]):
        val = to_int_like(row[c_inst])
        if pd.isna(val) or val < 0:
            score += 1

    if c_size and not pd.isna(row[c_size]):
        val = size_to_mb(row[c_size])
        if pd.isna(val) or val <= 0 or val > 5120:   # sanity: 0..5GB
            score += 1

    return score

def shift_row_right_by_1(row):
    """Simulate a one-column right shift to detect left-shifted rows."""
    vals = row.values
    shifted = np.empty_like(vals, dtype=object)
    shifted[0] = np.nan
    shifted[1:] = vals[:-1]
    return pd.Series(shifted, index=row.index)

# ---------- App ----------
try:
    df, dict_df, sheets = load_excel(FILE_NAME)
    st.caption(f"Loaded **{FILE_NAME}** | Sheets: {', '.join(sheets)}")

    # TABS — keep them INSIDE the try block
    slide1, slide2, slide3 = st.tabs([
        "📄 Slide 1 – Dataset + Definitions",
        "🧪 Slide 2 – Data Quality",
        "📌 Slide 3 – Insights & Findings"
    ])

    # =========================
    # Slide 1 – Dataset + Definitions
    # =========================
    with slide1:
        st.subheader("Dataset (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Definitions (Sheet 2)")
        if dict_df is not None:
            st.dataframe(dict_df, use_container_width=True)
        else:
            st.info("No second sheet with column dictionary was found.")

    # =========================
    # Slide 2 – Data Quality
    # =========================
    with slide2:
        # --- Duplicates ---
        st.subheader("Duplicate Records (all columns identical)")
        duplicated_mask = df.duplicated(keep=False)
        duplicates = df[duplicated_mask]

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total rows", len(df))
        with c2: st.metric("Duplicate rows", len(duplicates))
        with c3:
            pct = (len(duplicates) / len(df) * 100) if len(df) else 0
            st.metric("Duplicate %", f"{pct:.2f}%")

        if not duplicates.empty:
            st.dataframe(duplicates, use_container_width=True)
        else:
            st.success("✅ No duplicate records found.")

        # --- Misalignment explanation + likely misaligned ---
        st.markdown("---")
        st.subheader("Misaligned (left-shift) test – explanation")
        st.write(
            "- Validate each row (Rating 0–5, Installs/Reviews ≥ 0 after cleaning, Size 0–5120MB).  \n"
            "- Simulate a **1-column right shift** to detect rows likely left-shifted during export.  \n"
            "- If shifting right reduces failures significantly, flag as **Likely misaligned**."
        )

        # Compute validation scores
        orig_scores = df.apply(lambda r: row_fail_score(r, df.columns), axis=1)
        shifted_df = df.apply(shift_row_right_by_1, axis=1)
        shifted_scores = shifted_df.apply(lambda r: row_fail_score(r, shifted_df.columns), axis=1)
        results = pd.DataFrame({
            "orig_fail_score": orig_scores,
            "shift_right_fail_score": shifted_scores,
            "improvement_if_shift_right": orig_scores - shifted_scores
        })

        suspicious_mask = (results["orig_fail_score"] >= 2) & (results["improvement_if_shift_right"] >= 2)
        suspicious_idx = results[suspicious_mask].index

        c4, c5 = st.columns(2)
        with c4: st.metric("Rows with ≥1 failed check", int((results["orig_fail_score"] >= 1).sum()))
        with c5: st.metric("Likely misaligned rows", int(len(suspicious_idx)))

        if len(suspicious_idx) > 0:
            st.write("**Likely misaligned (left-shifted) rows:**")
            st.dataframe(df.loc[suspicious_idx], use_container_width=True)
        else:
            st.success("✅ No strongly suspicious misalignment patterns detected.")

        # --- Missing values per column (with exclusions + custom logic) ---
        st.markdown("---")
        st.subheader("Missing values per column (cleaned with custom rules)")

        # Toggles
        colA, colB, colC = st.columns(3)
        with colA:
            exclude_dupes = st.checkbox("Exclude duplicate rows", value=True)
        with colB:
            exclude_misaligned = st.checkbox("Exclude likely misaligned rows", value=True)
        with colC:
            show_compare = st.checkbox("Show before vs after", value=False)

        # Build row mask
        use_mask = pd.Series(True, index=df.index)
        if exclude_dupes:
            use_mask &= ~duplicated_mask
        if exclude_misaligned:
            use_mask &= ~df.index.isin(suspicious_idx)

        cleaned_df = df[use_mask]

        c6, c7 = st.columns(2)
        with c6: st.metric("Rows used for NA counts", len(cleaned_df))
        with c7: st.metric("Rows excluded", len(df) - len(cleaned_df))

        # ---- Custom missing logic ----
        rating_col  = find_col(cleaned_df.columns, "rating", "ratings")
        reviews_col = find_col(cleaned_df.columns, "reviews", "review_count", "num_reviews")
        size_col    = find_col(cleaned_df.columns, "size", "app_size")
        type_col    = find_col(cleaned_df.columns, "type")
        content_col = find_col(cleaned_df.columns, "content rating", "content_rating", "contentrating")

        # Start with true nulls
        miss_mask = cleaned_df.isna().copy()

        # Size: also count "Varies with device" as missing
        if size_col:
            varies_mask = cleaned_df[size_col].astype(str).str.strip().str.lower().str.contains("varies with device", na=False)
            miss_mask[size_col] = miss_mask[size_col] | varies_mask

        # Content Rating: count "Unrated" as missing
        if content_col:
            unrated_mask = cleaned_df[content_col].astype(str).str.strip().str.lower().eq("unrated")
            miss_mask[content_col] = miss_mask[content_col] | unrated_mask

        # Rating: missing only if Reviews > 0 and Rating is NaN
        if rating_col and reviews_col:
            rev_num = cleaned_df[reviews_col].apply(to_int_like)
            rating_is_null = cleaned_df[rating_col].isna()
            special_rating_missing = rating_is_null & (rev_num.fillna(0) > 0)
            miss_mask[rating_col] = special_rating_missing  # override
        # Type: keep default (true nulls already marked)

        # Final counts
        missing_clean = miss_mask.sum().rename("Missing Values").sort_values(ascending=False)
        st.dataframe(missing_clean.to_frame(), use_container_width=True)

        if show_compare:
            st.subheader("Before vs After (diagnostic)")
            # Compute "before" with the same custom logic but on full df
            df_full = df.copy()
            miss_full = df_full.isna().copy()

            size_col_full    = find_col(df_full.columns, "size", "app_size")
            content_col_full = find_col(df_full.columns, "content rating", "content_rating", "contentrating")
            rating_col_full  = find_col(df_full.columns, "rating", "ratings")
            reviews_col_full = find_col(df_full.columns, "reviews", "review_count", "num_reviews")

            if size_col_full:
                varies_full = df_full[size_col_full].astype(str).str.strip().str.lower().str.contains("varies with device", na=False)
                miss_full[size_col_full] = miss_full[size_col_full] | varies_full
            if content_col_full:
                unrated_full = df_full[content_col_full].astype(str).str.strip().str.lower().eq("unrated")
                miss_full[content_col_full] = miss_full[content_col_full] | unrated_full
            if rating_col_full and reviews_col_full:
                rev_num_full = df_full[reviews_col_full].apply(to_int_like)
                rating_null_full = df_full[rating_col_full].isna()
                miss_full[rating_col_full] = rating_null_full & (rev_num_full.fillna(0) > 0)

            missing_before = miss_full.sum().rename("Missing (before)")
            compare = pd.concat([missing_before, missing_clean], axis=1)
            compare["Δ removed (dupes/misaligned exclusions)"] = compare["Missing (before)"] - compare["Missing Values"]
            st.dataframe(compare.sort_values("Missing Values", ascending=False), use_container_width=True)

        # =========================
        # Slide 3 – Insights & Findings
        # =========================
        with slide3:
            # --- Analysis 1: Popular Categories by Installs ---
            st.subheader("Popular Categories by Installs")
        
            # 1) Locate needed columns
            cat_col = find_col(df.columns, "category", "categories")
            inst_col = find_col(df.columns, "installs", "downloads")
        
            if not cat_col or not inst_col:
                st.error("Could not find 'Category' or 'Installs' columns. Please check the column names in the dataset.")
            else:
                # 2) Deduplicate identical rows (keep first)
                dedup = df.drop_duplicates(keep="first").copy()
        
                # 3) Clean installs → numeric
                dedup["_installs_num"] = dedup[inst_col].apply(to_int_like)
        
                # 4) Drop rows with missing fields needed for this insight
                clean_insight_df = dedup.dropna(subset=[cat_col, "_installs_num"]).copy()
        
                # OPTIONAL: also exclude the likely-misaligned rows flagged in Slide 2
                # clean_insight_df = clean_insight_df[~clean_insight_df.index.isin(suspicious_idx)]
        
                # 5) Aggregate
                clean_insight_df["._dummy"] = 1
                agg = (
                    clean_insight_df
                    .groupby(cat_col, dropna=False, as_index=False)
                    .agg(
                        total_installs=("_installs_num", "sum"),
                        avg_installs_per_app=("_installs_num", "mean"),
                        apps=("._dummy", "size")
                    )
                )
        
                # Sort by total installs desc
                agg_sorted = agg.sort_values("total_installs", ascending=False, ignore_index=True)
        
                # UI controls
                col1, col2 = st.columns(2)
                with col1:
                    maxN = min(15, len(agg_sorted)) if len(agg_sorted) else 3
                    defaultN = min(10, len(agg_sorted)) if len(agg_sorted) else 3
                    top_n = st.slider("Show top N categories (by total installs)", 3, maxN, defaultN)
                with col2:
                    show_table = st.checkbox("Show full table", value=False)
        
                top = agg_sorted.head(top_n)
        
                # Bar chart (total installs)
                st.caption("Total installs (sum) by category")
                if not top.empty:
                    chart_df = top.set_index(cat_col)[["total_installs"]]
                    st.bar_chart(chart_df)
                else:
                    st.info("No data available after de-duplication and missing-value filtering.")
        
                # Quick highlights
                if not top.empty:
                    top_cat = top.iloc[0]
                    st.markdown(
                        f"- **Top category by total installs:** `{top_cat[cat_col]}`  \n"
                        f"- **Total installs:** {int(top_cat['total_installs']):,}  \n"
                        f"- **Apps in category:** {int(top_cat['apps']):,}  \n"
                        f"- **Average installs/app (top category):** {int(top_cat['avg_installs_per_app']):,}"
                    )
        
                # Optional table
                if show_table and not agg_sorted.empty:
                    nice = agg_sorted.copy()
                    nice["total_installs"] = nice["total_installs"].round(0).astype("int64")
                    nice["avg_installs_per_app"] = nice["avg_installs_per_app"].round(0).astype("int64")
                    st.dataframe(nice, use_container_width=True)
        
                # Method note (for graders)
                with st.expander("Method (data used for this insight)"):
                    kept_rows = len(clean_insight_df)
                    dropped_dupes = len(df) - len(dedup)
                    dropped_missing = len(dedup) - len(clean_insight_df)
                    st.write(
                        f"- Dropped **exact duplicate** rows: {dropped_dupes}  \n"
                        f"- Dropped rows with missing **Category** or **Installs** (after cleaning): {dropped_missing}  \n"
                        f"- Rows used in this analysis: {kept_rows}"
                    )
            # --- Analysis 2: Type (Free/Paid) → Reviews & Rating (by Category) ---
            st.markdown("### Type (Free/Paid) → Reviews & Rating (by Category)")
            
            type_col   = find_col(df.columns, "type")
            cat_col    = find_col(df.columns, "category", "categories")
            rating_col = find_col(df.columns, "rating", "ratings")
            reviews_col= find_col(df.columns, "reviews", "review_count", "num_reviews")
            
            if not all([type_col, cat_col, rating_col, reviews_col]):
                st.info("Type analysis skipped (missing Type/Category/Rating/Reviews column).")
            else:
                base_t = df.drop_duplicates(keep="first").copy()
                base_t["_type"]    = base_t[type_col].astype(str).str.strip().str.title()  # normalize to 'Free'/'Paid'
                base_t["_rating"]  = pd.to_numeric(base_t[rating_col], errors="coerce")
                base_t["_reviews"] = base_t[reviews_col].apply(to_int_like)
                base_t = base_t.dropna(subset=[cat_col, type_col])
                base_t["._dummy"] = 1
            
                cat_options_t = ["All categories"] + sorted(base_t[cat_col].dropna().astype(str).unique().tolist())
                sel_cat_t = st.selectbox("Choose category (Type analysis)", cat_options_t, key="type_cat_select")
                view_t = base_t if sel_cat_t == "All categories" else base_t[base_t[cat_col].astype(str) == sel_cat_t]
            
                if view_t.empty:
                    st.info("No data for the selected category after parsing and filtering.")
                else:
                    # Ratings (ignore NaN ratings)
                    rating_view_t = view_t.dropna(subset=["_rating"]).copy()
                    r_t = (
                        rating_view_t.groupby("_type", as_index=False)
                        .agg(apps=("._dummy", "size"),
                             mean_rating=("_rating", "mean"),
                             median_rating=("_rating", "median"))
                        .sort_values("_type")
                    )
            
                    # Reviews (keep zero; drop NaN reviews only)
                    reviews_view_t = view_t.dropna(subset=["_reviews"]).copy()
                    v_t = (
                        reviews_view_t.groupby("_type", as_index=False)
                        .agg(apps=("._dummy", "size"),
                             total_reviews=("_reviews", "sum"),
                             mean_reviews=("_reviews", "mean"),
                             median_reviews=("_reviews", "median"))
                        .sort_values("_type")
                    )
            
                    c3, c4 = st.columns(2)
                    with c3:
                        st.caption("Average rating by Type")
                        if not r_t.empty:
                            st.bar_chart(r_t.set_index("_type")[["mean_rating"]])
                        else:
                            st.info("No rating data after filtering.")
                    with c4:
                        st.caption("Total reviews by Type")
                        if not v_t.empty:
                            st.bar_chart(v_t.set_index("_type")[["total_reviews"]])
                        else:
                            st.info("No reviews data after filtering.")
            
                    with st.expander("Tables (optional) - Type impact"):
                        if not r_t.empty:
                            g1 = r_t.copy()
                            g1["mean_rating"] = g1["mean_rating"].round(2)
                            g1["median_rating"] = g1["median_rating"].round(2)
                            st.write("**Ratings summary**")
                            st.dataframe(g1, use_container_width=True)
                        if not v_t.empty:
                            g2 = v_t.copy()
                            g2["total_reviews"] = g2["total_reviews"].round(0).astype("int64").map(lambda x: f"{x:,}")
                            g2["mean_reviews"] = g2["mean_reviews"].round(1)
                            g2["median_reviews"] = g2["median_reviews"].round(1)
                            st.write("**Reviews summary**")
                            st.dataframe(g2, use_container_width=True)
 
except FileNotFoundError:
    st.error(f"File `{FILE_NAME}` not found in the repository. Upload it to the repo root and rerun.")
except Exception as e:
    st.error(f"Error: {e}")
