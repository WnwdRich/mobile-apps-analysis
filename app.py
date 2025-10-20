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
    data = pd.read_excel(xls, xls.sheet_names[0])
    dictionary = pd.read_excel(xls, xls.sheet_names[1]) if len(xls.sheet_names) > 1 else None
    return data, dictionary, xls.sheet_names

# ---------- Helpers for misalignment ----------
def to_int_like(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", "").replace("+", "")
    return pd.to_numeric(s, errors="coerce")

def size_to_mb(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if "varies" in s: return np.nan
    s = s.replace(" ", "")
    m = re.match(r"^([\d\.]+)([kmg])?b?$", s)
    if not m:
        return pd.to_numeric(s, errors="coerce")
    val = float(m.group(1)); unit = m.group(2)
    if unit == "k": return val/1024
    if unit == "g": return val*1024
    return val  # MB default

def find_col(cols, *aliases):
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low: return low[a.lower()]
    for c in cols:
        if any(a.lower() in c.lower() for a in aliases):
            return c
    return None

def row_fail_score(row, cols):
    score = 0
    c_rating  = find_col(cols, "rating", "ratings")
    c_reviews = find_col(cols, "reviews", "review_count", "num_reviews")
    c_inst    = find_col(cols, "installs", "downloads")
    c_size    = find_col(cols, "size", "app_size")

    if c_rating and not pd.isna(row[c_rating]):
        val = pd.to_numeric(row[c_rating], errors="coerce")
        if pd.isna(val) or val < 0 or val > 5: score += 1

    if c_reviews and not pd.isna(row[c_reviews]):
        val = pd.to_numeric(row[c_reviews], errors="coerce")
        if pd.isna(val) or val < 0: score += 1

    if c_inst and not pd.isna(row[c_inst]):
        val = to_int_like(row[c_inst])
        if pd.isna(val) or val < 0: score += 1

    if c_size and not pd.isna(row[c_size]):
        val = size_to_mb(row[c_size])
        if pd.isna(val) or val <= 0 or val > 5120: score += 1  # sanity 0..5GB

    return score

def shift_row_right_by_1(row):
    vals = row.values
    shifted = np.empty_like(vals, dtype=object)
    shifted[0] = np.nan
    shifted[1:] = vals[:-1]
    return pd.Series(shifted, index=row.index)

# ---------- App ----------
try:
    df, dict_df, sheets = load_excel(FILE_NAME)
    st.caption(f"Loaded **{FILE_NAME}** | Sheets: {', '.join(sheets)}")

    slide1, slide2 = st.tabs(["📄 Slide 1 – Dataset + Definitions", "🧪 Slide 2 – Data Quality"])

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
        # --- Misalignment explanation ---
        st.subheader("Misaligned (left-shift) test – explanation")
        st.write(
            "- Validate each row (Rating 0–5, Installs/Reviews ≥ 0, Size parseable 0–5120MB). "
            "Then simulate a **1-column right shift** to detect rows likely left-shifted during export. "
            "If shifting reduces failures meaningfully, flag as **Likely misaligned**."
        )

        # --- Compute validation scores + misalignment detection ---
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

        c0, c1, c2 = st.columns(3)
        with c0: st.metric("Total rows", len(df))
        with c1: st.metric("Rows with ≥1 failed check", int((results["orig_fail_score"] >= 1).sum()))
        with c2: st.metric("Likely misaligned rows", int(len(suspicious_idx)))

        if len(suspicious_idx) > 0:
            st.write("**Likely misaligned (left-shifted) rows (original view):**")
            st.dataframe(df.loc[suspicious_idx], use_container_width=True)
        else:
            st.success("✅ No strongly suspicious misalignment patterns detected.")

        st.markdown("---")

        # --- Build corrected_df by fixing only suspicious rows ---
        st.subheader("Apply misalignment auto-fix (used for missing-values + duplicates)")
        apply_fix = st.checkbox("Auto-fix likely misaligned rows (shift right by 1 where flagged)", value=True)

        if apply_fix and len(suspicious_idx) > 0:
            corrected_df = df.copy()
            corrected_df.loc[suspicious_idx] = shifted_df.loc[suspicious_idx]
            st.caption(f"Applied right-shift to {len(suspicious_idx)} rows.")
        else:
            corrected_df = df.copy()
            st.caption("No auto-fix applied (using original data).")

        # --- Duplicates on corrected data ---
        st.subheader("Duplicate Records (after optional fix)")
        duplicated_mask = corrected_df.duplicated(keep=False)
        duplicates = corrected_df[duplicated_mask]

        d1, d2 = st.columns(2)
        with d1: st.metric("Duplicate rows", len(duplicates))
        with d2:
            pct = (len(duplicates)/len(corrected_df)*100) if len(corrected_df) else 0
            st.metric("Duplicate %", f"{pct:.2f}%")

        if not duplicates.empty:
            st.dataframe(duplicates, use_container_width=True)
        else:
            st.success("✅ No duplicate records found.")

        st.markdown("---")

        # --- Missing values: BEFORE vs AFTER ---
        st.subheader("Missing values per column (before vs after fix)")
        missing_before = df.isna().sum().rename("Missing (before)")
        missing_after  = corrected_df.isna().sum().rename("Missing (after)")
        missing_compare = pd.concat([missing_before, missing_after], axis=1)
        missing_compare["Δ fixed"] = (missing_compare["Missing (before)"] - missing_compare["Missing (after)"])
        missing_compare = missing_compare.sort_values("Missing (after)", ascending=False)

        st.dataframe(missing_compare, use_container_width=True)

        # also show the final “after” table only (your requested slide content)
        st.subheader("Missing values per column (after fix)")
        st.dataframe(missing_after.to_frame("Missing Values"), use_container_width=True)

except FileNotFoundError:
    st.error(f"File `{FILE_NAME}` not found in the repository. Upload it to the repo root and rerun.")
except Exception as e:
    st.error(f"Error: {e}")
