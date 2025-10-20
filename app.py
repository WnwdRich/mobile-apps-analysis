import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Mobile Apps Dataset", layout="wide")
st.title("Mobile Apps Dataset – Data Quality Checks")

FILE_NAME = "GoogleAppData.xlsx"

@st.cache_data
def load_excel(path: str):
    xls = pd.ExcelFile(path)
    data = pd.read_excel(xls, xls.sheet_names[0])
    dictionary = pd.read_excel(xls, xls.sheet_names[1]) if len(xls.sheet_names) > 1 else None
    return data, dictionary, xls.sheet_names

# ---------- helpers ----------
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
    # case-insensitive finder
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low: return low[a.lower()]
    # contains match (e.g., "rating" inside "Average Rating")
    for c in cols:
        if any(a.lower() in c.lower() for a in aliases):
            return c
    return None

def row_fail_score(row, cols):
    """
    Returns number of validation failures in row for known columns.
    """
    score = 0
    # find columns on every call (robust to different files)
    c_rating  = find_col(cols, "rating", "ratings")
    c_reviews = find_col(cols, "reviews", "review_count", "num_reviews")
    c_inst    = find_col(cols, "installs", "downloads")
    c_size    = find_col(cols, "size", "app_size")

    # rating 0..5
    if c_rating and not pd.isna(row[c_rating]):
        val = pd.to_numeric(row[c_rating], errors="coerce")
        if pd.isna(val) or val < 0 or val > 5: score += 1

    # reviews >= 0 integer-like
    if c_reviews and not pd.isna(row[c_reviews]):
        val = pd.to_numeric(row[c_reviews], errors="coerce")
        if pd.isna(val) or val < 0: score += 1

    # installs >= 0 integer-like (after cleaning)
    if c_inst and not pd.isna(row[c_inst]):
        val = to_int_like(row[c_inst])
        if pd.isna(val) or val < 0: score += 1

    # size parseable
    if c_size and not pd.isna(row[c_size]):
        val = size_to_mb(row[c_size])
        if pd.isna(val) or val <= 0 or val > 5120:  # sanity: 0..5GB
            score += 1

    return score

def shift_row_right_by_1(row):
    vals = row.values
    shifted = np.empty_like(vals, dtype=object)
    shifted[0] = np.nan
    shifted[1:] = vals[:-1]
    return pd.Series(shifted, index=row.index)

try:
    df, dict_df, sheets = load_excel(FILE_NAME)
    st.caption(f"Loaded **{FILE_NAME}** | Sheets: {', '.join(sheets)}")

    tab_data, tab_schema, tab_duplicates, tab_missing, tab_dict = st.tabs(
        ["📊 Data (Sheet 1)", "🧪 Schema & Misalignment", "🔁 Duplicate Records", "⚠️ Missing Values", "📖 Dictionary"]
    )

    # ---- Data preview ----
    with tab_data:
        st.subheader("First 50 rows of dataset")
        st.dataframe(df.head(50), use_container_width=True)

    # ---- Schema & Misalignment tests ----
    with tab_schema:
        st.subheader("Row validation & misalignment detection")
        st.caption("Detect rows where typed expectations fail, and check if a 1-column right shift reduces failures (indicating a left-shifted row).")

        # Compute original fail score
        orig_scores = df.apply(lambda r: row_fail_score(r, df.columns), axis=1)

        # Compute score after simulating a 1-col right shift (to detect rows shifted left)
        shifted_df = df.apply(shift_row_right_by_1, axis=1)
        shifted_scores = shifted_df.apply(lambda r: row_fail_score(r, shifted_df.columns), axis=1)

        results = pd.DataFrame({
            "orig_fail_score": orig_scores,
            "shift_right_fail_score": shifted_scores,
            "improvement_if_shift_right": orig_scores - shifted_scores
        })

        # Thresholds / flags
        # Suspicious if original score >=2 and improves by >=2 after right-shift
        suspicious_mask = (results["orig_fail_score"] >= 2) & (results["improvement_if_shift_right"] >= 2)
        suspicious_idx = results[suspicious_mask].index

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Rows with ≥1 failed check", int((results["orig_fail_score"] >= 1).sum()))
        with c2: st.metric("Likely left-shifted rows", int(len(suspicious_idx)))
        with c3: st.metric("Total rows", len(df))

        st.write("**Validation results (top 200 by failure score):**")
        to_show = results.sort_values("orig_fail_score", ascending=False).head(200)
        st.dataframe(to_show, use_container_width=True)

        if len(suspicious_idx) > 0:
            st.subheader("Likely misaligned (left-shifted) rows")
            st.caption("Showing the original rows that look misaligned. Inspect them manually.")
            st.dataframe(df.loc[suspicious_idx].head(200), use_container_width=True)

            # Optional: show a side-by-side of original vs. 1-right-shift for first few rows
            with st.expander("Show original vs shifted (first 5 suspicious rows)"):
                sample_idx = suspicious_idx[:5]
                side_by_side = pd.concat(
                    {
                        "original": df.loc[sample_idx],
                        "shift_right_by_1": shifted_df.loc[sample_idx]
                    },
                    axis=1
                )
                st.dataframe(side_by_side, use_container_width=True)
        else:
            st.success("✅ No strongly suspicious misalignment patterns detected.")

        st.info(
            "Heuristics used: rating must be in [0,5], installs/reviews ≥ 0 (after cleaning), size parseable to MB (0–5120MB). "
            "We test a 1-column right shift to detect rows that slid left. Adjust thresholds if needed."
        )

    # ---- Duplicate detection ----
    with tab_duplicates:
        st.subheader("Detect duplicate rows (all columns identical)")
        duplicated_mask = df.duplicated(keep=False)
        duplicates = df[duplicated_mask]

        st.write(f"**Total duplicate rows:** {len(duplicates)} "
                 f"out of {len(df)} total rows "
                 f"({(len(duplicates)/len(df)*100):.2f}% duplicates).")

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

        st.write(f"**Columns with at least one missing value:** {(missing_count > 0).sum()} / {len(df.columns)}")
        st.dataframe(missing_summary, use_container_width=True)

        st.subheader("Missing % by column (bar)")
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
