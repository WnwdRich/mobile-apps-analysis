import streamlit as st
import pandas as pd
import numpy as np
import re

# =========================
# App config
# =========================
st.set_page_config(page_title="Mobile Apps – Slides", layout="wide")
st.title("Mobile Apps – Concise Slides")

FILE_NAME = "GoogleAppData.xlsx"  # make sure this file is in the repo root

# =========================
# Data loading
# =========================
@st.cache_data
def load_excel(path: str):
    xls = pd.ExcelFile(path)
    data = pd.read_excel(xls, xls.sheet_names[0])                   # Sheet 1: dataset
    dictionary = pd.read_excel(xls, xls.sheet_names[1]) if len(xls.sheet_names) > 1 else None  # Sheet 2: definitions
    return data, dictionary, xls.sheet_names

# =========================
# Helpers for validation/alignment
# =========================
def find_col(cols, *aliases):
    """Find a column by exact (case-insensitive) or substring match from a list of aliases."""
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    for c in cols:
        if any(a.lower() in c.lower() for a in aliases):
            return c
    return None

def to_int_like(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", "").replace("+", "")
    return pd.to_numeric(s, errors="coerce")

def size_to_mb(x):
    """Parse sizes like '15M', '500k', '1.2G', 'Varies with device' → MB."""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace(" ", "")
    if "varies" in s: return np.nan
    m = re.match(r"^([\d\.]+)([kmg])?b?$", s)
    if not m: return pd.to_numeric(s, errors="coerce")
    val = float(m.group(1)); unit = m.group(2)
    if unit == "k": return val/1024.0
    if unit == "g": return val*1024.0
    return val  # MB default

def build_validators(df):
    """Return a dict {column_name: validator_fn(value)->bool} using column-name heuristics."""
    cols = list(df.columns)
    c_app  = find_col(cols, "app", "app_name", "title", "name")
    c_cat  = find_col(cols, "category", "categories")
    c_rate = find_col(cols, "rating", "ratings")
    c_rev  = find_col(cols, "reviews", "review_count", "num_reviews")
    c_inst = find_col(cols, "installs", "downloads")
    c_size = find_col(cols, "size", "app_size")

    def looks_text(v):
        if pd.isna(v): return False
        s = str(v).strip()
        return bool(re.search(r"[A-Za-z]", s))  # at least one letter

    validators = {}
    if c_app:  validators[c_app]  = lambda v: looks_text(v)
    if c_cat:  validators[c_cat]  = lambda v: looks_text(v)
    if c_rate: validators[c_rate] = lambda v: (pd.to_numeric(v, errors="coerce") is not None) and (0 <= float(pd.to_numeric(v, errors="coerce")) <= 5)
    if c_rev:  validators[c_rev]  = lambda v: (pd.to_numeric(v, errors="coerce") is not None) and (float(pd.to_numeric(v, errors="coerce")) >= 0)
    if c_inst: validators[c_inst] = lambda v: (to_int_like(v) is not None) and pd.notna(to_int_like(v)) and (to_int_like(v) >= 0)
    if c_size: validators[c_size] = lambda v: (size_to_mb(v) is not None) and pd.notna(size_to_mb(v)) and (0 < size_to_mb(v) <= 5120)
    return validators

def cellwise_repair(df, window=2, passes=2, protect_first_cols=1):
    """
    Try to fix partially misaligned rows by relocating individual cell values
    into a nearby column (±window) where they validate. Run multiple passes
    until no more moves happen.

    protect_first_cols: don't move values *out of* the first N columns (e.g., App name).
    """
    validators = build_validators(df)
    cols = list(df.columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    corrected = df.copy()
    changes = []

    for _ in range(passes):
        moved_any = False

        for ridx in range(len(corrected)):
            row = corrected.iloc[ridx].copy()

            for c in cols:
                j = col_idx[c]
                v = row[c]

                # nothing to move
                if pd.isna(v):
                    continue

                # if this column has a validator and passes -> keep it
                if c in validators and validators[c](v):
                    continue

                # don't move out of protected first columns
                if j < protect_first_cols:
                    continue

                # try to relocate into a nearby valid target
                best_target = None
                left = max(0, j - window)
                right = min(len(cols), j + window + 1)

                for t in range(left, right):
                    if t == j:
                        continue
                    target_col = cols[t]
                    target_val = row[target_col]
                    target_empty = pd.isna(target_val)
                    target_invalid = (target_col in validators) and (not target_empty) and (not validators[target_col](target_val))

                    if target_col in validators and validators[target_col](v) and (target_empty or target_invalid):
                        best_target = target_col
                        break  # greedy: first nearby valid slot

                if best_target:
                    corrected.at[corrected.index[ridx], best_target] = v
                    corrected.at[corrected.index[ridx], c] = np.nan
                    changes.append({"row": corrected.index[ridx], "from": c, "to": best_target, "value": v})
                    moved_any = True

        if not moved_any:
            break

    changes_df = pd.DataFrame(changes)
    return corrected, changes_df

# =========================
# Main app
# =========================
try:
    df, dict_df, sheets = load_excel(FILE_NAME)
    st.caption(f"Loaded **{FILE_NAME}** | Sheets: {', '.join(sheets)}")

    slide1, slide2 = st.tabs(["📄 Slide 1 – Dataset + Definitions", "🧪 Slide 2 – Data Quality"])

    # -------------------------
    # Slide 1 – Dataset + Definitions
    # -------------------------
    with slide1:
        st.subheader("Dataset (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Definitions (Sheet 2)")
        if dict_df is not None:
            st.dataframe(dict_df, use_container_width=True)
        else:
            st.info("No second sheet with column dictionary was found.")

    # -------------------------
    # Slide 2 – Data Quality (cell-level alignment → duplicates → missing)
    # -------------------------
    with slide2:
        st.subheader("Cell-level alignment")
        st.caption("Fix partially misaligned rows by relocating invalid cell values into nearby columns where they validate.")
        cols = st.columns(3)
        with cols[0]:
            w = st.slider("Relocation window (± columns)", 1, 4, 2)
        with cols[1]:
            passes = st.slider("Repair passes", 1, 5, 2)
        with cols[2]:
            protect = st.number_input("Protect first N columns", min_value=0, max_value=len(df.columns), value=1)

        corrected_df, changes_df = cellwise_repair(df, window=w, passes=passes, protect_first_cols=protect)

        c1, c2 = st.columns(2)
        with c1: st.metric("Rows changed", int(changes_df["row"].nunique() if not changes_df.empty else 0))
        with c2: st.metric("Total cell moves", len(changes_df))

        with st.expander("Show cell moves (first 200)"):
            if changes_df.empty:
                st.write("No cell-level moves suggested.")
            else:
                st.dataframe(changes_df.head(200), use_container_width=True)

        with st.expander("Compare original vs corrected (sample of changed rows)"):
            if changes_df.empty:
                st.write("No differences to show.")
            else:
                sample_idx = list(changes_df["row"].unique())[:10]
                side_by_side = pd.concat({"original": df.loc[sample_idx], "corrected": corrected_df.loc[sample_idx]}, axis=1)
                st.dataframe(side_by_side, use_container_width=True)

        st.markdown("---")
        st.subheader("Duplicate Records (after alignment)")
        dup_mask = corrected_df.duplicated(keep=False)
        dups = corrected_df[dup_mask]
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Total rows", len(corrected_df))
        with m2: st.metric("Duplicate rows", len(dups))
        with m3: st.metric("Duplicate %", f"{(len(dups)/len(corrected_df)*100 if len(corrected_df) else 0):.2f}%")
        st.dataframe(dups if not dups.empty else pd.DataFrame({"note": ["No duplicates found."]}), use_container_width=True)

        st.markdown("---")
        st.subheader("Missing values per column (after alignment)")
        missing_after = corrected_df.isna().sum().rename("Missing Values").sort_values(ascending=False)
        st.dataframe(missing_after.to_frame(), use_container_width=True)

        # short method explanation (for your slide notes)
        with st.expander("Method – misalignment test (short explanation)"):
            st.write(
                "- Validate each cell by the intended column type (e.g., Rating ∈ [0,5], Installs/Reviews ≥ 0, Size parseable 0–5120MB, App/Category look like text).\n"
                "- For cells that don't validate, try moving the value only a few columns left/right to a place where it *does* validate, without overwriting a clearly valid value.\n"
                "- Run multiple passes until no more improvements are found.\n"
                "- Then compute **duplicates** and **missing values** on the aligned table."
            )

except FileNotFoundError:
    st.error(f"File `{FILE_NAME}` not found in the repository. Upload it to the repo root and rerun.")
except Exception as e:
    st.error(f"Error: {e}")
