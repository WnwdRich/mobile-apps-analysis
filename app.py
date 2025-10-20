import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.optimize import linear_sum_assignment

# =========================
# App config
# =========================
st.set_page_config(page_title="Mobile Apps – Slides", layout="wide")
st.title("Mobile Apps – Concise Slides (Optimal Alignment)")

FILE_NAME = "GoogleAppData.xlsx"  # ensure file is in repo root

# =========================
# Load data
# =========================
@st.cache_data
def load_excel(path: str):
    xls = pd.ExcelFile(path)
    data = pd.read_excel(xls, xls.sheet_names[0])                   # Sheet 1: dataset
    dictionary = pd.read_excel(xls, xls.sheet_names[1]) if len(xls.sheet_names) > 1 else None  # Sheet 2: definitions
    return data, dictionary, xls.sheet_names

# =========================
# Validators & parsing helpers
# =========================
def find_col(cols, *aliases):
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low: return low[a.lower()]
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
    return val

def build_validators(cols):
    """Return {col_name: validator(value)->bool} using name heuristics."""
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

def row_fail_score(row, validators):
    """Count how many validator checks fail in this row."""
    score = 0
    for col, fn in validators.items():
        val = row.get(col, np.nan)
        if not fn(val):
            score += 1
    return score

# =========================
# Optimal cell-level alignment (Hungarian)
# =========================
def optimal_align_row(row: pd.Series,
                      validators: dict,
                      window: int = 2,
                      protect_first_cols: int = 1,
                      distance_weight: float = 1.0,
                      invalid_penalty: float = 100.0,
                      stay_bias: float = 0.0):
    """
    Reassign only the cells that look wrong, using Hungarian assignment:
    - Agents = suspicious (non-null & invalid) cells
    - Tasks  = candidate target columns within ±window that validate that value
              (+ fallback including original column with high invalid_penalty)
    We keep columns that already validate as FIXED (not changed).
    Returns: corrected_row (Series), changes (list of dict), improved(bool), delta_fail(int)
    """
    cols = list(row.index)
    n = len(cols)
    col_idx = {c:i for i,c in enumerate(cols)}
    vals = row.values.copy()

    # Build validators map for quick check
    def is_valid(col, v):
        fn = validators.get(col)
        return bool(fn and fn(v))

    # Fixed columns: non-null & valid (or within protected first N columns)
    fixed_mask = np.zeros(n, dtype=bool)
    for j, c in enumerate(cols):
        v = vals[j]
        if j < protect_first_cols and pd.notna(v):
            fixed_mask[j] = True
        elif pd.notna(v) and is_valid(c, v):
            fixed_mask[j] = True

    # Suspicious cells set (non-null & not fixed)
    mis_cols = [j for j,c in enumerate(cols) if (pd.notna(vals[j]) and not fixed_mask[j])]
    if not mis_cols:
        return row, [], False, 0  # nothing to do

    # Candidate target columns (free columns only)
    free_cols = [j for j in range(n) if not fixed_mask[j]]

    # Build cost matrix for Hungarian: shape (len(mis_cols), len(free_cols))
    # Cost rules:
    # - If value validates in target col: cost = distance_weight * |j - t|  (+ small bias for staying put)
    # - If not valid: cost = invalid_penalty + distance_weight * |j - t|
    # We’ll solve and then only apply moves that land on valid targets.
    cost = np.zeros((len(mis_cols), len(free_cols)))
    valid_matrix = np.zeros_like(cost, dtype=bool)

    for a_idx, j in enumerate(mis_cols):
        v = vals[j]
        for b_idx, t in enumerate(free_cols):
            col_t = cols[t]
            dist = abs(j - t)
            valid_here = is_valid(col_t, v)
            valid_matrix[a_idx, b_idx] = valid_here
            cst = (distance_weight * dist) + (0.0 if (t != j) else stay_bias)
            if not valid_here:
                cst += invalid_penalty
            cost[a_idx, b_idx] = cst

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = list(zip([mis_cols[i] for i in row_ind], [free_cols[j] for j in col_ind]))

    # Build corrected copy, but only apply moves that land on VALID targets
    corrected = row.copy()
    changes = []
    used_targets = set()

    for orig_j, tgt_j in assigned:
        v = vals[orig_j]
        tgt_col = cols[tgt_j]
        # Only move if target validates and not already used
        if valid_matrix[row_ind[list(mis_cols).index(orig_j)],
                        col_ind[list(free_cols).index(tgt_j)]] and (tgt_j not in used_targets):
            # Don't overwrite a currently valid (fixed) target (shouldn't be in free_cols, but double-check)
            if fixed_mask[tgt_j]:
                continue
            # Move value
            corrected.iat[tgt_j] = v
            corrected.iat[orig_j] = np.nan
            used_targets.add(tgt_j)
            changes.append({"from": cols[orig_j], "to": tgt_col, "value": v})

    # Evaluate improvement
    before = row_fail_score(row, validators)
    after  = row_fail_score(corrected, validators)
    improved = after < before
    return corrected, changes, improved, (before - after)

def optimal_align_dataframe(df: pd.DataFrame,
                            window: int = 2,
                            protect_first_cols: int = 1,
                            distance_weight: float = 1.0,
                            invalid_penalty: float = 100.0,
                            stay_bias: float = 0.0,
                            min_improvement: int = 1):
    """
    Apply optimal alignment row-wise; only accept a row’s reassignment if fail-score improves
    by at least min_improvement.
    """
    validators = build_validators(df.columns)
    corrected = df.copy()
    all_changes = []
    improved_rows = 0

    for i in range(len(df)):
        row = df.iloc[i]
        fixed_row, changes, improved, delta = optimal_align_row(
            row, validators,
            window=window,
            protect_first_cols=protect_first_cols,
            distance_weight=distance_weight,
            invalid_penalty=invalid_penalty,
            stay_bias=stay_bias
        )
        if improved and delta >= min_improvement:
            corrected.iloc[i] = fixed_row
            if changes:
                for ch in changes:
                    ch["row"] = df.index[i]
                all_changes.extend(changes)
            improved_rows += 1

    changes_df = pd.DataFrame(all_changes)
    return corrected, changes_df, improved_rows

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
    # Slide 2 – Data Quality (optimal alignment → duplicates → missing)
    # -------------------------
    with slide2:
        st.subheader("Optimal cell-level alignment (Hungarian)")
        st.caption("Reassign only suspicious cells to nearby valid columns with global optimal matching per row.")

        c1, c2, c3 = st.columns(3)
        with c1:
            window = st.slider("Candidate window (± columns)", 1, 6, 2)
        with c2:
            min_improv = st.slider("Min improvement to accept", 1, 5, 1)
        with c3:
            protect = st.number_input("Protect first N columns", min_value=0, max_value=len(df.columns), value=1)

        c4, c5, c6 = st.columns(3)
        with c4:
            dist_w = st.slider("Distance weight", 0.0, 5.0, 1.0, 0.1)
        with c5:
            invalid_pen = st.slider("Invalid-placement penalty", 10.0, 500.0, 100.0, 10.0)
        with c6:
            stay_bias = st.slider("Stay-in-place bias (lower = easier to move)", -2.0, 2.0, 0.0, 0.1)

        corrected_df, changes_df, improved_rows = optimal_align_dataframe(
            df,
            window=window,
            protect_first_cols=protect,
            distance_weight=dist_w,
            invalid_penalty=invalid_pen,
            stay_bias=stay_bias,
            min_improvement=min_improv
        )

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Total rows", len(df))
        with m2: st.metric("Rows improved", improved_rows)
        with m3: st.metric("Total cell moves", 0 if changes_df.empty else len(changes_df))

        with st.expander("Show cell moves (first 200)"):
            if changes_df.empty:
                st.write("No cell-level moves applied.")
            else:
                st.dataframe(changes_df.head(200), use_container_width=True)

        with st.expander("Compare original vs corrected (sample of changed rows)"):
            if changes_df.empty:
                st.write("No differences to show.")
            else:
                sample_idx = list(changes_df["row"].unique())[:10]
                side_by_side = pd.concat(
                    {"original": df.loc[sample_idx], "corrected": corrected_df.loc[sample_idx]},
                    axis=1
                )
                st.dataframe(side_by_side, use_container_width=True)

        st.markdown("---")
        st.subheader("Duplicate Records (after alignment)")
        dup_mask = corrected_df.duplicated(keep=False)
        dups = corrected_df[dup_mask]
        d1, d2, d3 = st.columns(3)
        with d1: st.metric("Rows (aligned)", len(corrected_df))
        with d2: st.metric("Duplicate rows", len(dups))
        with d3: st.metric("Duplicate %", f"{(len(dups)/len(corrected_df)*100 if len(corrected_df) else 0):.2f}%")
        st.dataframe(dups if not dups.empty else pd.DataFrame({"note": ["No duplicates found."]}), use_container_width=True)

        st.markdown("---")
        st.subheader("Missing values per column (after alignment)")
        missing_after = corrected_df.isna().sum().rename("Missing Values").sort_values(ascending=False)
        st.dataframe(missing_after.to_frame(), use_container_width=True)

        with st.expander("Method – short explanation"):
            st.write(
                "- Build simple validators per column (Rating ∈ [0,5], Installs/Reviews ≥ 0, Size parseable 0–5120MB, App/Category look like text).\n"
                "- For each row, identify only cells that fail their column.\n"
                "- Consider candidate target columns within ±window where that value would validate; cost = distance_weight·|Δpos|, invalid targets add a big penalty.\n"
                "- Solve a global minimum-cost assignment (Hungarian) across the suspicious cells; apply moves only if row fail-score improves by ≥ threshold.\n"
                "- Compute duplicates and missing values on the aligned table."
            )

except FileNotFoundError:
    st.error(f"File `{FILE_NAME}` not found in the repository. Upload it to the repo root and rerun.")
except Exception as e:
    st.error(f"Error: {e}")
