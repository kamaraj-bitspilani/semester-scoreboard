import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import os

st.set_page_config(
    page_title="Semester Scoreboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📘 Semester Scoreboard Dashboard")

# Handle file paths for cloud deployment (generic path)
import tempfile
import os

# Use current working directory for file storage - works in both local and cloud environments
DATA_FILE = "marks.csv"

# Define subject names first
STATS_NAME = "Statistics"
ML_NAME = "Machine Learning"
DNN_NAME = "Deep Neural Networks"
MATH_NAME = "Mathematics"

SUBJECTS = [STATS_NAME, ML_NAME, DNN_NAME, MATH_NAME]

REQUIRED_COLUMNS = [
    "Subject",
    "EC1 Quiz",
    "EC1 Assignment",
    # ML
    "ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2",
    # STATS
    "ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2",
    # MATH
    "MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2",
    # DNN
    "DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3",
    # Computed + others
    "EC1 Total", "EC2 Mid-Sem", "EC3 Final", "Grand Total", "Grade", "Status"
]

# Load CSV with better error handling for cloud environments
def load_saved_df() -> pd.DataFrame:
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            # Validate required columns exist
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                st.warning(f"CSV missing columns: {missing_cols}. Using default data...")
                return create_default_df()
            return df
        else:
            # File doesn't exist, create default data
            return create_default_df()
    except Exception as e:
        st.warning(f"Could not read marks file. Using default data. (Error: {e})")
        return create_default_df()

def create_default_df() -> pd.DataFrame:
    """Create default DataFrame with all required columns"""
    init_rows = []
    for s in SUBJECTS:
        init_rows.append({
            "Subject": s,
            "EC1 Quiz": 0.0, "EC1 Assignment": 0.0,
            "ML Quiz 1": 0.0, "ML Quiz 2": 0.0, "ML Assignment 1": 0.0, "ML Assignment 2": 0.0,
            "ST Quiz 1": 0.0, "ST Quiz 2": 0.0, "ST Assignment 1": 0.0, "ST Assignment 2": 0.0,
            "MATH Quiz 1": 0.0, "MATH Quiz 2": 0.0, "MATH Assignment 1": 0.0, "MATH Assignment 2": 0.0,
            "DNN Quiz 1": 0.0, "DNN Quiz 2": 0.0, "DNN Assignment 1": 0.0, "DNN Assignment 2": 0.0, "DNN Assignment 3": 0.0,
            "EC1 Total": 0.0, "EC2 Mid-Sem": 0.0, "EC3 Final": 0.0, "Grand Total": 0.0,
            "Grade": "F", "Status": "FAIL"
        })
    return pd.DataFrame(init_rows, columns=REQUIRED_COLUMNS)

# Initialize data and load saved marks (with session state backup for cloud)
if 'df_data' not in st.session_state:
    st.session_state.df_data = load_saved_df()

df_loaded = st.session_state.df_data

# Ensure all required editable columns exist and are float
editable_cols = [
    "EC1 Quiz", "EC1 Assignment", "EC2 Mid-Sem", "EC3 Final",
    # ML
    "ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2",
    # STATS
    "ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2",
    # MATH
    "MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2",
    # DNN
    "DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3",
]
for col in editable_cols:
    if col not in df_loaded.columns:
        df_loaded[col] = 0.0
    df_loaded[col] = pd.to_numeric(df_loaded[col], errors="coerce").astype(float).fillna(0.0)

# Build editors
st.subheader("✍️ Enter / Edit Marks")

# Simple (non-special) subjects editor
simple_rows = df_loaded[~df_loaded["Subject"].isin([ML_NAME, DNN_NAME, STATS_NAME, MATH_NAME])].copy()
simple_cols = ["Subject", "EC1 Quiz", "EC1 Assignment", "EC2 Mid-Sem", "EC3 Final"]
simple_config = {
    "Subject": st.column_config.TextColumn("Subject", disabled=True),
    "EC1 Quiz": st.column_config.NumberColumn("EC1 Quiz", min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
    "EC1 Assignment": st.column_config.NumberColumn("EC1 Assignment", min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
    "EC2 Mid-Sem": st.column_config.NumberColumn("EC2 Mid-Sem (30)", min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
    "EC3 Final": st.column_config.NumberColumn("EC3 Final (40)", min_value=0.0, max_value=40.0, step=0.1, format="%.1f"),
}
if not simple_rows.empty:
    edited_simple = st.data_editor(simple_rows[simple_cols], column_config=simple_config, num_rows="fixed", use_container_width=True, key="editor_simple")
else:
    edited_simple = simple_rows[simple_cols]

# DNN editor
st.markdown("#### Deep Neural Networks (Special EC1)")
st.caption("EC1 = max(Quiz 1, Quiz 2) + (A1 + A2 + A3) × 5/6 (out of 30)")
dnn_rows = df_loaded[df_loaded["Subject"] == DNN_NAME].copy()
dnn_cols = ["Subject", "DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3", "EC2 Mid-Sem", "EC3 Final"]
dnn_config = {
    "Subject": st.column_config.TextColumn("Subject", disabled=True),
    "DNN Quiz 1": st.column_config.NumberColumn("Quiz 1", min_value=0.0, step=0.1, format="%.1f"),
    "DNN Quiz 2": st.column_config.NumberColumn("Quiz 2", min_value=0.0, step=0.1, format="%.1f"),
    "DNN Assignment 1": st.column_config.NumberColumn("Assignment 1", min_value=0.0, step=0.1, format="%.1f"),
    "DNN Assignment 2": st.column_config.NumberColumn("Assignment 2", min_value=0.0, step=0.1, format="%.1f"),
    "DNN Assignment 3": st.column_config.NumberColumn("Assignment 3", min_value=0.0, step=0.1, format="%.1f"),
    "EC2 Mid-Sem": st.column_config.NumberColumn("EC2 Mid-Sem (30)", min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
    "EC3 Final": st.column_config.NumberColumn("EC3 Final (40)", min_value=0.0, max_value=40.0, step=0.1, format="%.1f"),
}
edited_dnn = st.data_editor(dnn_rows[dnn_cols], column_config=dnn_config, num_rows="fixed", use_container_width=True, key="editor_dnn")

# ML editor
st.markdown("#### Machine Learning (Special EC1)")
st.caption("EC1 = max(Quiz 1, Quiz 2) + Assignment 1 + Assignment 2 (out of 30)")
ml_rows = df_loaded[df_loaded["Subject"] == ML_NAME].copy()
ml_cols = ["Subject", "ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2", "EC2 Mid-Sem", "EC3 Final"]
ml_config = {
    "Subject": st.column_config.TextColumn("Subject", disabled=True),
    "ML Quiz 1": st.column_config.NumberColumn("Quiz 1 (10)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
    "ML Quiz 2": st.column_config.NumberColumn("Quiz 2 (10)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
    "ML Assignment 1": st.column_config.NumberColumn("Assignment 1 (5)", min_value=0.0, max_value=5.0, step=0.1, format="%.1f"),
    "ML Assignment 2": st.column_config.NumberColumn("Assignment 2 (15)", min_value=0.0, max_value=15.0, step=0.1, format="%.1f"),
    "EC2 Mid-Sem": st.column_config.NumberColumn("EC2 Mid-Sem (30)", min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
    "EC3 Final": st.column_config.NumberColumn("EC3 Final (40)", min_value=0.0, max_value=40.0, step=0.1, format="%.1f"),
}
edited_ml = st.data_editor(ml_rows[ml_cols], column_config=ml_config, num_rows="fixed", use_container_width=True, key="editor_ml")

# Statistics editor
st.markdown("#### Statistics (Special EC1)")
st.caption("EC1 = Quiz 1 + Quiz 2 + Assignment 1 + Assignment 2 (out of 30)")
st_rows = df_loaded[df_loaded["Subject"] == STATS_NAME].copy()
st_cols = ["Subject", "ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2", "EC2 Mid-Sem", "EC3 Final"]
st_config = {
    "Subject": st.column_config.TextColumn("Subject", disabled=True),
    "ST Quiz 1": st.column_config.NumberColumn("Quiz 1 (5)", min_value=0.0, max_value=5.0, step=0.1, format="%.1f"),
    "ST Quiz 2": st.column_config.NumberColumn("Quiz 2 (5)", min_value=0.0, max_value=5.0, step=0.1, format="%.1f"),
    "ST Assignment 1": st.column_config.NumberColumn("Assignment 1 (10)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
    "ST Assignment 2": st.column_config.NumberColumn("Assignment 2 (10)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
    "EC2 Mid-Sem": st.column_config.NumberColumn("EC2 Mid-Sem (30)", min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
    "EC3 Final": st.column_config.NumberColumn("EC3 Final (40)", min_value=0.0, max_value=40.0, step=0.1, format="%.1f"),
}
edited_st = st.data_editor(st_rows[st_cols], column_config=st_config, num_rows="fixed", use_container_width=True, key="editor_stats")

# Mathematics editor
st.markdown("#### Mathematics (Special EC1)")
st.caption("EC1 = Quiz 1 + Quiz 2 + Assignment 1 + Assignment 2 (out of 30)")
math_rows = df_loaded[df_loaded["Subject"] == MATH_NAME].copy()
math_cols = ["Subject", "MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2", "EC2 Mid-Sem", "EC3 Final"]
math_config = {
    "Subject": st.column_config.TextColumn("Subject", disabled=True),
    "MATH Quiz 1": st.column_config.NumberColumn("Quiz 1 (5)", min_value=0.0, max_value=5.0, step=0.1, format="%.1f"),
    "MATH Quiz 2": st.column_config.NumberColumn("Quiz 2 (5)", min_value=0.0, max_value=5.0, step=0.1, format="%.1f"),
    "MATH Assignment 1": st.column_config.NumberColumn("Assignment 1 (10)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
    "MATH Assignment 2": st.column_config.NumberColumn("Assignment 2 (10)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
    "EC2 Mid-Sem": st.column_config.NumberColumn("EC2 Mid-Sem (30)", min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
    "EC3 Final": st.column_config.NumberColumn("EC3 Final (40)", min_value=0.0, max_value=40.0, step=0.1, format="%.1f"),
}
edited_math = st.data_editor(math_rows[math_cols], column_config=math_config, num_rows="fixed", use_container_width=True, key="editor_math")

# Combine edited inputs
edited_simple = edited_simple.copy()
edited_dnn = edited_dnn.copy()
edited_ml = edited_ml.copy()
edited_st = edited_st.copy()
edited_math = edited_math.copy()

# Ensure all numeric fields accept and preserve floating-point values
def coerce_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

numeric_cols_common = ["EC1 Quiz", "EC1 Assignment", "EC2 Mid-Sem", "EC3 Final"]
numeric_cols_dnn = ["DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3"]
numeric_cols_ml = ["ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2"]
numeric_cols_stats = ["ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2"]
numeric_cols_math = ["MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2"]

coerce_numeric(edited_simple, numeric_cols_common)
coerce_numeric(edited_dnn, numeric_cols_dnn + numeric_cols_common)
coerce_numeric(edited_ml, numeric_cols_ml + numeric_cols_common)
coerce_numeric(edited_st, numeric_cols_stats + numeric_cols_common)
coerce_numeric(edited_math, numeric_cols_math + numeric_cols_common)

# Compute EC1 totals
def compute_ec1(row: pd.Series) -> float:
    if row.get("Subject") == DNN_NAME:
        q1 = float(row.get("DNN Quiz 1", 0) or 0)
        q2 = float(row.get("DNN Quiz 2", 0) or 0)
        a1 = float(row.get("DNN Assignment 1", 0) or 0)
        a2 = float(row.get("DNN Assignment 2", 0) or 0)
        a3 = float(row.get("DNN Assignment 3", 0) or 0)
        return max(q1, q2) + (a1 + a2 + a3) * 5.0 / 6.0
    elif row.get("Subject") == ML_NAME:
        q1 = float(row.get("ML Quiz 1", 0) or 0)
        q2 = float(row.get("ML Quiz 2", 0) or 0)
        a1 = float(row.get("ML Assignment 1", 0) or 0)
        a2 = float(row.get("ML Assignment 2", 0) or 0)
        return max(q1, q2) + a1 + a2
    elif row.get("Subject") == STATS_NAME:
        q1 = float(row.get("ST Quiz 1", 0) or 0)
        q2 = float(row.get("ST Quiz 2", 0) or 0)
        a1 = float(row.get("ST Assignment 1", 0) or 0)
        a2 = float(row.get("ST Assignment 2", 0) or 0)
        return q1 + q2 + a1 + a2
    elif row.get("Subject") == MATH_NAME:
        q1 = float(row.get("MATH Quiz 1", 0) or 0)
        q2 = float(row.get("MATH Quiz 2", 0) or 0)
        a1 = float(row.get("MATH Assignment 1", 0) or 0)
        a2 = float(row.get("MATH Assignment 2", 0) or 0)
        return q1 + q2 + a1 + a2
    else:
        e1 = float(row.get("EC1 Quiz", 0) or 0)
        e2 = float(row.get("EC1 Assignment", 0) or 0)
        return e1 + e2

# Align columns and merge
simple_aligned = edited_simple.copy()
for col in ["DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3"]:
    simple_aligned[col] = 0.0
for col in ["ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2"]:
    simple_aligned[col] = 0.0
for col in ["ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2"]:
    simple_aligned[col] = 0.0
for col in ["MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2"]:
    simple_aligned[col] = 0.0

ml_aligned = edited_ml.copy()
for col in ["DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3"]:
    ml_aligned[col] = 0.0
for col in ["EC1 Quiz", "EC1 Assignment"]:
    ml_aligned[col] = 0.0
for col in ["ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2"]:
    ml_aligned[col] = 0.0
for col in ["MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2"]:
    ml_aligned[col] = 0.0

dnn_aligned = edited_dnn.copy()
for col in ["ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2"]:
    dnn_aligned[col] = 0.0
for col in ["EC1 Quiz", "EC1 Assignment"]:
    dnn_aligned[col] = 0.0
for col in ["ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2"]:
    dnn_aligned[col] = 0.0
for col in ["MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2"]:
    dnn_aligned[col] = 0.0

st_aligned = edited_st.copy()
for col in ["ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2"]:
    st_aligned[col] = 0.0
for col in ["EC1 Quiz", "EC1 Assignment"]:
    st_aligned[col] = 0.0
for col in ["DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3"]:
    st_aligned[col] = 0.0
for col in ["MATH Quiz 1", "MATH Quiz 2", "MATH Assignment 1", "MATH Assignment 2"]:
    st_aligned[col] = 0.0

math_aligned = edited_math.copy()
for col in ["ML Quiz 1", "ML Quiz 2", "ML Assignment 1", "ML Assignment 2"]:
    math_aligned[col] = 0.0
for col in ["EC1 Quiz", "EC1 Assignment"]:
    math_aligned[col] = 0.0
for col in ["DNN Quiz 1", "DNN Quiz 2", "DNN Assignment 1", "DNN Assignment 2", "DNN Assignment 3"]:
    math_aligned[col] = 0.0
for col in ["ST Quiz 1", "ST Quiz 2", "ST Assignment 1", "ST Assignment 2"]:
    math_aligned[col] = 0.0

merged_inputs = pd.concat([simple_aligned, dnn_aligned, ml_aligned, st_aligned, math_aligned], ignore_index=True)

# Compute totals and grades from merged inputs
df = merged_inputs.copy()
df["EC1 Total"] = df.apply(compute_ec1, axis=1)
df["Grand Total"] = df["EC1 Total"].astype(float) + df["EC2 Mid-Sem"].astype(float) + df["EC3 Final"].astype(float)

def grade_for_total(t: float) -> str:
    if t >= 75:
        return "A"
    if t >= 65:
        return "B"
    if t >= 55:
        return "C"
    if t >= 45:
        return "D"
    return "F"

df["Grade"] = df["Grand Total"].apply(grade_for_total)
df["Status"] = df["Grade"].apply(lambda g: "PASS" if g != "F" else "FAIL")

df = pd.DataFrame(df)

st.divider()
st.subheader("📊 Semester Summary")

# Define the column order for display - showing key columns with EC2 and EC3 before Grand Total
display_columns = [
    "Subject", "EC1 Total", "EC2 Mid-Sem", "EC3 Final", "Grand Total", "Grade", "Status"
]

# Ensure all display columns exist in the dataframe
display_df = df.copy()
for col in display_columns:
    if col not in display_df.columns:
        display_df[col] = 0.0 if col not in ["Subject", "Grade", "Status"] else ""

# Format numeric columns
numeric_cols = ["EC1 Total", "EC2 Mid-Sem", "EC3 Final", "Grand Total"]
format_map = {col: "{:.1f}" for col in numeric_cols}

st.dataframe(display_df[display_columns].style.format(format_map), use_container_width=True)

# Save button with cloud-compatible error handling
if st.button(f"💾 Save changes to {DATA_FILE}"):
    try:
        # Ensure all required columns exist in save output
        save_df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in save_df.columns:
                save_df[col] = 0.0
        # Reorder columns for consistency
        save_df = save_df[REQUIRED_COLUMNS]
        save_df.to_csv(DATA_FILE, index=False)
        # Update session state
        st.session_state.df_data = save_df
        st.success(f"✅ Saved marks successfully!")
    except PermissionError:
        # Update session state even if file save fails
        st.session_state.df_data = df.copy()
        st.warning("📝 Data saved to session (file save restricted in cloud environment)")
    except Exception as e:
        # Update session state as fallback
        st.session_state.df_data = df.copy()
        st.warning(f"📝 Data saved to session. File save error: {str(e)[:50]}...")

# Charts
st.subheader("📈 Performance Overview")
bar = alt.Chart(df).mark_bar().encode(
    x=alt.X("Subject:N", sort=None),
    y=alt.Y("Grand Total:Q", axis=alt.Axis(format=".1f")),
    tooltip=["Subject", alt.Tooltip("Grand Total:Q", format=".1f"), alt.Tooltip("EC1 Total:Q", format=".1f")]
).properties(width=700)
st.altair_chart(bar, use_container_width=True)

# CGPA preview (simple mapping)
grade_points = {"A": 10, "B": 8, "C": 6, "D": 5, "F": 0}
df["GP"] = df["Grade"].map(grade_points)
cgpa = round(df["GP"].mean(), 2)

st.success(f"📌 Estimated CGPA: **{cgpa}**")

# Footer
st.divider()
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        📊 Semester Scoreboard Dashboard | Built with Streamlit<br>
        💡 Track your academic progress and visualize your performance
    </div>
    """, 
    unsafe_allow_html=True
)