# pages/pages/1_protein.py
import streamlit as st
import pandas as pd
import re
import io

st.set_page_config(page_title="Protein Import", layout="wide")

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header {
        margin: 0; padding: 0; box-sizing: border-box;
    }
    .header {
        background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
        padding: 20px 40px;
        color: white;
        margin: -60px -60px 40px -60px;
    }
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .fixed-restart {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        width: 340px;
    }
    .stButton>button {
        background:#E71316 !important;
        color:white !important;
        border-radius:8px !important;
    }
    .footer {
        text-align:center;
        padding:30px;
        color:#666;
        font-size:12px;
        border-top:1px solid #eee;
        margin-top:80px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Protein Data Import & Species Detection</h1></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RESTORE SESSION DATA
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state and not st.session_state.get("reconfigure_prot", False):
    df = st.session_state.prot_df
    c1 = st.session_state.prot_c1
    c2 = st.session_state.prot_c2
    sp_col = st.session_state.prot_sp_col
    sp_counts = st.session_state.prot_sp_counts

    st.success("Data restored from session")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Condition A", f"{len(c1)} replicates")
        st.write(" | ".join(c1))
    with col2:
        st.metric("Condition B", f"{len(c2)} replicates")
        st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            st.session_state.reconfigure_prot = True
            st.rerun()

    st.info(f"**Species column**: `{sp_col}` → detected {len(sp_counts)} species")

    if not sp_counts.empty:
        st.markdown("### Proteins per Species")
        st.dataframe(sp_counts, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)
        with c2:
            for _, row in sp_counts.iterrows():
                st.write(f"• **{row['Species']}**: {row['Total']} proteins ({row['A']} in A • {row['B']} in B)")

    # Restart button
    st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True):
        keys = [k for k in st.session_state.keys() if k.startswith("prot_") or k == "reconfigure_prot"]
        for k in keys:
            del st.session_state[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Allow re-upload during reconfiguration
if st.session_state.get("reconfigure_prot", False):
    st.warning("Reconfiguring — please upload the same file again")

# ─────────────────────────────────────────────────────────────
# 1. UPLOAD FILE
# ─────────────────────────────────────────────────────────────
st.markdown("### 1. Upload Protein Data (.csv, .tsv, .txt)")
uploaded_file = st.file_uploader("Drag and drop or click to browse", type=["csv", "txt", "tsv"])

if not uploaded_file:
    st.info("Please upload a file to continue")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading file...")
def load_file(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"):
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python")
    return df

df_raw = load_file(uploaded_file)
st.success(f"Loaded {len(df_raw):,} protein rows")

# Detect intensity columns (numeric)
intensity_cols = []
for col in df_raw.columns:
    numeric_series = pd.to_numeric(df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if numeric_series.notna().sum() > len(df_raw) * 0.3:  # at least 30% real numbers
        df_raw[col] = numeric_series
        intensity_cols.append(col)

if not intensity_cols:
    st.error("No intensity/replicate columns detected. Check file format.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 3. ASSIGN REPLICATES TO CONDITION A / B
# ─────────────────────────────────────────────────────────────
st.markdown("### 2. Assign Replicates to Conditions")
rows = []
for col in intensity_cols:
    preview = df_raw[col].dropna().head(3).astype(str).tolist()
    rows.append({
        "Column": col,
        "Preview": " | ".join(preview),
        "Condition A → A1,A2...": True,
        "Condition B → B1,B2...": False,
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
        "Preview": st.column_config.TextColumn(disabled=True),
        "Condition A → A1,A2...": st.column_config.CheckboxColumn("Condition A"),
        "Condition B → B1,B2...": st.column_config.CheckboxColumn("Condition B"),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

cond_a = edited[edited["Condition A → A1,A2..."]]["Column"].tolist()
cond_b = edited[edited["Condition B → B1,B2..."]]["Column"].tolist()

if len(cond_a) == 0 or len(cond_b) == 0:
    st.error("Both conditions must have at least one replicate")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 4. RENAME TO A1/A2... and B1/B2...
# ─────────────────────────────────────────────────────────────
rename_map = {}
for i, old_name in enumerate(cond_a):
    rename_map[old_name] = f"A{i+1}"
for i, old_name in enumerate(cond_b):
    rename_map[old_name] = f"B{i+1}"

df = df_raw.rename(columns=rename_map).copy()
final_c1 = [f"A{i+1}" for i in range(len(cond_a))]
final_c2 = [f"B{i+1}" for i in range(len(cond_b))]

st.success(f"Renamed → Condition A: {', '.join(final_c1)} | Condition B: {', '.join(final_c2)}")

# ─────────────────────────────────────────────────────────────
# 5. AUTO DETECT SPECIES COLUMN
# ─────────────────────────────────────────────────────────────
st.markdown("### 3. Auto-Detecting Species Column")

def find_species_column(df):
    species_keywords = ["HUMAN", "MOUSE", "RAT", "BOVIN", "YEAST", "RABIT", "CANFA", "PANTR", "MACMU", "CHICK"]
    pattern = "|".join(species_keywords)
    for col in df.columns:
        if col in final_c1 + final_c2:
            continue
        if df[col].astype(str).str.upper().str.contains(pattern).any():
            return col
    return None

species_column = find_species_column(df)

if not species_column:
    st.error("No column containing species names (HUMAN, MOUSE, etc.) found.")
    st.stop()

st.success(f"Species column detected: `{species_column}`")

# Extract clean species
def extract_species(val):
    if pd.isna(val):
        return "Unknown"
    text = str(val).upper()
    known = ["HUMAN", "MOUSE", "RAT", "BOVIN", "YEAST", "RABIT", "CANFA", "MACMU", "PANTR", "CHICK", "HORSE", "PIG"]
    for sp in known:
        if sp in text:
            return sp
    return "Other"

df["Species"] = df[species_column].apply(extract_species)

# ─────────────────────────────────────────────────────────────
# 6. COUNT PROTEINS PER SPECIES & CONDITION
# ─────────────────────────────────────────────────────────────
st.markdown("### 4. Protein Counts per Species")

threshold = 1  # intensity > 1 counts as detected

counts = []
for sp in df["Species"].unique():
    if sp in ["Unknown", "Other"] and len(df["Species"].unique()) > 2:
        continue
    sub = df[df["Species"] == sp]
    detected_in_a = (sub[final_c1] > threshold).any(axis=1).sum()
    detected_in_b = (sub[final_c2] > threshold).any(axis=1).sum()
    total = len(sub)

    counts.append({
        "Species": sp,
        "A": detected_in_a,
        "B": detected_in_b,
        "Total": total
    })

sp_counts = pd.DataFrame(counts).sort_values("Total", ascending=False)

# ─────────────────────────────────────────────────────────────
# 7. SAVE TO SESSION STATE
# ─────────────────────────────────────────────────────────────
st.session_state.update({
    "prot_df": df,
    "prot_c1": final_c1,
    "prot_c2": final_c2,
    "prot_sp_col": species_column,
    "prot_sp_counts": sp_counts,
    "reconfigure_prot": False,
})

# ─────────────────────────────────────────────────────────────
# 8. FINAL DISPLAY
# ─────────────────────────────────────────────────────────────
st.success("All set! Data is ready for analysis.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Condition A", ", ".join(final_c1))
with col2:
    st.metric("Condition B", ", ".join(final_c2))

st.markdown("### Proteins Detected per Species")
st.dataframe(sp_counts, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    chart_data = sp_counts.set_index("Species")[["A", "B"]]
    st.bar_chart(chart_data, use_container_width=True)
with col2:
    st.write("**Legend**")
    st.write("• **A** = detected in any replicate of Condition A")
    st.write("• **B** = detected in any replicate of Condition B")

# Final restart button
st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
if st.button("Restart Full Analysis", type="primary", use_container_width=True):
    keys = [k for k in st.session_state.keys() if k.startswith("prot_") or k == "reconfigure_prot"]
    for k in keys:
        del st.session_state[k]
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
