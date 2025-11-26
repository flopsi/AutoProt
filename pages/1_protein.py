# pages/1_protein.py
import streamlit as st
import pandas as pd
import re
import io

st.set_page_config(page_title="Protein Import", layout="wide")

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .stButton>button {background:#E71316 !important; color:white !important; border:none !important; border-radius:8px !important;}
    .stButton>button:hover {background:#c71a2e !important;}
    .fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 320px;}
    .footer {text-align:center; padding:30px; color:#666; font-size:12px; border-top:1px solid #eee; margin-top:60px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Protein Data Import & Species Detection</h1></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RESTORE DATA IF EXISTS
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state and not st.session_state.get("reconfigure_prot", False):
    df = st.session_state.prot_df
    c1 = st.session_state.prot_c1
    c2 = st.session_state.prot_c2
    sp_col = st.session_state.prot_sp_col
    sp_counts = st.session_state.prot_sp_counts

    st.success("Data restored from previous session")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Condition A", f"{len(c1)} replicates", help="Renamed to A1, A2, ...")
        st.write(" | ".join(c1))
    with col2:
        st.metric("Condition B", f"{len(c2)} replicates", help="Renamed to B1, B2, ...")
        st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            st.session_state.reconfigure_prot = True
            st.rerun()

    st.info(f"**Species column**: `{sp_col}`")
    
    if not sp_counts.empty:
        st.markdown("### Proteins Detected per Species")
        st.dataframe(sp_counts, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            chart_data = sp_counts.set_index("Species")[["A", "B", "Both"]]
            st.bar_chart(chart_data, use_container_width=True)
        with col2:
            st.write("**Summary**")
            for _, row in sp_counts.iterrows():
                st.write(f"• **{row['Species}**: {row['Total']} proteins ({row['A']} in A • {row['B']} in B • {row['Both']} in both)")

    # Restart button
    st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("prot_") or k == "reconfigure_prot":
                del st.session_state[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# RE-UPLOAD IF RECONFIGURING
# ─────────────────────────────────────────────────────────────
if st.session_state.get("reconfigure_prot", False):
    st.warning("Reconfiguring — please re-upload the same file")

st.markdown("### 1. Upload Your Protein Data")
uploaded_file = st.file_uploader("Supports .csv, .tsv, .txt", type=["csv", "tsv", "txt"])

if not uploaded_file:
    st.info("Upload a file to begin")
    st.stop()

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data(file):
    content = file.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python")
    return df

df_raw = load_data(uploaded_file)
st.success(f"Loaded {len(df_raw):,} proteins")

# Convert intensity columns to numeric
intensity_cols = []
for col in df_raw.columns:
    # Try converting to numeric — if many succeed → it's intensity
    sample = pd.to_numeric(df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if sample.notna().sum() > len(df_raw) * 0.3:  # at least 30% numeric
        df_raw[col] = sample
        intensity_cols.append(col)

st.write(f"Detected {len(intensity_cols)} intensity columns")

# ─────────────────────────────────────────────────────────────
# 2. ASSIGN CONDITIONS
# ─────────────────────────────────────────────────────────────
st.markdown("### 2. Assign Replicates to Conditions")

# Prepare editor
rows = []
for col in intensity_cols:
    rows.append({
        "Column": col,
        "Preview": " | ".join(df_raw[col].dropna().head(3).astype(str).tolist()),
        "Condition A": True,   # default to A
        "Condition B": False,
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
        "Preview": st.column_config.TextColumn(disabled=True),
        "Condition A": st.column_config.CheckboxColumn("Condition A → A1, A2, A3..."),
        "Condition B": st.column_config.CheckboxColumn("Condition B → B1, B2, B3..."),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

# Extract selections
cond_a_cols = edited[edited["Condition A"]]["Column"].tolist()
cond_b_cols = edited[edited["Condition B"]]["Column"].tolist()

if len(cond_a_cols) == 0 or len(cond_b_cols) == 0:
    st.error("Both conditions must have at least one replicate")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 3. RENAME REPLICATES: A1,A2,... B1,B2,...
# ─────────────────────────────────────────────────────────────
rename_map = {}
new_c1_names = [f"A{i+1}" for i in range(len(cond_a_cols))]
new_c2_names = [f"B{i+1}" for i in range(len(cond_b_cols))]

for old, new in zip(cond_a_cols, new_c1_names):
    rename_map[old] = new
for old, new in zip(cond_b_cols, new_c2_names):
    rename_map[old] = new

df = df_raw.rename(columns=rename_map).copy()
c1_final = new_c1_names
c2_final = new_c2_names

st.success(f"Renamed replicates → Condition A: {', '.join(c1_final)} | Condition B: {', '.join(c2_final)}")

# ─────────────────────────────────────────────────────────────
# 4. AUTO DETECT SPECIES COLUMN
# ─────────────────────────────────────────────────────────────
st.markdown("### 4. Detecting Species Information")

def find_species_column(df):
    candidates = []
    for col in df.columns:
        if col in c1_final + c2_final:
            continue  # skip intensity
        text_sample = df[col].dropna().astype(str).str.upper()
        if text_sample.str.contains("HUMAN|MOUSE|RAT|YEAST|BOVIN", regex=True).any():
            candidates.append((col, text_sample.str.contains("HUMAN|MOUSE|RAT|YEAST|BOVIN", regex=True).sum()))
    if not candidates:
        return None
    # Pick column with most species mentions
    return max(candidates, key=lambda x: x[1])[0]

species_column = find_species_column(df)

if not species_column:
    st.error("Could not find a column containing species names (HUMAN, MOUSE, etc.)")
    st.stop()

st.success(f"Species column detected: `{species_column}`")

# Extract clean species
def extract_species(text):
    if pd.isna(text):
        return "Unknown"
    text = str(text).upper()
    matches = re.findall(r"\b(HUMAN|MOUSE|RAT|BOVIN|YEAST|ARATH|ECOLI|DROME|CANFA|XENLA|PANTR|MACMU|PIG|CHICK|RABIT|HORSE)\b", text)
    return matches[0] if matches else "Other"

df["Detected_Species"] = df[species_column].apply(extract_species)
actual_species = df["Detected_Species"].value_counts().index.tolist()
if "Unknown" in actual_species:
    actual_species.remove("Unknown")
if "Other" in actual_species and len(actual_species) > 1:
    actual_species.remove("Other")

st.write("Species found:", ", ".join(actual_species) or "Only HUMAN")

# ─────────────────────────────────────────────────────────────
# 5. COUNT PROTEINS PER SPECIES & CONDITION
# ─────────────────────────────────────────────────────────────
st.markdown("### 5. Protein Counts per Species")

threshold = 1  # minimum intensity to count as "detected"

counts = []
for sp in ["HUMAN"] + [s for s in actual_species if s != "HUMAN"]:
    subset = df[df["Detected_Species"] == sp]
    in_a = (subset[c1_final] > threshold).any(axis=1).sum()
    in_b = (subset[c2_final] > threshold).any(axis=1).sum()
    in_both = (subset[c1_final] > threshold).all(axis=1) | (subset[c2_final] > threshold).all(axis=1)
    in_both = in_both.sum()
    total = len(subset)

    counts.append({
        "Species": sp,
        "A": in_a,
        "B": in_b,
        "Both": in_both,
        "Total": total
    })

sp_counts = pd.DataFrame(counts)

# ─────────────────────────────────────────────────────────────
# SAVE TO SESSION
# ─────────────────────────────────────────────────────────────
st.session_state.update({
    "prot_df": df,
    "prot_c1": c1_final,
    "prot_c2": c2_final,
    "prot_sp_col": species_column,
    "prot_sp_counts": sp_counts,
    "reconfigure_prot": False,  # reset flag
})

# ─────────────────────────────────────────────────────────────
# FINAL DISPLAY
# ─────────────────────────────────────────────────────────────
st.success("All done! Your data is ready for downstream analysis.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Condition A", ", ".join(c1_final))
with col2:
    st.metric("Condition B", ", ".join(c2_final))

st.markdown("### Proteins per Species")
st.dataframe(sp_counts, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)
with col2:
    st.write("**Legend**")
    st.write("• **A**: detected in Condition A (any replicate)")
    st.write("• **B**: detected in Condition B")
    st.write("• **Both**: detected in all replicates of A or B")

# Final restart
st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
if st.button("Restart Full Analysis", type="primary", use_container_width=True):
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith("prot_") or k == "reconfigure_prot"]
    for k in keys_to_clear:
        del st.session_state[k]
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
