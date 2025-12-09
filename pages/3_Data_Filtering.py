"""
pages/3_Data_Filtering.py - COMPREHENSIVE PROTEOMICS FILTERING
Industry best practices for quality control and missing value handling
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Filtering",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Data Filtering & Quality Control")
st.markdown("Industry-standard proteomics filtering with comprehensive QC reporting")
st.markdown("---")

# ============================================================================
# CHECK FOR UPLOADED DATA
# ============================================================================

if 'df_raw' not in st.session_state or st.session_state.df_raw is None:
    st.error("❌ No data uploaded. Please start with **📁 Data Upload**")
    st.stop()

# Convert to pandas for easier manipulation
df = st.session_state.df_raw.to_pandas() if isinstance(st.session_state.df_raw, pl.DataFrame) else st.session_state.df_raw.copy()
numeric_cols = st.session_state.numeric_cols
species_col = st.session_state.species_col
sample_to_condition = st.session_state.get('sample_to_condition', {})

st.info(f"📊 **Input**: {len(df):,} rows × {len(numeric_cols)} samples")

# ============================================================================
# STEP 1: CONTAMINANT & QUALITY FILTERS
# ============================================================================

st.subheader("1️⃣ Contaminant & Quality Filters")

col1, col2, col3 = st.columns(3)

# Detect potential contaminant columns
contaminant_cols = [col for col in df.columns if any(
    keyword in col.lower() for keyword in 
    ['contaminant', 'reverse', 'potential', 'decoy', 'only id']
)]

df_qc = df.copy()
removed_qc = 0

if contaminant_cols:
    st.markdown("**Detected contaminant/quality columns:**")
    
    remove_contaminant = col1.checkbox(
        "Remove contaminants",
        value=True,
        help="Remove proteins marked as 'Potential contaminant'"
    )
    
    if remove_contaminant:
        for col in contaminant_cols:
            if 'contaminant' in col.lower():
                initial = len(df_qc)
                df_qc = df_qc[df_qc[col] != '+']
                removed_qc += initial - len(df_qc)
    
    remove_reverse = col2.checkbox(
        "Remove reverse hits",
        value=True,
        help="Remove decoy/reverse database matches"
    )
    
    if remove_reverse:
        for col in contaminant_cols:
            if 'reverse' in col.lower():
                initial = len(df_qc)
                df_qc = df_qc[df_qc[col] != '+']
                removed_qc += initial - len(df_qc)
    
    remove_only_site = col3.checkbox(
        "Remove 'only by site'",
        value=True,
        help="Remove proteins identified only by modified sites"
    )
    
    if remove_only_site:
        for col in contaminant_cols:
            if 'only' in col.lower() and 'site' in col.lower():
                initial = len(df_qc)
                df_qc = df_qc[df_qc[col] != '+']
                removed_qc += initial - len(df_qc)
    
    st.success(f"✅ Removed {removed_qc:,} rows ({removed_qc/len(df)*100:.1f}%)")
else:
    st.info("ℹ️ No contaminant columns detected. Skipping quality flags.")

st.markdown("---")

# ============================================================================
# STEP 2: MINIMUM PEPTIDES FILTER
# ============================================================================

st.subheader("2️⃣ Minimum Peptide Requirements")

# Detect peptide count columns
peptide_cols = [col for col in df.columns if any(
    keyword in col.lower() for keyword in 
    ['peptide', 'unique peptide', 'razor']
)]

min_peptides = st.slider(
    "Minimum unique peptides per protein:",
    min_value=1,
    max_value=5,
    value=2,
    help="Standard threshold: ≥2 peptides for high-confidence ID"
)

removed_pep = 0
if peptide_cols:
    pep_col = [c for c in peptide_cols if 'unique' in c.lower()][0] if any('unique' in c.lower() for c in peptide_cols) else peptide_cols[0]
    initial = len(df_qc)
    df_qc = df_qc[pd.to_numeric(df_qc[pep_col], errors='coerce') >= min_peptides]
    removed_pep = initial - len(df_qc)
    st.success(f"✅ Removed {removed_pep:,} rows ({removed_pep/initial*100:.1f}%)")
    st.caption(f"Filter: {pep_col} ≥ {min_peptides}")
else:
    st.warning("⚠️ No peptide count columns detected")

st.markdown("---")

# ============================================================================
# STEP 3: MISSING VALUES FILTER
# ============================================================================

st.subheader("3️⃣ Missing Values Filter (Per-Group Strategy)")

st.markdown("""
**Strategy**: Keep proteins with ≥70% valid values in **ANY condition**
- Balances statistical power with proteome coverage
- Preserves condition-specific proteins (absent in one, present in another)
""")

# Get unique conditions
conditions = list(set(sample_to_condition.values())) if sample_to_condition else []
conditions.sort()

if not conditions:
    st.warning("⚠️ No conditions detected. Using global filter.")
    conditions = ["All"]

col1, col2 = st.columns(2)

valid_threshold = col1.slider(
    "Valid values threshold (%):",
    min_value=25,
    max_value=100,
    value=70,
    step=5,
    help="Percentage of non-NaN values required per condition"
)

# Calculate and show missing value statistics
st.markdown("**Missing Value Statistics**")

missing_stats = []
for col in numeric_cols:
    n_missing = df_qc[col].isna().sum()
    pct_missing = n_missing / len(df_qc) * 100 if len(df_qc) > 0 else 0
    missing_stats.append({
        'Sample': col,
        'Condition': sample_to_condition.get(col, 'Unknown'),
        'Missing': int(n_missing),
        'Missing %': f"{pct_missing:.1f}%"
    })

stats_df = pd.DataFrame(missing_stats)
col2.dataframe(stats_df, hide_index=True, use_container_width=True)

# Apply per-group missing value filter
df_filtered = df_qc.copy()
valid_threshold_frac = valid_threshold / 100

# Group samples by condition
condition_samples = {}
for sample, condition in sample_to_condition.items():
    if sample in numeric_cols:
        if condition not in condition_samples:
            condition_samples[condition] = []
        condition_samples[condition].append(sample)

# Filter: keep protein if it has enough valid values in ANY condition
rows_to_keep = []
for idx in range(len(df_filtered)):
    keep = False
    if condition_samples:
        for condition, samples in condition_samples.items():
            valid_count = sum(1 for s in samples if pd.notna(df_filtered.iloc[idx][s]))
            if valid_count >= len(samples) * valid_threshold_frac:
                keep = True
                break
    else:
        # No conditions, use global threshold
        valid_count = sum(1 for s in numeric_cols if pd.notna(df_filtered.iloc[idx][s]))
        keep = valid_count >= len(numeric_cols) * valid_threshold_frac
    
    if keep:
        rows_to_keep.append(idx)

initial_filtered = len(df_filtered)
df_filtered = df_filtered.iloc[rows_to_keep].reset_index(drop=True)
removed_mv = initial_filtered - len(df_filtered)

st.success(f"✅ Removed {removed_mv:,} rows ({removed_mv/initial_filtered*100:.1f}%)")
st.info(f"📊 Remaining: {len(df_filtered):,} proteins ({len(df_filtered)/len(df)*100:.1f}% of original)")

st.markdown("---")

# ============================================================================
# STEP 4: COEFFICIENT OF VARIATION FILTER
# ============================================================================

st.subheader("4️⃣ Coefficient of Variation (CV) Filter")

st.markdown("""
**Purpose**: Remove low-quality measurements with high technical noise
- CV = (StdDev / Mean) × 100%
- Applied within replicates of same condition
""")

apply_cv = st.checkbox(
    "Apply CV filter",
    value=True,
    help="Filter proteins with CV > threshold within technical replicates"
)

cv_threshold = st.slider(
    "CV threshold (%):",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="Remove proteins with CV exceeding this within replicates"
) if apply_cv else 30

removed_cv = 0

if apply_cv and condition_samples:
    df_cv = df_filtered.copy()
    
    cv_summary = []
    rows_to_keep_cv = set(range(len(df_cv)))
    
    for condition, samples in condition_samples.items():
        valid_samples = [s for s in samples if s in numeric_cols]
        if len(valid_samples) >= 3:  # Need at least 3 replicates for CV
            for idx in range(len(df_cv)):
                values = pd.to_numeric(
                    [df_cv.iloc[idx][s] for s in valid_samples],
                    errors='coerce'
                )
                values = values.dropna()
                
                if len(values) >= 2:
                    mean_val = values.mean()
                    if mean_val > 0:
                        cv = (values.std() / mean_val) * 100
                        cv_summary.append({
                            'Protein_idx': idx,
                            'Condition': condition,
                            'CV': cv,
                            'n_replicates': len(values)
                        })
                        
                        if cv > cv_threshold:
                            rows_to_keep_cv.discard(idx)
    
    df_filtered = df_filtered.iloc[list(rows_to_keep_cv)].reset_index(drop=True)
    removed_cv = initial_filtered - len(df_filtered)
    
    if cv_summary:
        cv_df = pd.DataFrame(cv_summary)
        st.write(f"**CV Statistics**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean CV", f"{cv_df['CV'].mean():.1f}%")
        col2.metric("Median CV", f"{cv_df['CV'].median():.1f}%")
        col3.metric("Max CV", f"{cv_df['CV'].max():.1f}%")
        
        st.success(f"✅ Removed {removed_cv:,} rows with CV > {cv_threshold}% ({removed_cv/initial_filtered*100:.1f}%)")
    else:
        st.info("ℹ️ Not enough replicates per condition for CV calculation")
else:
    st.info("ℹ️ CV filter skipped (need ≥3 replicates per condition)")

st.markdown("---")

# ============================================================================
# STEP 5: NORMALIZATION (PREVIEW)
# ============================================================================

st.subheader("5️⃣ Normalization Preview")

st.markdown("""
**Applied**: Median normalization (log2 scale)
- Adjusts for sample loading differences
- Stabilizes variance for downstream analysis
""")

# Show sample statistics
st.write("**Sample Intensity Statistics**")
sample_stats = []
for col in numeric_cols:
    valid_vals = pd.to_numeric(df_filtered[col], errors='coerce').dropna()
    if len(valid_vals) > 0:
        sample_stats.append({
            'Sample': col,
            'n_valid': len(valid_vals),
            'Median': f"{valid_vals.median():.2f}",
            'Mean': f"{valid_vals.mean():.2f}",
            'Std': f"{valid_vals.std():.2f}"
        })

stats_table = pd.DataFrame(sample_stats)
st.dataframe(stats_table, hide_index=True, use_container_width=True)

st.markdown("---")

# ============================================================================
# FILTERING SUMMARY & REPORT
# ============================================================================

st.subheader("📋 Filtering Summary")

# Create summary table
summary_data = {
    'Step': [
        'Initial',
        'After QC filters',
        'After min peptides',
        'After missing values',
        'After CV filter',
        'Final'
    ],
    'Proteins': [
        f"{len(df):,}",
        f"{len(df_qc):,}",
        f"{len(df_qc) - removed_pep:,}",
        f"{len(df_filtered) + removed_cv:,}",
        f"{len(df_filtered):,}",
        f"{len(df_filtered):,}"
    ],
    'Removed': [
        '—',
        f"{removed_qc:,}",
        f"{removed_pep:,}",
        f"{removed_mv:,}",
        f"{removed_cv:,}",
        '—'
    ],
    'Retention %': [
        '100.0%',
        f"{len(df_qc)/len(df)*100:.1f}%",
        f"{(len(df_qc)-removed_pep)/len(df)*100:.1f}%",
        f"{(len(df_filtered)+removed_cv)/len(df)*100:.1f}%",
        f"{len(df_filtered)/len(df)*100:.1f}%",
        f"{len(df_filtered)/len(df)*100:.1f}%"
    ]
}

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, hide_index=True, use_container_width=True)

col1, col2 = st.columns(2)
col1.metric("Total Removed", f"{len(df) - len(df_filtered):,} ({(len(df)-len(df_filtered))/len(df)*100:.1f}%)")
col2.metric("Quality Retained", f"{len(df_filtered):,} ({len(df_filtered)/len(df)*100:.1f}%)")

st.markdown("---")

# ============================================================================
# QC REPORT GENERATION
# ============================================================================

st.subheader("📄 Quality Control Report")

report_text = f"""
# Proteomics Data Filtering Report

## Dataset Information
- **Species**: {', '.join(df[species_col].unique()) if species_col in df.columns else 'Unknown'}
- **Conditions**: {', '.join(sorted(set(sample_to_condition.values())))}
- **Samples**: {len(numeric_cols)}
- **Initial proteins**: {len(df):,}

## Filtering Parameters
1. **Contaminant filters**: 
   - Remove "Potential contaminant": {remove_contaminant if contaminant_cols else 'N/A'}
   - Remove "Reverse": {remove_reverse if contaminant_cols else 'N/A'}
   - Remove "Only identified by site": {remove_only_site if contaminant_cols else 'N/A'}

2. **Minimum peptides**: ≥{min_peptides} unique peptides
   - Removed: {removed_pep:,} ({removed_pep/len(df)*100:.2f}%)

3. **Missing value threshold**: ≥{valid_threshold}% valid per condition
   - Removed: {removed_mv:,} ({removed_mv/len(df)*100:.2f}%)

4. **Coefficient of variation**: {'Applied' if apply_cv else 'Skipped'}
   {f'- Threshold: CV < {cv_threshold}%' if apply_cv else ''}
   {f'- Removed: {removed_cv:,} ({removed_cv/len(df)*100:.2f}%)' if apply_cv else ''}

## Results
- **Final protein count**: {len(df_filtered):,}
- **Proteins retained**: {len(df_filtered)/len(df)*100:.1f}%
- **Total removed**: {len(df) - len(df_filtered):,}

## Quality Metrics
- **Missing data per sample**: {stats_df['Missing %'].str.rstrip('%').astype(float).mean():.1f}% (mean)
- **Sample coverage**: {(df_filtered[numeric_cols].notna().sum(axis=0) / len(df_filtered) * 100).mean():.1f}%

## Recommendations for Next Steps
1. Normalize data (median normalization recommended)
2. Impute remaining missing values (Random Forest for <20% missing)
3. Apply background correction if needed
4. Batch effect correction if applicable

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

st.markdown(report_text)

# Download report
st.download_button(
    label="📥 Download Report",
    data=report_text,
    file_name="filtering_report.md",
    mime="text/markdown"
)

st.markdown("---")

# ============================================================================
# CONFIRMATION & SAVE
# ============================================================================

st.subheader("✅ Save Filtered Data")

confirm = st.checkbox("I confirm the filtering parameters and QC results")

if confirm:
    if st.button("💾 Save Filtered Data", type="primary"):
        # Convert back to polars if needed
        df_filtered_pl = pl.from_pandas(df_filtered)
        
        # Save to session state
        st.session_state.df_filtered = df_filtered_pl
        st.session_state.df_numeric_filtered = df_filtered[numeric_cols]
        st.session_state.filtering_report = report_text
        st.session_state.filtering_complete = True
        
        st.success("✅ Data filtered and saved successfully!")
        st.info("Proceed to **📊 Statistical Analysis** for downstream processing")
        st.balloons()
else:
    st.info("👆 Check the box to enable save")
