"""
pages/4_Missing_Value_Imputation.py - SMART MISSING VALUE IMPUTATION
Per-condition imputation strategies for proteomics data
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
import sys

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Missing Value Imputation",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Missing Value Imputation")
st.markdown("Smart per-condition imputation strategies for proteomics data")
st.markdown("---")

# ============================================================================
# CHECK FOR FILTERED DATA
# ============================================================================

if 'df_filtered' not in st.session_state or st.session_state.df_filtered is None:
    st.error("❌ No filtered data. Please complete **🧹 Data Filtering** first")
    st.stop()

# Load data
df = st.session_state.df_filtered.to_pandas() if isinstance(st.session_state.df_filtered, pl.DataFrame) else st.session_state.df_filtered.copy()
numeric_cols = st.session_state.numeric_cols
sample_to_condition = st.session_state.get('sample_to_condition', {})
species_col = st.session_state.species_col

st.info(f"📊 **Input**: {len(df):,} proteins × {len(numeric_cols)} samples")

# ============================================================================
# MISSING VALUE ANALYSIS
# ============================================================================

st.subheader("📈 Missing Value Analysis")

# Calculate missing value statistics
missing_stats = []
for col in numeric_cols:
    n_missing = df[col].isna().sum()
    pct_missing = n_missing / len(df) * 100
    condition = sample_to_condition.get(col, 'Unknown')
    missing_stats.append({
        'Sample': col,
        'Condition': condition,
        'Missing': int(n_missing),
        'Missing %': f"{pct_missing:.1f}%"
    })

stats_df = pd.DataFrame(missing_stats)

col1, col2, col3 = st.columns(3)
col1.metric("Total Missing Values", f"{df[numeric_cols].isna().sum().sum():,}")
col2.metric("Average Missing %", f"{df[numeric_cols].isna().sum().sum() / (len(df) * len(numeric_cols)) * 100:.1f}%")
col3.metric("Samples with Data", f"{len(numeric_cols)}")

st.dataframe(stats_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ============================================================================
# IMPUTATION STRATEGY SELECTION
# ============================================================================

st.subheader("⚙️ Imputation Strategy")

strategy = st.selectbox(
    "Select imputation method:",
    options=[
        "min_prob",
        "knn",
        "random_forest",
        "condition_mean",
        "global_mean"
    ],
    format_func=lambda x: {
        "min_prob": "MinProb - Minimum from dataset (Conservative, for MNAR data)",
        "knn": "k-Nearest Neighbors (Good for mixed MCAR/MNAR, preserves structure)",
        "random_forest": "Random Forest (Best accuracy, handles complex patterns)",
        "condition_mean": "Condition Mean (Simple, per-group approach)",
        "global_mean": "Global Mean (Simplest, baseline)"
    }[x],
    index=2
)

st.markdown(f"""
**Selected: {strategy.upper()}**

**Strategy Details:**
""")

strategy_info = {
    "min_prob": """
- Uses minimum value from dataset per feature
- Conservative approach for left-censored (MNAR) data
- Appropriate for proteomics where missing = below detection limit
- Assumes missing values are biologically small
""",
    "knn": """
- Imputes using k-nearest neighbor samples
- Good for MCAR data with moderate missingness
- Preserves local patterns and correlations
- Best with k=3-5 neighbors
""",
    "random_forest": """
- Uses iterative Random Forest regression
- Best for complex, non-linear relationships
- Handles mixed MCAR/MNAR scenarios
- More computationally intensive but most accurate
""",
    "condition_mean": """
- Imputes missing values with condition group mean
- Simple, biologically-aware approach
- Good when protein is present in other samples of same condition
- Per-condition imputation
""",
    "global_mean": """
- Imputes with global mean across all samples
- Baseline/simplest method
- Not recommended for proteomics with group structure
- Use only if other methods fail
"""
}

st.markdown(strategy_info.get(strategy, ""))

# Get unique conditions for per-condition imputation
conditions = sorted(list(set(sample_to_condition.values())))
condition_samples = {}
for sample, condition in sample_to_condition.items():
    if sample in numeric_cols:
        if condition not in condition_samples:
            condition_samples[condition] = []
        condition_samples[condition].append(sample)

st.markdown("---")

# ============================================================================
# PARAMETER SELECTION
# ============================================================================

st.subheader("🎯 Imputation Parameters")

if strategy == "knn":
    n_neighbors = st.slider(
        "Number of neighbors (k):",
        min_value=2,
        max_value=10,
        value=5,
        help="Typical range: 3-5 for proteomics data"
    )
elif strategy == "random_forest":
    n_estimators = st.slider(
        "Number of trees:",
        min_value=10,
        max_value=200,
        value=100,
        help="More trees = better quality but slower"
    )
    max_iter = st.slider(
        "Maximum iterations:",
        min_value=1,
        max_value=10,
        value=3,
        help="Iterative refinement rounds (typically 2-4)"
    )
elif strategy == "min_prob":
    percentile = st.slider(
        "Percentile for MinProb:",
        min_value=1,
        max_value=10,
        value=1,
        help="Use this percentile of minimum values (1-10%)"
    )

st.markdown("---")

# ============================================================================
# PERFORM IMPUTATION
# ============================================================================

st.subheader("🔄 Perform Imputation")

if st.button("Run Imputation", type="primary"):
    with st.spinner("Imputing missing values..."):
        df_imputed = df.copy()
        
        if strategy == "min_prob":
            # MinProb: use minimum value per feature (percentile-based)
            for col in numeric_cols:
                valid_vals = df[col].dropna()
                if len(valid_vals) > 0:
                    min_val = np.percentile(valid_vals, percentile)
                    df_imputed[col].fillna(min_val, inplace=True)
            method_name = f"MinProb (p={percentile})"
        
        elif strategy == "knn":
            # KNN imputation
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            method_name = f"KNN (k={n_neighbors})"
        
        elif strategy == "random_forest":
            # Iterative Random Forest imputation
            df_imputed[numeric_cols] = df[numeric_cols].copy()
            
            for iteration in range(max_iter):
                for col in numeric_cols:
                    missing_mask = df_imputed[col].isna()
                    
                    if missing_mask.sum() > 0:
                        # Features = all other columns
                        other_cols = [c for c in numeric_cols if c != col]
                        X = df_imputed[other_cols].copy()
                        y = df_imputed[col].copy()
                        
                        # Train on non-missing values
                        train_mask = ~missing_mask
                        if train_mask.sum() > 0:
                            rf = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=15,
                                random_state=42,
                                n_jobs=-1
                            )
                            rf.fit(X[train_mask], y[train_mask])
                            # Predict missing values
                            df_imputed.loc[missing_mask, col] = rf.predict(X[missing_mask])
            
            method_name = f"Random Forest (n_trees={n_estimators}, iterations={max_iter})"
        
        elif strategy == "condition_mean":
            # Per-condition mean imputation
            for condition, samples in condition_samples.items():
                valid_samples = [s for s in samples if s in numeric_cols]
                
                for col in numeric_cols:
                    missing_mask = df_imputed[col].isna()
                    
                    if missing_mask.sum() > 0:
                        # Mean of other samples in same condition
                        other_samples = [s for s in valid_samples if s != col]
                        if other_samples:
                            condition_mean = df_imputed.loc[missing_mask, other_samples].mean(axis=1)
                            df_imputed.loc[missing_mask, col] = condition_mean
            
            method_name = "Condition Mean"
        
        elif strategy == "global_mean":
            # Global mean imputation
            for col in numeric_cols:
                mean_val = df[col].mean()
                df_imputed[col].fillna(mean_val, inplace=True)
            
            method_name = "Global Mean"
        
        # Check for remaining missing values
        remaining_missing = df_imputed[numeric_cols].isna().sum().sum()
        
        if remaining_missing == 0:
            st.success(f"✅ Imputation complete! All {len(df) * len(numeric_cols):,} values filled")
            st.info(f"**Method**: {method_name}")
            
            # Show before/after statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Before Imputation**")
                missing_before = df[numeric_cols].isna().sum().sum()
                st.metric("Missing Values", f"{missing_before:,}")
            
            with col2:
                st.markdown("**After Imputation**")
                st.metric("Missing Values", f"{remaining_missing:,}")
            
            # Preview of imputed data
            st.markdown("**Preview: First 5 proteins (showing all samples)**")
            preview = df_imputed[numeric_cols].head(5)
            st.dataframe(preview, use_container_width=True)
            
            # Save to session state
            st.session_state.df_imputed = df_imputed
            st.session_state.imputation_method = method_name
            st.session_state.imputation_complete = True
            
            st.markdown("---")
            
            # Summary statistics
            st.subheader("📊 Imputation Summary")
            
            summary_data = {
                'Metric': [
                    'Method',
                    'Total values processed',
                    'Values imputed',
                    'Coverage achieved'
                ],
                'Value': [
                    method_name,
                    f"{len(df) * len(numeric_cols):,}",
                    f"{df[numeric_cols].isna().sum().sum():,}",
                    f"100%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            
            st.balloons()
        
        else:
            st.warning(f"⚠️ {remaining_missing:,} values still missing after imputation")

st.markdown("---")

# ============================================================================
# NEXT STEPS
# ============================================================================

if 'imputation_complete' in st.session_state and st.session_state.imputation_complete:
    st.subheader("✅ Next Steps")
    st.markdown("""
    Your data is now ready for downstream analysis!
    
    **Recommended workflow:**
    1. ✅ Data Upload
    2. ✅ Data Filtering  
    3. ✅ Missing Value Imputation
    4. → **Statistical Analysis** (Differential expression, volcano plots, etc.)
    5. → **Visualization** (Heatmaps, PCA, etc.)
    """)
    
    if st.button("Proceed to Statistical Analysis →", type="primary"):
        st.info("Ready to analyze! Navigate to the next page.")
