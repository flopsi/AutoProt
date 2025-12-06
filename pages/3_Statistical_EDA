"""
pages/3_Statistical_EDA.py

Statistical exploratory data analysis
Differential expression testing with t-test/ANOVA, volcano plots, and results export
"""

import streamlit as st
import pandas as pd
import numpy as np
from helpers.stats import (
    perform_ttest, perform_anova, classify_regulation,
    calculate_error_metrics, compute_species_rmse
)
from helpers.viz import create_volcano_plot, create_qc_dashboard
from helpers.core import get_theme
from helpers.ui import download_button_csv, show_data_summary
from helpers.analysis import (
    detect_conditions_from_columns, group_columns_by_condition,
    validate_group_comparison
)

# ============================================================================
# PAGE CONFIGURATION
# Check if transformed data is available
# ============================================================================

st.title("🧪 Statistical EDA & Differential Expression")

# Check for transformed data
if "df_transformed" not in st.session_state or st.session_state.df_transformed is None:
    st.warning("⚠️ No transformed data available. Please complete Visual EDA first.")
    if st.button("← Go to Visual EDA"):
        st.switch_page("pages/2_Visual_EDA.py")
    st.stop()

df_transformed = st.session_state.df_transformed
trans_cols = st.session_state.trans_cols
protein_data = st.session_state.protein_data
current_transform = st.session_state.get("current_transform", "log2")
theme_name = st.session_state.get("theme", "light")

st.markdown(f"""
Using: **{current_transform}** transformed data | 
{len(df_transformed)} proteins × {len(trans_cols)} samples
""")

# ============================================================================
# SECTION: GROUP DEFINITION
# Define experimental groups for comparison
# ============================================================================

st.header("1️⃣ Define Experimental Groups")

# Auto-detect conditions
conditions = detect_conditions_from_columns(trans_cols)

if len(conditions) < 2:
    st.error("❌ Need at least 2 experimental conditions for comparison")
    st.stop()

st.info(f"📊 Detected {len(conditions)} conditions: {', '.join(conditions)}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Group (Control)")
    group1_condition = st.selectbox(
        "Select reference condition:",
        options=conditions,
        index=0,
        help="Typically the control or baseline condition"
    )
    group1_cols = group_columns_by_condition(trans_cols, group1_condition)
    st.success(f"✅ {len(group1_cols)} samples: {', '.join(group1_cols)}")

with col2:
    st.subheader("Treatment Group")
    remaining_conditions = [c for c in conditions if c != group1_condition]
    group2_condition = st.selectbox(
        "Select treatment condition:",
        options=remaining_conditions,
        index=0 if remaining_conditions else 0,
        help="The experimental or treatment condition"
    )
    group2_cols = group_columns_by_condition(trans_cols, group2_condition)
    st.success(f"✅ {len(group2_cols)} samples: {', '.join(group2_cols)}")

# Validate groups
is_valid, msg = validate_group_comparison(group1_cols, group2_cols, min_samples=2)
if not is_valid:
    st.error(msg)
    st.stop()
else:
    st.success(msg)

# ============================================================================
# SECTION: STATISTICAL TEST SELECTION
# Choose between t-test and ANOVA
# ============================================================================

st.header("2️⃣ Statistical Test Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    test_method = st.selectbox(
        "Test method:",
        options=["T-test (2 groups)", "ANOVA (3+ groups)"],
        index=0,
        help="T-test for pairwise comparison, ANOVA for multiple groups"
    )

with col2:
    fc_threshold = st.number_input(
        "log2FC threshold:",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Fold change cutoff (1.0 = 2-fold change)"
    )

with col3:
    pval_threshold = st.number_input(
        "P-value threshold:",
        min_value=0.001,
        max_value=0.1,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Significance level for FDR-corrected p-values"
    )

# ============================================================================
# SECTION: RUN ANALYSIS
# Execute statistical test with caching
# ============================================================================

st.header("3️⃣ Run Differential Expression Analysis")

if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    
    with st.spinner("⏳ Running statistical tests..."):
        
        # Run t-test (cached)
        results_df = perform_ttest(
            df_transformed,
            group1_cols,
            group2_cols,
            min_valid=2
        )
        
        # Classify regulation
        results_df["regulation"] = results_df.apply(
            lambda row: classify_regulation(
                row["log2fc"],
                row["fdr"],  # Use FDR instead of p-value
                fc_threshold,
                pval_threshold
            ),
            axis=1
        )
        
        # Store in session state
        st.session_state.analysis_results = results_df
    
    st.success("✅ Analysis complete!")

# Check if results available
if "analysis_results" not in st.session_state or st.session_state.analysis_results is None:
    st.info("👆 Click 'Run Analysis' to perform differential expression testing")
    st.stop()

results_df = st.session_state.analysis_results

# ============================================================================
# SECTION: RESULTS SUMMARY
# Display counts and key metrics
# ============================================================================

st.header("4️⃣ Results Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    n_up = (results_df["regulation"] == "up").sum()
    st.metric("Upregulated", n_up, delta=f"{n_up/len(results_df)*100:.1f}%")

with col2:
    n_down = (results_df["regulation"] == "down").sum()
    st.metric("Downregulated", n_down, delta=f"{n_down/len(results_df)*100:.1f}%")

with col3:
    n_sig = n_up + n_down
    st.metric("Total Significant", n_sig, delta=f"{n_sig/len(results_df)*100:.1f}%")

with col4:
    n_tested = (results_df["regulation"] != "not_tested").sum()
    st.metric("Tested", n_tested, delta=f"{n_tested/len(results_df)*100:.1f}%")

# Regulation breakdown
st.subheader("Regulation Breakdown")
reg_counts = results_df["regulation"].value_counts()
st.dataframe(
    pd.DataFrame({
        "Category": reg_counts.index,
        "Count": reg_counts.values,
        "Percentage": (reg_counts.values / len(results_df) * 100).round(1)
    }),
    hide_index=True,
    use_container_width=True
)

# ============================================================================
# SECTION: VOLCANO PLOT
# Interactive visualization of differential expression
# ============================================================================

st.header("5️⃣ Volcano Plot")

col1, col2 = st.columns([3, 1])

with col2:
    show_labels = st.checkbox("Show protein labels", value=False)
    top_n = st.slider("Top N proteins to label:", 5, 50, 10) if show_labels else 0

with col1:
    with st.spinner("Generating volcano plot..."):
        fig_volcano = create_volcano_plot(
            results_df["log2fc"],
            results_df["neg_log10_pval"],
            results_df["regulation"],
            fc_threshold=fc_threshold,
            pval_threshold=pval_threshold,
            theme_name=theme_name
        )
        st.plotly_chart(fig_volcano, use_container_width=True)

# ============================================================================
# SECTION: RESULTS TABLE
# Interactive table with filtering and sorting
# ============================================================================

st.header("6️⃣ Detailed Results")

# Filter options
col1, col2, col3 = st.columns(3)

with col1:
    filter_regulation = st.multiselect(
        "Filter by regulation:",
        options=["up", "down", "not_significant", "not_tested"],
        default=["up", "down"],
        help="Show only selected categories"
    )

with col2:
    sort_by = st.selectbox(
        "Sort by:",
        options=["fdr", "pvalue", "log2fc", "mean_g1", "mean_g2"],
        index=0
    )

with col3:
    sort_order = st.radio("Order:", ["Ascending", "Descending"], index=0)

# Apply filters
results_filtered = results_df[results_df["regulation"].isin(filter_regulation)].copy()

# Sort
ascending = (sort_order == "Ascending")
results_sorted = results_filtered.sort_values(sort_by, ascending=ascending)

# Display
st.dataframe(
    results_sorted[[
        "log2fc", "pvalue", "fdr", "neg_log10_pval",
        "mean_g1", "mean_g2", "n_g1", "n_g2", "regulation"
    ]].head(100),
    use_container_width=True,
    height=400,
    column_config={
        "log2fc": st.column_config.NumberColumn(format="%.3f"),
        "pvalue": st.column_config.NumberColumn(format="%.4e"),
        "fdr": st.column_config.NumberColumn(format="%.4e"),
        "neg_log10_pval": st.column_config.NumberColumn(format="%.2f"),
    }
)

st.caption(f"Showing top 100 of {len(results_sorted)} proteins")

# ============================================================================
# SECTION: SPECIES-SPECIFIC ANALYSIS
# If species data available, show per-species metrics
# ============================================================================

if protein_data.species_col is not None:
    st.header("7️⃣ Species-Specific Analysis")
    
    # This would require theoretical FC input - placeholder for now
    st.info("ℹ️ Species-specific RMSE calculation requires theoretical fold changes (spike-in validation)")

# ============================================================================
# SECTION: EXPORT OPTIONS
# Download results in multiple formats
# ============================================================================

st.header("8️⃣ Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    download_button_csv(
        results_df,
        filename=f"results_{group1_condition}_vs_{group2_condition}.csv",
        label="📥 Download All Results"
    )

with col2:
    # Export significant only
    results_sig = results_df[results_df["regulation"].isin(["up", "down"])]
    download_button_csv(
        results_sig,
        filename=f"significant_{group1_condition}_vs_{group2_condition}.csv",
        label="📥 Download Significant Only"
    )

with col3:
    # Export upregulated only
    results_up = results_df[results_df["regulation"] == "up"]
    download_button_csv(
        results_up,
        filename=f"upregulated_{group1_condition}_vs_{group2_condition}.csv",
        label="📥 Download Upregulated"
    )

# ============================================================================
# FOOTER
# Analysis completion message
# ============================================================================

st.markdown("---")
st.success("✅ Analysis complete! Download results above or adjust thresholds and re-run.")
st.caption("**Tip:** Use the volcano plot to visually identify significantly regulated proteins.")
