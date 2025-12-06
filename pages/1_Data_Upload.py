# ============================================================================
# SECTION: DATA CLEANING (NEW)
# Replace missing/invalid values and filter by quality metrics
# ============================================================================

st.header("5️⃣ Data Cleaning (Optional)")

with st.expander("🧹 Clean Data", expanded=False):
    st.markdown("""
    **Data Cleaning Strategy:**
    1. Replace NaN and zero values with **1.0** (placeholder for missing/invalid)
    2. Filter proteins by missing rate (count intensities == 1.0)
    3. Filter by coefficient of variation (CV)
    
    **Why 1.0?** Using 1.0 as a placeholder allows easy counting:
    - `sum(intensities == 1.0)` = number of missing values
    - Distinguishes missing from actual low intensities
    """)
    
    # Track original data
    n_proteins_original = len(df_raw)
    n_samples = len(selected_numeric)
    
    # --- Step 1: Replace NaN and 0 with 1.0 ---
    st.subheader("1. Replace Missing & Zero Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        replace_nan = st.checkbox(
            "Replace NaN with 1.0",
            value=True,
            help="Convert all missing values to 1.0"
        )
    
    with col2:
        replace_zero = st.checkbox(
            "Replace 0 with 1.0",
            value=True,
            help="Convert zero intensities to 1.0 (often invalid)"
        )
    
    # Preview impact
    if replace_nan or replace_zero:
        preview_stats = {
            "Metric": [],
            "Count": [],
            "Percentage": []
        }
        
        if replace_nan:
            n_nan = df_raw[selected_numeric].isna().sum().sum()
            preview_stats["Metric"].append("NaN values")
            preview_stats["Count"].append(n_nan)
            preview_stats["Percentage"].append(f"{n_nan / (len(df_raw) * n_samples) * 100:.2f}%")
        
        if replace_zero:
            n_zero = (df_raw[selected_numeric] == 0).sum().sum()
            preview_stats["Metric"].append("Zero values")
            preview_stats["Count"].append(n_zero)
            preview_stats["Percentage"].append(f"{n_zero / (len(df_raw) * n_samples) * 100:.2f}%")
        
        st.dataframe(
            pd.DataFrame(preview_stats),
            hide_index=True,
            use_container_width=True
        )
    
    if st.button("🔄 Apply Replacement", key="replace_btn"):
        df_cleaned = df_raw.copy()
        
        total_replaced = 0
        
        # Replace NaN with 1.0
        if replace_nan:
            n_nan_before = df_cleaned[selected_numeric].isna().sum().sum()
            df_cleaned[selected_numeric] = df_cleaned[selected_numeric].fillna(1.0)
            total_replaced += n_nan_before
            st.info(f"✅ Replaced {n_nan_before} NaN values with 1.0")
        
        # Replace 0 with 1.0
        if replace_zero:
            mask_zero = df_cleaned[selected_numeric] == 0
            n_zero = mask_zero.sum().sum()
            df_cleaned[selected_numeric] = df_cleaned[selected_numeric].mask(mask_zero, 1.0)
            total_replaced += n_zero
            st.info(f"✅ Replaced {n_zero} zero values with 1.0")
        
        df_raw = df_cleaned
        st.success(f"✅ Total replaced: {total_replaced} values ({total_replaced / (len(df_raw) * n_samples) * 100:.2f}%)")
        st.rerun()
    
    st.markdown("---")
    
    # --- Step 2: Filter by missing rate (count of 1.0 values) ---
    st.subheader("2. Filter by Missing Data Rate")
    
    st.markdown("""
    Remove proteins with too many missing/invalid values (intensities == 1.0).
    """)
    
    max_missing_rate = st.slider(
        "Maximum missing rate per protein:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format="%.0f%%",
        help="Proteins exceeding this threshold will be removed"
    )
    
    # Calculate missing rate by counting 1.0 values
    count_ones = (df_raw[selected_numeric] == 1.0).sum(axis=1)
    missing_rate_per_protein = count_ones / n_samples
    n_would_remove = (missing_rate_per_protein > max_missing_rate).sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Proteins to remove",
            n_would_remove,
            delta=f"-{n_would_remove/len(df_raw)*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Proteins remaining",
            len(df_raw) - n_would_remove,
            delta=f"{(len(df_raw) - n_would_remove)/len(df_raw)*100:.1f}%"
        )
    
    # Show distribution of missing rates
    st.markdown("**Missing Rate Distribution**")
    missing_dist = pd.DataFrame({
        "Missing Rate": ["0-25%", "25-50%", "50-75%", "75-100%"],
        "Count": [
            ((missing_rate_per_protein >= 0) & (missing_rate_per_protein < 0.25)).sum(),
            ((missing_rate_per_protein >= 0.25) & (missing_rate_per_protein < 0.5)).sum(),
            ((missing_rate_per_protein >= 0.5) & (missing_rate_per_protein < 0.75)).sum(),
            (missing_rate_per_protein >= 0.75).sum(),
        ]
    })
    st.dataframe(missing_dist, hide_index=True, use_container_width=True)
    
    if st.button("🗑️ Apply Missing Rate Filter", key="missing_filter_btn"):
        # Keep proteins below threshold
        df_cleaned = df_raw[missing_rate_per_protein <= max_missing_rate].copy()
        
        n_removed = len(df_raw) - len(df_cleaned)
        st.success(f"✅ Removed {n_removed} proteins ({n_removed/len(df_raw)*100:.1f}%)")
        
        # Update dataframe
        df_raw = df_cleaned
        st.rerun()
    
    st.markdown("---")
    
   # --- Step 3: Filter by Coefficient of Variation ---
st.subheader("3. Filter by Coefficient of Variation")

st.markdown("""
Remove proteins with high variability across samples.
**CV = std / mean** (excludes intensities == 1.0 from calculation)
""")

# Detect conditions from column names
from helpers.analysis import detect_conditions_from_columns, group_columns_by_condition

conditions = detect_conditions_from_columns(selected_numeric)

# Calculate CVs
cv_data = {
    'Protein': [],
    'CV_Overall': [],
}

# Add per-condition CV columns
for cond in conditions:
    cv_data[f'CV_{cond}'] = []

for protein_id, row in df_raw.iterrows():
    cv_data['Protein'].append(protein_id)
    
    # --- Overall CV ---
    vals_overall = row[selected_numeric]
    valid_vals_overall = vals_overall[vals_overall != 1.0]
    
    if len(valid_vals_overall) >= 2:
        mean_val = valid_vals_overall.mean()
        std_val = valid_vals_overall.std()
        cv_overall = std_val / mean_val if mean_val > 0 else np.nan
    else:
        cv_overall = np.nan
    
    cv_data['CV_Overall'].append(cv_overall)
    
    # --- Per-condition CV ---
    for cond in conditions:
        cond_cols = group_columns_by_condition(selected_numeric, cond)
        
        if len(cond_cols) >= 2:
            vals_cond = row[cond_cols]
            valid_vals_cond = vals_cond[vals_cond != 1.0]
            
            if len(valid_vals_cond) >= 2:
                mean_cond = valid_vals_cond.mean()
                std_cond = valid_vals_cond.std()
                cv_cond = std_cond / mean_cond if mean_cond > 0 else np.nan
            else:
                cv_cond = np.nan
        else:
            cv_cond = np.nan
        
        cv_data[f'CV_{cond}'].append(cv_cond)

# Create CV DataFrame
cv_df = pd.DataFrame(cv_data).set_index('Protein')

# --- Filter Options ---
st.markdown("**Filter Strategy**")

filter_strategy = st.radio(
    "Choose CV filter strategy:",
    options=[
        "Overall CV only",
        "Per-condition CV (all must pass)",
        "Per-condition CV (any must pass)",
        "Custom thresholds per condition"
    ],
    help="""
    - Overall: Filter by CV across all samples
    - All must pass: Protein must have low CV in ALL conditions
    - Any must pass: Protein must have low CV in AT LEAST ONE condition
    - Custom: Set different thresholds for each condition
    """
)

# --- Threshold Selection ---
if filter_strategy == "Overall CV only":
    max_cv_overall = st.slider(
        "Maximum overall CV:",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="CV > 1.0 means std > mean (high variability)"
    )
    
    # Apply filter
    mask_pass = cv_df['CV_Overall'] <= max_cv_overall
    n_would_remove = (~mask_pass).sum()

elif filter_strategy == "Per-condition CV (all must pass)":
    max_cv_condition = st.slider(
        "Maximum CV per condition:",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="All conditions must be below this threshold"
    )
    
    # Apply filter: all conditions must pass
    mask_pass = True
    for cond in conditions:
        mask_pass = mask_pass & (cv_df[f'CV_{cond}'] <= max_cv_condition)
    
    n_would_remove = (~mask_pass).sum()

elif filter_strategy == "Per-condition CV (any must pass)":
    max_cv_condition = st.slider(
        "Maximum CV per condition:",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="At least one condition must be below this threshold"
    )
    
    # Apply filter: any condition can pass
    mask_pass = False
    for cond in conditions:
        mask_pass = mask_pass | (cv_df[f'CV_{cond}'] <= max_cv_condition)
    
    n_would_remove = (~mask_pass).sum()

else:  # Custom thresholds
    st.markdown("**Set Custom Thresholds**")
    
    custom_thresholds = {}
    cols = st.columns(len(conditions))
    
    for idx, cond in enumerate(conditions):
        with cols[idx]:
            custom_thresholds[cond] = st.number_input(
                f"Max CV for {cond}:",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key=f"cv_threshold_{cond}"
            )
    
    # Apply custom filters (all must pass)
    mask_pass = True
    for cond, threshold in custom_thresholds.items():
        mask_pass = mask_pass & (cv_df[f'CV_{cond}'] <= threshold)
    
    n_would_remove = (~mask_pass).sum()

# --- Preview Impact ---
st.markdown("**Filter Impact Preview**")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Proteins to remove",
        int(n_would_remove),
        delta=f"-{n_would_remove/len(cv_df)*100:.1f}%"
    )

with col2:
    if len(cv_df['CV_Overall'].dropna()) > 0:
        st.metric(
            "Median Overall CV",
            f"{cv_df['CV_Overall'].median():.2f}",
            help="Lower is better (less variable)"
        )

# --- CV Statistics Table ---
st.markdown("**CV Statistics by Condition**")

cv_stats = pd.DataFrame({
    'Condition': ['Overall'] + conditions,
    'Mean CV': [cv_df['CV_Overall'].mean()] + [cv_df[f'CV_{cond}'].mean() for cond in conditions],
    'Median CV': [cv_df['CV_Overall'].median()] + [cv_df[f'CV_{cond}'].median() for cond in conditions],
    'Std CV': [cv_df['CV_Overall'].std()] + [cv_df[f'CV_{cond}'].std() for cond in conditions],
    'Min CV': [cv_df['CV_Overall'].min()] + [cv_df[f'CV_{cond}'].min() for cond in conditions],
    'Max CV': [cv_df['CV_Overall'].max()] + [cv_df[f'CV_{cond}'].max() for cond in conditions],
})

st.dataframe(
    cv_stats.style.format({
        'Mean CV': '{:.3f}',
        'Median CV': '{:.3f}',
        'Std CV': '{:.3f}',
        'Min CV': '{:.3f}',
        'Max CV': '{:.3f}',
    }),
    hide_index=True,
    use_container_width=True
)

# --- CV Distribution Visualization ---
with st.expander("📊 View CV Distributions", expanded=False):
    
    # Create distribution table
    st.markdown("**Overall CV Distribution**")
    cv_overall_clean = cv_df['CV_Overall'].dropna()
    
    cv_dist_overall = pd.DataFrame({
        "CV Range": ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"],
        "Count": [
            ((cv_overall_clean >= 0) & (cv_overall_clean < 0.25)).sum(),
            ((cv_overall_clean >= 0.25) & (cv_overall_clean < 0.5)).sum(),
            ((cv_overall_clean >= 0.5) & (cv_overall_clean < 0.75)).sum(),
            ((cv_overall_clean >= 0.75) & (cv_overall_clean < 1.0)).sum(),
            ((cv_overall_clean >= 1.0) & (cv_overall_clean < 1.5)).sum(),
            (cv_overall_clean >= 1.5).sum(),
        ]
    })
    cv_dist_overall['Percentage'] = (cv_dist_overall['Count'] / len(cv_overall_clean) * 100).round(1)
    
    st.dataframe(cv_dist_overall, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Per-condition distributions
    for cond in conditions:
        st.markdown(f"**Condition {cond} - CV Distribution**")
        cv_cond_clean = cv_df[f'CV_{cond}'].dropna()
        
        if len(cv_cond_clean) > 0:
            cv_dist_cond = pd.DataFrame({
                "CV Range": ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"],
                "Count": [
                    ((cv_cond_clean >= 0) & (cv_cond_clean < 0.25)).sum(),
                    ((cv_cond_clean >= 0.25) & (cv_cond_clean < 0.5)).sum(),
                    ((cv_cond_clean >= 0.5) & (cv_cond_clean < 0.75)).sum(),
                    ((cv_cond_clean >= 0.75) & (cv_cond_clean < 1.0)).sum(),
                    ((cv_cond_clean >= 1.0) & (cv_cond_clean < 1.5)).sum(),
                    (cv_cond_clean >= 1.5).sum(),
                ]
            })
            cv_dist_cond['Percentage'] = (cv_dist_cond['Count'] / len(cv_cond_clean) * 100).round(1)
            
            st.dataframe(cv_dist_cond, hide_index=True, use_container_width=True)
        else:
            st.warning(f"No valid CV data for condition {cond}")

# --- Top Variable Proteins ---
with st.expander("🔍 View Most Variable Proteins", expanded=False):
    st.markdown("**Top 20 Proteins by Overall CV**")
    
    top_variable = cv_df.nlargest(20, 'CV_Overall')[['CV_Overall'] + [f'CV_{cond}' for cond in conditions]]
    
    st.dataframe(
        top_variable.style.format({col: '{:.3f}' for col in top_variable.columns}),
        use_container_width=True
    )

# --- Apply CV Filter ---
if st.button("🗑️ Apply CV Filter", key="cv_filter_btn"):
    # Get proteins that pass the filter
    keep_proteins = cv_df[mask_pass].index
    df_cleaned = df_raw.loc[keep_proteins].copy()
    
    n_removed = len(df_raw) - len(df_cleaned)
    st.success(f"✅ Removed {n_removed} proteins ({n_removed/len(df_raw)*100:.1f}%)")
    
    # Show per-condition removal stats
    st.markdown("**Removal Details by Condition:**")
    removal_details = []
    
    for cond in conditions:
        cv_cond = cv_df[f'CV_{cond}']
        if filter_strategy == "Custom thresholds":
            threshold = custom_thresholds[cond]
        elif filter_strategy == "Overall CV only":
            threshold = max_cv_overall
        else:
            threshold = max_cv_condition
        
        n_fail = (cv_cond > threshold).sum()
        removal_details.append({
            'Condition': cond,
            'Threshold': f'{threshold:.2f}',
            'Failed': int(n_fail),
            'Pass Rate': f'{(1 - n_fail/len(cv_cond))*100:.1f}%'
        })
    
    st.dataframe(
        pd.DataFrame(removal_details),
        hide_index=True,
        use_container_width=True
    )
    
    # Update dataframe
    df_raw = df_cleaned
    st.rerun()

# --- Export CV Data ---
st.markdown("**Export CV Data**")
download_button_csv(
    cv_df,
    filename="cv_analysis.csv",
    label="📥 Download CV Data"
)

    
    # Preview impact
    n_would_remove_cv = (cv_series_clean > max_cv).sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Proteins to remove",
            n_would_remove_cv,
            delta=f"-{n_would_remove_cv/len(cv_series_clean)*100:.1f}%"
        )
    
    with col2:
        if len(cv_series_clean) > 0:
            st.metric(
                "Median CV",
                f"{cv_series_clean.median():.2f}",
                help="Lower is better (less variable)"
            )
    
    # Show CV distribution
    st.markdown("**CV Distribution**")
    cv_dist = pd.DataFrame({
        "CV Range": ["0-0.5", "0.5-1.0", "1.0-1.5", ">1.5"],
        "Count": [
            ((cv_series_clean >= 0) & (cv_series_clean < 0.5)).sum(),
            ((cv_series_clean >= 0.5) & (cv_series_clean < 1.0)).sum(),
            ((cv_series_clean >= 1.0) & (cv_series_clean < 1.5)).sum(),
            (cv_series_clean >= 1.5).sum(),
        ]
    })
    st.dataframe(cv_dist, hide_index=True, use_container_width=True)
    
    if st.button("🗑️ Apply CV Filter", key="cv_filter_btn"):
        # Keep proteins below threshold
        keep_indices = cv_series_clean[cv_series_clean <= max_cv].index
        df_cleaned = df_raw.loc[keep_indices].copy()
        
        n_removed = len(df_raw) - len(df_cleaned)
        st.success(f"✅ Removed {n_removed} proteins ({n_removed/len(df_raw)*100:.1f}%)")
        
        # Update dataframe
        df_raw = df_cleaned
        st.rerun()
    
    st.markdown("---")
    
    # --- Summary of all cleaning operations ---
    st.subheader("📊 Cleaning Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Original Proteins",
            n_proteins_original
        )
    
    with col2:
        st.metric(
            "Current Proteins",
            len(df_raw),
            delta=f"{len(df_raw) - n_proteins_original}"
        )
    
    with col3:
        retention_rate = len(df_raw) / n_proteins_original * 100
        st.metric(
            "Retention Rate",
            f"{retention_rate:.1f}%"
        )
    
    # Count current missing values (1.0)
    current_missing = (df_raw[selected_numeric] == 1.0).sum().sum()
    total_values = len(df_raw) * n_samples
    
    st.info(f"""
    **Current Data Quality:**
    - Missing/Invalid values (==1.0): {current_missing} ({current_missing/total_values*100:.2f}%)
    - Valid measurements: {total_values - current_missing} ({(total_values - current_missing)/total_values*100:.2f}%)
    """)
