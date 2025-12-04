# ========== USE FILTERED DATA ==========
# Check for filtered data from previous page
if "last_filtered_data" in st.session_state and st.session_state.last_filtered_data is not None:
    transform_data = st.session_state.last_filtered_data.copy()
    filter_params = st.session_state.last_filtered_params
    st.info(f"✅ Using filtered dataset: {len(transform_data):,} proteins")
else:
    # Fallback to full raw data
    transform_data = get_transform_data(protein_model, "log2")
    filter_params = None
    st.warning("⚠️ Using full dataset (no filtered data from Filtering page). Go to Filtering page and store data first.")

numeric_cols = protein_model.numeric_cols
condition_groups = build_condition_groups(numeric_cols)

# ========== TOP: Dataset Overview ==========
st.markdown("### 📊 Dataset Overview")

cv_data = compute_cv_per_condition(transform_data, numeric_cols)

col_o1, col_o2, col_o3, col_o4 = st.columns(4)

with col_o1:
    st.metric("Total Proteins", f"{len(transform_data):,}")

with col_o2:
    if not cv_data.empty:
        cv_mean = cv_data.to_numpy().ravel()
        cv_mean = cv_mean[~np.isnan(cv_mean)].mean()
        st.metric("Mean CV%", f"{cv_mean:.1f}")
    else:
        st.metric("Mean CV%", "N/A")

with col_o3:
    if not cv_data.empty:
        cv_median = np.median(cv_data.to_numpy().ravel()[~np.isnan(cv_data.to_numpy().ravel())])
        st.metric("Median CV%", f"{cv_median:.1f}")
    else:
        st.metric("Median CV%", "N/A")

with col_o4:
    st.metric("Conditions", len(condition_groups))

if filter_params:
    st.caption(f"Filter info: {', '.join(filter_params.get('active_filters', []))}")

st.markdown("---")

# ========== SIDEBAR: Analysis Settings ==========
with st.sidebar:
    st.markdown("## 🎛️ Analysis Settings")
    
    # Condition selection
    st.markdown("### Condition Comparison")
    
    available_conditions = sorted(condition_groups.keys())
    
    if len(available_conditions) < 2:
        st.error("Need at least 2 conditions for differential expression analysis")
        st.stop()
    
    group1 = st.selectbox(
        "Group 1 (control/reference)",
        options=available_conditions,
        index=0,
        key="de_group1"
    )
    
    group2 = st.selectbox(
        "Group 2 (treatment/test)",
        options=[c for c in available_conditions if c != group1],
        index=0,
        key="de_group2"
    )
    
    st.caption(f"**Comparison:** {group2} vs {group1}")
    st.caption(f"**Interpretation:** Positive log2FC = higher in {group2}")
    
    st.markdown("---")
    
    # Transformation
    st.markdown("### Transformation")
    transform_key = st.selectbox(
        "Select transformation",
        options=["log2", "log10", "sqrt", "cbrt", "yeo_johnson", "quantile"],
        format_func=lambda x: {
            "log2": "log2",
            "log10": "log10",
            "sqrt": "Square root",
            "cbrt": "Cube root",
            "yeo_johnson": "Yeo-Johnson",
            "quantile": "Quantile Norm",
        }[x],
        index=0,
        key="de_transform"
    )
    
    st.info("ℹ️ **Note:** log2 is recommended for interpretable fold changes.")
    
    st.markdown("---")
    
    # Statistical thresholds
    st.markdown("### Statistical Thresholds")
    
    fc_threshold = st.number_input(
        "log2 Fold Change threshold",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        key="de_fc_threshold",
        help="Absolute log2FC threshold (1.0 = 2-fold change)"
    )
    
    st.caption(f"Fold change: {2**fc_threshold:.2f}x")
    
    pval_threshold = st.number_input(
        "P-value threshold",
        min_value=0.001,
        max_value=0.1,
        value=0.05,
        step=0.01,
        format="%.3f",
        key="de_pval_threshold",
    )
    
    use_fdr = st.checkbox(
        "Use FDR instead of p-value",
        value=False,
        key="de_use_fdr",
    )
    
    st.markdown("---")
    
    # Minimum valid values
    st.markdown("### Data Requirements")
    
    min_valid = st.number_input(
        "Min valid values per group",
        min_value=2,
        max_value=10,
        value=2,
        step=1,
        key="de_min_valid",
    )

# ========== MAIN ANALYSIS ==========

# Get transformed data for selected transformation
transform_data_selected = get_transform_data(protein_model, transform_key)
if "last_filtered_data" in st.session_state and st.session_state.last_filtered_data is not None:
    # Apply same filter to transformed data
    transform_data_selected = transform_data_selected.loc[st.session_state.last_filtered_data.index]

# Get columns for each group
group1_cols = condition_groups[group1]
group2_cols = condition_groups[group2]

st.markdown(f"### Comparison: {group2} vs {group1}")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Group 1 samples", len(group1_cols))
    st.caption(", ".join(group1_cols))
with col_info2:
    st.metric("Group 2 samples", len(group2_cols))
    st.caption(", ".join(group2_cols))
with col_info3:
    st.metric("Total proteins", len(transform_data_selected))

st.markdown("---")

# ========== RUN/RESTART BUTTON ==========
col_btn1, col_btn2 = st.columns([1, 3])

with col_btn1:
    run_analysis = st.button("🔬 Run DE Analysis", type="primary", key="run_de")

with col_btn2:
    if "de_results" in st.session_state:
        st.caption("✅ Analysis cached. Click button to re-run with new parameters.")

if run_analysis:
    with st.spinner("Performing t-test analysis..."):
        results_df = perform_ttest_analysis(
            transform_data_selected,
            group1_cols,
            group2_cols,
            min_valid=min_valid
        )
        
        st.session_state.de_results = results_df
        st.session_state.de_params = {
            "group1": group1,
            "group2": group2,
            "group1_cols": group1_cols,
            "group2_cols": group2_cols,
            "transform_key": transform_key,
            "fc_threshold": fc_threshold,
            "pval_threshold": pval_threshold,
            "use_fdr": use_fdr,
        }
        st.success("✅ Analysis complete!")

# Display results if available
if "de_results" in st.session_state:
    results_df = st.session_state.de_results.copy()
    params = st.session_state.de_params
    
    # Classify results
    pval_col = "fdr" if use_fdr else "pvalue"
    results_df["regulation"] = results_df.apply(
        lambda row: classify_regulation(
            pd.Series({"log2fc": row["log2fc"], "pvalue": row[pval_col]}),
            fc_threshold,
            pval_threshold
        ),
        axis=1
    )
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    n_up = (results_df["regulation"] == "Up-regulated").sum()
    n_down = (results_df["regulation"] == "Down-regulated").sum()
    n_ns = (results_df["regulation"] == "Not significant").sum()
    n_not_tested = (results_df["regulation"] == "Not tested").sum()
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Up-regulated", f"{n_up:,}", delta=f"{n_up/len(results_df)*100:.1f}%")
    col_s2.metric("Down-regulated", f"{n_down:,}", delta=f"{n_down/len(results_df)*100:.1f}%")
    col_s3.metric("Not significant", f"{n_ns:,}")
    col_s4.metric("Not tested", f"{n_not_tested:,}")
    
    st.markdown("---")
    
    # ========== THEORETICAL FOLD CHANGES - SIMPLE INPUT ==========
    st.markdown("### 🎯 Theoretical Fold Changes (Optional)")
    st.caption("Enter expected log2 fold changes for spike-in proteins to calculate error rates")
    
    col_theo1, col_theo2, col_theo3 = st.columns(3)
    
    with col_theo1:
        theo_protein_id = st.text_input(
            "Protein ID",
            placeholder="e.g., HUMAN_P12345",
            key="theo_protein_id"
        )
    
    with col_theo2:
        theo_log2fc = st.number_input(
            "log2 Fold Change",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            key="theo_log2fc"
        )
    
    with col_theo3:
        add_theo = st.button("➕ Add", key="add_theo")
    
    # Store theoretical values in session state
    if "de_theoretical" not in st.session_state:
        st.session_state.de_theoretical = {}
    
    if add_theo and theo_protein_id.strip():
        st.session_state.de_theoretical[theo_protein_id.strip()] = theo_log2fc
        st.success(f"✅ Added {theo_protein_id}: log2FC={theo_log2fc:.1f}")
    
    # Display added theoretical values
    if st.session_state.de_theoretical:
        st.markdown("#### Added Theoretical Values")
        
        theo_display = []
        for pid, fc in st.session_state.de_theoretical.items():
            theo_display.append({"Protein ID": pid, "log2FC": f"{fc:.2f}"})
        
        theo_df = pd.DataFrame(theo_display)
        
        col_theo_table, col_theo_clear = st.columns([3, 1])
        
        with col_theo_table:
            st.dataframe(theo_df, use_container_width=True, hide_index=True)
        
        with col_theo_clear:
            if st.button("🗑️ Clear All", key="clear_theo"):
                st.session_state.de_theoretical = {}
                st.rerun()
        
        # Calculate error rates
        if st.button("📊 Calculate Error Rates", key="calc_error_rates"):
            error_metrics = calculate_error_rates(
                results_df.copy(),
                st.session_state.de_theoretical,
                fc_threshold,
                pval_threshold
            )
            st.session_state.de_error_metrics = error_metrics
    
    # Display error rates if calculated
    if "de_error_metrics" in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Error Rate Analysis")
        
        metrics = st.session_state.de_error_metrics
        
        # Confusion matrix
        col_cm1, col_cm2 = st.columns(2)
        
        with col_cm1:
            st.markdown("#### Confusion Matrix")
            cm_df = pd.DataFrame({
                "Predicted Positive": [metrics["TP"], metrics["FP"]],
                "Predicted Negative": [metrics["FN"], metrics["TN"]],
            }, index=["Actual Positive", "Actual Negative"])
            st.dataframe(cm_df, use_container_width=True)
        
        with col_cm2:
            st.markdown("#### Performance Metrics")
            perf_df = pd.DataFrame({
                "Metric": ["Sensitivity", "Specificity", "Precision", "FPR", "FNR"],
                "Value": [
                    f"{metrics['sensitivity']:.1%}",
                    f"{metrics['specificity']:.1%}",
                    f"{metrics['precision']:.1%}",
                    f"{metrics['FPR']:.1%}",
                    f"{metrics['FNR']:.1%}",
                ]
            })
            st.dataframe(perf_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization tabs
    tab_volcano, tab_ma, tab_dist, tab_table = st.tabs(["Volcano Plot", "MA Plot", "Distribution + Boxplot", "Results Table"])
    
    with tab_volcano:
        st.markdown("### Volcano Plot")
        fig_volcano = create_volcano_plot(
            results_df.copy(),
            fc_threshold,
            pval_threshold,
            title=f"Volcano Plot: {params['group2']} vs {params['group1']}"
        )
        st.plotly_chart(fig_volcano, use_container_width=True)
    
    with tab_ma:
        st.markdown("### MA Plot")
        fig_ma = create_ma_plot(
            results_df.copy(),
            fc_threshold,
            pval_threshold
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    
    with tab_dist:
        st.markdown("### Distribution + Boxplot")
        st.caption("Left: Histogram with KDE curves | Right: Boxplots with mean ± SD")
        fig_combined = create_combined_distplot_boxplot(
            results_df.copy(),
            fc_threshold,
            pval_threshold
        )
        st.plotly_chart(fig_combined, use_container_width=True)
    
    with tab_table:
        st.markdown("### Results Table")
        
        # Filter options
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            show_filter = st.selectbox(
                "Show proteins",
                options=["All", "Significant only", "Up-regulated", "Down-regulated"],
                index=0,
                key="de_table_filter"
            )
        
        with col_f2:
            sort_by = st.selectbox(
                "Sort by",
                options=["P-value", "log2FC", "Mean abundance"],
                index=0,
                key="de_table_sort"
            )
        
        # Apply filter
        if show_filter == "Significant only":
            display_df = results_df[results_df["regulation"].isin(["Up-regulated", "Down-regulated"])]
        elif show_filter == "Up-regulated":
            display_df = results_df[results_df["regulation"] == "Up-regulated"]
        elif show_filter == "Down-regulated":
            display_df = results_df[results_df["regulation"] == "Down-regulated"]
        else:
            display_df = results_df
        
        # Sort
        if sort_by == "P-value":
            display_df = display_df.sort_values(pval_col)
        elif sort_by == "log2FC":
            display_df = display_df.sort_values("log2fc", ascending=False, key=abs)
        else:
            display_df["mean_abundance"] = (display_df["mean_group1"] + display_df["mean_group2"]) / 2
            display_df = display_df.sort_values("mean_abundance", ascending=False)
        
        # Display table
        display_cols = [
            "log2fc",
            "pvalue",
            "fdr",
            "mean_group1",
            "mean_group2",
            "regulation",
            "n_group1",
            "n_group2",
        ]
        
        styled_df = display_df[display_cols].style.format({
            "log2fc": "{:.3f}",
            "pvalue": "{:.2e}",
            "fdr": "{:.2e}",
            "mean_group1": "{:.2f}",
            "mean_group2": "{:.2f}",
            "n_group1": "{:.0f}",
            "n_group2": "{:.0f}",
        }).background_gradient(
            subset=["log2fc"],
            cmap="RdBu_r",
            vmin=-3,
            vmax=3
        )
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        st.caption(f"Showing {len(display_df):,} of {len(results_df):,} proteins")
    
    st.markdown("---")
    
    # Export results
    st.markdown("### Export Results")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv_all = results_df.to_csv(index=True)
        st.download_button(
            label="📥 Download All Results (CSV)",
            data=csv_all,
            file_name=f"de_results_{params['group2']}_vs_{params['group1']}.csv",
            mime="text/csv",
        )
    
    with col_exp2:
        sig_df = results_df[results_df["regulation"].isin(["Up-regulated", "Down-regulated"])]
        csv_sig = sig_df.to_csv(index=True)
        st.download_button(
            label="📥 Download Significant Only (CSV)",
            data=csv_sig,
            file_name=f"de_significant_{params['group2']}_vs_{params['group1']}.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Click 'Run DE Analysis' to start the analysis with current parameters")

render_navigation(back_page="pages/4_Filtering.py", next_page=None)
render_footer()
