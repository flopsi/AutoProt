# === 8. NORMALITY TESTING & RECOMMENDED TRANSFORMATION (Schessner et al., 2022) ===
st.subheader("Normality Testing & Recommended Transformation")

# Choose dataset
data_source = st.radio(
    "Apply transformation on:",
    ["All original data", "Final filtered data"],
    index=1,
    key="transform_data_source"
)

test_df = df if data_source == "All original data" else df_final

# All transformations to test
transform_options = {
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: stats.boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: stats.yeojohnson(x + 1)[0],
}
# === 8. NORMALITY TESTING & RECOMMENDED TRANSFORMATION (Schessner et al., 2022) ===
st.subheader("Normality Testing & Recommended Transformation")

# Choose dataset
data_source = st.radio(
    "Apply transformation on:",
    ["All original data", "Final filtered data"],
    index=1,
    key="transform_data_source"
)

test_df = df if data_source == "All original data" else df_final

# All transformations to test
transform_options = {
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: stats.boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: stats.yeojohnson(x + 1)[0],
}

# Find best transformation
best_transform = "Raw"
best_score = float('inf')
results = []

for rep in all_reps:
    raw_vals = test_df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8:
        continue
        
    row = {"Replicate": rep}
    
    # Raw stats
    raw_skew = stats.skew(raw_vals)
    raw_kurt = stats.kurtosis(raw_vals)
    _, raw_p = stats.shapiro(raw_vals)
    row["Raw Skew"] = f"{raw_skew:+.3f}"
    row["Raw Kurtosis"] = f"{raw_kurt:+.3f}"
    row["Raw p-value"] = f"{raw_p:.2e}"
    
    rep_best = "Raw"
    rep_score = float('inf')
    
    for name, func in transform_options.items():
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)): 
                row[f"{name} Tested"] = "❌"
                continue
                
            skew = stats.skew(t_vals)
            kurt = stats.kurtosis(t_vals)
            _, p = stats.shapiro(t_vals)
            
            score = abs(skew) + abs(kurt - 3)
            row[f"{name} Tested"] = "✓"
            
            if score < rep_score:
                rep_score = score
                rep_best = name
                
        except:
            row[f"{name} Tested"] = "❌"
    
    row["Recommended"] = rep_best
    
    # Show only recommended transformation
    if rep_best != "Raw":
        try:
            final_vals = transform_options[rep_best](raw_vals)
            row["After Skew"] = f"{stats.skew(final_vals):+.3f}"
            row["After Kurtosis"] = f"{stats.kurtosis(final_vals):+.3f}"
            _, p_final = stats.shapiro(final_vals)
            row["After p-value"] = f"{p_final:.2e}"
        except:
            row["After Skew"] = "—"
            row["After Kurtosis"] = "—"
            row["After p-value"] = "—"
    else:
        row["After Skew"] = "—"
        row["After Kurtosis"] = "—"
        row["After p-value"] = "—"
    
    if rep_score < best_score:
        best_score = rep_score
        best_transform = rep_best
        
    results.append(row)

# Display
results_df = pd.DataFrame(results)
st.table(results_df)

# Final recommendation
st.success(f"**Recommended transformation: {best_transform}**")
st.info("Based on minimizing skewness + excess kurtosis — Schessner et al., 2022")

if best_transform in ["log₂", "log₁₀"]:
    st.info("**Log transformation** is the gold standard in proteomics — stabilizes variance and normalizes distributions")
elif best_transform in ["Box-Cox", "Yeo-Johnson"]:
    st.info("**Power transformation** optimal — handles non-constant variance")
else:
    st.warning("No strong improvement — data may already be suitable for parametric tests")

# === PROCEED BUTTON AT BOTTOM ===
st.markdown("### Confirm & Proceed")
if st.button("Accept Transformation & Proceed to Differential Analysis", type="primary"):
    # Apply best transformation to final filtered data
    if best_transform != "Raw":
        transform_func = transform_options[best_transform]
        transformed_data = test_df[all_reps].replace(0, np.nan).apply(transform_func)
    else:
        transformed_data = test_df[all_reps].replace(0, np.nan)
    
    st.session_state.intensity_transformed = transformed_data
    st.session_state.df_filtered = test_df
    st.session_state.transform_applied = best_transform
    st.session_state.qc_accepted = True
    st.success("**Transformation applied!** Ready for differential analysis.")
    st.balloons()
# Find best transformation
best_transform = "Raw"
best_score = float('inf')
results = []

for rep in all_reps:
    raw_vals = test_df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8:
        continue
        
    row = {"Replicate": rep}
    
    # Raw stats
    raw_skew = stats.skew(raw_vals)
    raw_kurt = stats.kurtosis(raw_vals)
    _, raw_p = stats.shapiro(raw_vals)
    row["Raw Skew"] = f"{raw_skew:+.3f}"
    row["Raw Kurtosis"] = f"{raw_kurt:+.3f}"
    row["Raw p-value"] = f"{raw_p:.2e}"
    
    rep_best = "Raw"
    rep_score = float('inf')
    
    for name, func in transform_options.items():
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)): 
                row[f"{name} Tested"] = "❌"
                continue
                
            skew = stats.skew(t_vals)
            kurt = stats.kurtosis(t_vals)
            _, p = stats.shapiro(t_vals)
            
            score = abs(skew) + abs(kurt - 3)
            row[f"{name} Tested"] = "✓"
            
            if score < rep_score:
                rep_score = score
                rep_best = name
                
        except:
            row[f"{name} Tested"] = "❌"
    
    row["Recommended"] = rep_best
    
    # Show only recommended transformation
    if rep_best != "Raw":
        try:
            final_vals = transform_options[rep_best](raw_vals)
            row["After Skew"] = f"{stats.skew(final_vals):+.3f}"
            row["After Kurtosis"] = f"{stats.kurtosis(final_vals):+.3f}"
            _, p_final = stats.shapiro(final_vals)
            row["After p-value"] = f"{p_final:.2e}"
        except:
            row["After Skew"] = "—"
            row["After Kurtosis"] = "—"
            row["After p-value"] = "—"
    else:
        row["After Skew"] = "—"
        row["After Kurtosis"] = "—"
        row["After p-value"] = "—"
    
    if rep_score < best_score:
        best_score = rep_score
        best_transform = rep_best
        
    results.append(row)

# Display
results_df = pd.DataFrame(results)
st.table(results_df)

# Final recommendation
st.success(f"**Recommended transformation: {best_transform}**")
st.info("Based on minimizing skewness + excess kurtosis — Schessner et al., 2022")

if best_transform in ["log₂", "log₁₀"]:
    st.info("**Log transformation** is the gold standard in proteomics — stabilizes variance and normalizes distributions")
elif best_transform in ["Box-Cox", "Yeo-Johnson"]:
    st.info("**Power transformation** optimal — handles non-constant variance")
else:
    st.warning("No strong improvement — data may already be suitable for parametric tests")

# === PROCEED BUTTON AT BOTTOM ===
st.markdown("### Confirm & Proceed")
if st.button("Accept Transformation & Proceed to Differential Analysis", type="primary"):
    # Apply best transformation to final filtered data
    if best_transform != "Raw":
        transform_func = transform_options[best_transform]
        transformed_data = test_df[all_reps].replace(0, np.nan).apply(transform_func)
    else:
        transformed_data = test_df[all_reps].replace(0, np.nan)
    
    st.session_state.intensity_transformed = transformed_data
    st.session_state.df_filtered = test_df
    st.session_state.transform_applied = best_transform
    st.session_state.qc_accepted = True
    st.success("**Transformation applied!** Ready for differential analysis.")
    st.balloons()
