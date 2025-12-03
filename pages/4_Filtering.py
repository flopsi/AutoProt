import streamlit as st
import numpy as np
import pandas as pd

# Assume these already exist:
# SPECIES_ORDER, TRANSFORMS, compute_stats, apply_filters, protein_model, df_raw, numeric_cols, protein_species_col

# ---------- 1. Build filter row model ----------
if "filter_state_df" not in st.session_state:
    st.session_state.filter_state_df = pd.DataFrame(
        [{
            "species": ["HUMAN", "ECOLI", "YEAST"],  # multiselect
            "all_species": False,
            "use_min_peptides": True,
            "min_peptides": 1,
            "use_cv": True,
            "cv_cutoff": 30,
            "use_missing": True,
            "max_missing_pct": 34,
            "transform": "log2",
            "use_intensity": False,
        }]
    )

filters_df = st.session_state.filter_state_df

# ---------- 2. Editable filter table ----------
st.markdown("### Filter Settings")

edited = st.data_editor(
    filters_df,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={
        "species": st.column_config.MultiselectColumn(
            "Species",
            options=["HUMAN", "ECOLI", "YEAST", "MOUSE"],
            default=["HUMAN", "ECOLI", "YEAST"],
            help="Species to keep",
            color=["#87CEEB", "#008B8B", "#FF8C00", "#9370DB"],
        ),
        "all_species": st.column_config.CheckboxColumn(
            "All species",
            default=False,
            help="Select all species",
        ),
        "use_min_peptides": st.column_config.CheckboxColumn(
            "Use min peptides",
            default=True,
        ),
        "min_peptides": st.column_config.NumberColumn(
            "Min peptides/protein",
            min_value=1,
            max_value=10,
            step=1,
            format="%d",
        ),
        "use_cv": st.column_config.CheckboxColumn(
            "Use CV cutoff",
            default=True,
        ),
        "cv_cutoff": st.column_config.NumberColumn(
            "CV% cutoff",
            min_value=0,
            max_value=100,
            step=5,
            format="%d",
        ),
        "use_missing": st.column_config.CheckboxColumn(
            "Use max missing",
            default=True,
        ),
        "max_missing_pct": st.column_config.NumberColumn(
            "Max missing %",
            min_value=0,
            max_value=100,
            step=5,
            format="%d",
        ),
        "transform": st.column_config.SelectboxColumn(
            "Transformation",
            options=list(TRANSFORMS.keys()),
            format_func=TRANSFORMS.get,
        ),
        "use_intensity": st.column_config.CheckboxColumn(
            "Use intensity range",
            default=False,
        ),
    },
)

# Persist edits
st.session_state.filter_state_df = edited
row = edited.iloc[0]

# ---------- 3. Derive effective filter values ----------
# Species
if row["all_species"]:
    selected_species = ["HUMAN", "ECOLI", "YEAST", "MOUSE"]
else:
    selected_species = row["species"] or []

st.caption(
    "**Active species:** "
    + (", ".join(selected_species) if selected_species else "None (no species filter)")
)

# Scalars + flags
min_peptides = int(row["min_peptides"]) if row["use_min_peptides"] else 1
cv_cutoff = float(row["cv_cutoff"]) if row["use_cv"] else 1000.0
max_missing_ratio = (
    float(row["max_missing_pct"]) / 100.0 if row["use_missing"] else 1.0
)
transform_key = row["transform"]
use_intensity = bool(row["use_intensity"])

# ---------- 4. Optional intensity range widget ----------
intensity_range = None
transform_data = get_transform_data(protein_model, transform_key)

if use_intensity:
    min_intensity = float(transform_data[numeric_cols].min().min())
    max_intensity = float(transform_data[numeric_cols].max().max())
    intensity_range = st.slider(
        "Intensity range (mean per protein)",
        min_value=min_intensity,
        max_value=max_intensity,
        value=(min_intensity, max_intensity),
        key="intensity_range_slider",
    )

# ---------- 5. Apply filters + show before stats ----------
initial_stats = compute_stats(df_raw, protein_model, numeric_cols, protein_species_col)

st.markdown("### Before Filtering")
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Total Proteins", f"{initial_stats['n_proteins']:,}")

with c2:
    species_str = ", ".join(
        f"{s}:{initial_stats['species_counts'].get(s, 0)}"
        for s in SPECIES_ORDER
        if s in initial_stats["species_counts"]
    )
    st.metric("Species Count", species_str or "N/A")

with c3:
    st.metric(
        "Mean CV%",
        f"{initial_stats['cv_mean']:.1f}"
        if not np.isnan(initial_stats["cv_mean"])
        else "N/A",
    )

with c4:
    st.metric(
        "Median CV%",
        f"{initial_stats['cv_median']:.1f}"
        if not np.isnan(initial_stats["cv_median"])
        else "N/A",
    )

with c5:
    st.metric(
        "PERMANOVA F",
        f"{initial_stats['permanova_f']:.2f}"
        if not np.isnan(initial_stats["permanova_f"])
        else "N/A",
    )

with c6:
    st.metric(
        "Shapiro W",
        f"{initial_stats['shapiro_w']:.4f}"
        if not np.isnan(initial_stats["shapiro_w"])
        else "N/A",
    )

st.markdown("---")

# Apply filters (reusing your existing apply_filters)
filtered_df = apply_filters(
    df_raw,
    protein_model,
    numeric_cols,
    protein_species_col,
    selected_species,
    min_peptides,
    cv_cutoff,
    max_missing_ratio,
    intensity_range,
    transform_key,
)
