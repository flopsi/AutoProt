import streamlit as st
import polars as pl
from plotnine import *
import numpy as np

# ============================================================================
# LOAD DATA
# ============================================================================

if 'df' not in st.session_state:
    st.warning("⚠️ Please upload data first")
    st.stop()

df = st.session_state.df
numeric_cols = st.session_state.numeric_cols
species_col = st.session_state.species_col
id_col = st.session_state.id_col

# ============================================================================
# CACHE TRANSFORMS
# ============================================================================

@st.cache_data
def compute_log2(df_dict: dict, cols: list) -> dict:
    """Cache log2 transformation."""
    df = pl.from_dict(df_dict)
    df_log2 = df.with_columns([
        pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in cols
    ])
    return df_log2.to_dict(as_series=False)

# Get log2 data
df_log2 = pl.from_dict(compute_log2(df.to_dict(as_series=False), numeric_cols))

# ============================================================================
# PAGE
# ============================================================================

st.title("📊 Visual EDA")

st.info(f"📁 {df.shape[0]:,} proteins × {len(numeric_cols)} samples")
st.markdown("---")

# ============================================================================
# 1. OVERVIEW
# ============================================================================

st.subheader("1️⃣ Dataset Overview")

n_species = df[species_col].n_unique()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Proteins", f"{df.shape[0]:,}")
c2.metric("Total Samples", len(numeric_cols))
c3.metric("Species", n_species)
c4.metric("Avg/Species", int(df.shape[0] / n_species))

st.markdown("---")

# ============================================================================
# 2. STACKED BAR - PROTEINS PER SPECIES PER SAMPLE
# ============================================================================

st.subheader("2️⃣ Valid Proteins per Species per Sample")
st.info("**Valid = intensity > 1.0** (excludes missing/NaN/zero)")

# Count valid proteins (>1.0) per species per sample
df_counts = df.select([id_col, species_col] + numeric_cols).melt(
    id_vars=[id_col, species_col],
    value_vars=numeric_cols,
    variable_name='sample'
).filter(
    (pl.col('value') > 1.0) & (pl.col('value').is_finite())  # Valid = not 1.0 and finite
).group_by(['sample', species_col]).agg(
    pl.count().alias('count')
).sort(['sample', species_col]).with_columns([
    (pl.col('count').cum_sum().over('sample') - pl.col('count') / 2).alias('label_pos')
])

# Plot
plot = (ggplot(df_counts.to_pandas(), aes(x='sample', y='count', fill=species_col)) +
 geom_bar(stat='identity') +
 geom_text(aes(y='label_pos', label='count'), 
           size=8, color='white', fontweight='bold') +
 labs(title='Valid Protein Count by Species per Sample',
      x='Sample', y='Protein Count', fill='Species') +
 theme_minimal() +
 theme(axis_text_x=element_text(rotation=45, hjust=1),
       figure_size=(10, 5)))

st.pyplot(ggplot.draw(plot))

# ============================================================================
# TABLE - DETAILED COUNTS
# ============================================================================

# Pivot table: species × samples
df_table = df_counts.pivot(
    index=species_col,
    columns='sample',
    values='count'
).fill_null(0)

# Add total column (unique proteins per species across all samples with ANY valid value)
species_totals = df.select([id_col, species_col] + numeric_cols).melt(
    id_vars=[id_col, species_col],
    value_vars=numeric_cols
).filter(
    (pl.col('value') > 1.0) & (pl.col('value').is_finite())
).group_by([id_col, species_col]).agg(
    pl.count()  # Any valid measurement
).group_by(species_col).agg(
    pl.count().alias('Total')
)

df_table = df_table.join(species_totals, on=species_col).sort('Total', descending=True)

st.markdown("**Valid Proteins per Species per Sample:**")
st.dataframe(df_table.to_pandas(), use_container_width=True)

st.download_button(
    "📥 Download Table (CSV)",
    df_table.write_csv(),
    "valid_proteins_per_species.csv",
    "text/csv"
)

st.markdown("---")


# ============================================================================
# 3. VIOLIN/BOX PLOT - LOG2 INTENSITY DISTRIBUTION
# ============================================================================

st.subheader("3️⃣ Log2 Intensity Distribution by Sample")

# Long format for plotting
df_long = df_log2.select([id_col] + numeric_cols).melt(
    id_vars=[id_col],
    value_vars=numeric_cols,
    variable_name='sample',
    value_name='log2_intensity'
).filter(
    pl.col('log2_intensity').is_finite()
)

# Add condition grouping (A vs B)
df_long = df_long.with_columns(
    pl.when(pl.col('sample').str.starts_with('A'))
    .then(pl.lit('A'))
    .otherwise(pl.lit('B'))
    .alias('condition')
)

# Violin plot
plot = (ggplot(df_long.to_pandas(), aes(x='sample', y='log2_intensity', fill='condition')) +
 geom_violin(alpha=0.7) +
 geom_boxplot(width=0.1, fill='white', outlier_alpha=0.3) +
 scale_fill_manual(values=['#66c2a5', '#fc8d62']) +
 labs(title='Log2 Intensity Distribution',
      x='Sample', y='Log2 Intensity', fill='Condition') +
 theme_minimal() +
 theme(axis_text_x=element_text(rotation=45, hjust=1),
       figure_size=(12, 6)))

st.pyplot(ggplot.draw(plot))

# Summary statistics
df_stats = df_long.group_by('sample').agg([
    pl.col('log2_intensity').count().alias('n'),
    pl.col('log2_intensity').mean().alias('mean'),
    pl.col('log2_intensity').median().alias('median'),
    pl.col('log2_intensity').std().alias('std'),
    pl.col('log2_intensity').quantile(0.25).alias('q25'),
    pl.col('log2_intensity').quantile(0.75).alias('q75')
]).sort('sample')

st.markdown("**Summary Statistics:**")
st.dataframe(df_stats.to_pandas(), use_container_width=True)

st.download_button(
    "📥 Download Statistics (CSV)",
    df_stats.write_csv(),
    "log2_intensity_statistics.csv",
    "text/csv"
)
