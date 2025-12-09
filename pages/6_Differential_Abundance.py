"""
pages/6_Differential_Abundance.py
Limma-style empirical Bayes DA with composition-based spike-in validation.
- Comparison: A (reference) vs B (treatment)
- Significance by p/FDR only (no FC cutoff)
- MA/Volcano colored by species
"""

from __future__ import annotations

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

from scipy.stats import t as t_dist
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------
# LIMMA-STYLE HELPERS
# ---------------------------------------------------------------------


def fit_linear_model(
    df: pd.DataFrame, group1_cols: List[str], group2_cols: List[str]
) -> pd.DataFrame:
    """OLS per feature: group1 - group2 on log2 scale."""
    results: List[dict] = []

    for prot_id, row in df.iterrows():
        g1 = row[group1_cols].dropna()
        g2 = row[group2_cols].dropna()

        if len(g1) < 2 or len(g2) < 2:
            results.append(
                dict(
                    protein_id=prot_id,
                    log2fc=np.nan,
                    mean_g1=np.nan,
                    mean_g2=np.nan,
                    se=np.nan,
                    sigma=np.nan,
                    df=np.nan,
                    n_g1=len(g1),
                    n_g2=len(g2),
                )
            )
            continue

        mean_g1 = g1.mean()
        mean_g2 = g2.mean()
        log2fc = mean_g1 - mean_g2  # A vs B

        n1, n2 = len(g1), len(g2)
        var1 = g1.var(ddof=1)
        var2 = g2.var(ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        sigma = np.sqrt(pooled_var)
        se = sigma * np.sqrt(1 / n1 + 1 / n2)
        df_dof = n1 + n2 - 2

        results.append(
            dict(
                protein_id=prot_id,
                log2fc=log2fc,
                mean_g1=mean_g1,
                mean_g2=mean_g2,
                se=se,
                sigma=sigma,
                df=df_dof,
                n_g1=n1,
                n_g2=n2,
            )
        )

    out = pd.DataFrame(results).set_index("protein_id")
    return out


def empirical_bayes_moderation(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Moderate variances (limma-style)."""
    from scipy.special import polygamma

    valid = fit_df.dropna(subset=["sigma", "df"]).copy()
    if valid.empty:
        return fit_df

    s2 = valid["sigma"] ** 2
    df_gene = valid["df"]
    log_s2 = np.log(s2)
    m = log_s2.mean()
    v = log_s2.var()

    def trigamma_inv(y: float) -> float:
        if y < 1e-6:
            return 1 / y
        d = 1 / y
        for _ in range(10):
            tri = polygamma(1, d / 2)
            tetra = -0.5 * polygamma(2, d / 2)
            d = d - (tri - y) / tetra
        return d

    d0 = max(trigamma_inv(v), 1.0)
    s0_sq = np.exp(m + polygamma(1, d0 / 2))

    df_post = d0 + df_gene
    s2_post = (d0 * s0_sq + df_gene * s2) / df_post

    valid["s2_prior"] = s0_sq
    valid["df_prior"] = d0
    valid["s2_post"] = s2_post
    valid["df_post"] = df_post
    valid["se_post"] = np.sqrt(s2_post) * valid["se"] / valid["sigma"]
    valid["t_stat"] = valid["log2fc"] / valid["se_post"]
    valid["pvalue"] = 2 * (1 - t_dist.cdf(np.abs(valid["t_stat"]), valid["df_post"]))

    out = fit_df.copy()
    for col in ["s2_prior", "df_prior", "s2_post", "df_post", "se_post", "t_stat", "pvalue"]:
        out[col] = valid[col]
    return out


def benjamini_hochberg_fdr(pvals: pd.Series) -> pd.Series:
    """BH FDR."""
    v = pvals.dropna().sort_values()
    n = len(v)
    if n == 0:
        return pd.Series(index=pvals.index, dtype=float)

    ranks = np.arange(1, n + 1)
    f = v.values * n / ranks
    f = np.minimum.accumulate(f[::-1])[::-1]
    f = np.minimum(f, 1.0)
    mapping = dict(zip(v.index, f))
    return pvals.index.to_series().map(mapping)


# ---------------------------------------------------------------------
# ERROR METRICS (VARIABLE + STABLE PROTEOME)
# ---------------------------------------------------------------------


def error_metrics_variable(
    res: pd.DataFrame,
    true_fc: Dict[str, float],
    species_map: Dict[str, str],
    stable_thr: float = 0.5,
    p_thr: float = 0.05,
) -> Tuple[Dict, pd.DataFrame]:
    """Metrics for variable proteome (|true log2FC| >= stable_thr)."""
    df = res.copy()
    df = df[df["regulation"] != "not_tested"]

    df["species_key"] = df.index.map(lambda x: species_map.get(x, "Unknown"))
    df["true_log2fc"] = df["species_key"].map(lambda s: true_fc.get(s, np.nan))
    df = df.dropna(subset=["true_log2fc"])
    if df.empty:
        return {}, pd.DataFrame()

    df = df[np.abs(df["true_log2fc"]) >= stable_thr].copy()
    if df.empty:
        return {}, pd.DataFrame()

    df["true_regulated"] = np.abs(df["true_log2fc"]) >= stable_thr
    df["observed_regulated"] = df["regulation"].isin(["up", "down"])

    tp = int((df["true_regulated"] & df["observed_regulated"]).sum())
    fn = int((df["true_regulated"] & ~df["observed_regulated"]).sum())
    tn = int((~df["true_regulated"] & ~df["observed_regulated"]).sum())
    fp = int((~df["true_regulated"] & df["observed_regulated"]).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def wilson(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
        if trials == 0:
            return (0.0, 0.0)
        p = successes / trials
        denom = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denom
        adj = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
        return (max(0, center - adj), min(1, center + adj))

    sens_ci = wilson(tp, tp + fn)
    spec_ci = wilson(tn, tn + fp)

    overall = dict(
        Total_Variable=len(df),
        True_Positives=tp,
        False_Negatives=fn,
        True_Negatives=tn,
        False_Positives=fp,
        Sensitivity=sens,
        Sensitivity_CI=sens_ci,
        Specificity=spec,
        Specificity_CI=spec_ci,
        Precision=prec,
    )

    rows: List[dict] = []
    for sp in df["species_key"].unique():
        sub = df[df["species_key"] == sp].copy()
        theo = true_fc.get(sp, 0.0)
        err = sub["log2fc"] - theo
        n = len(sub)
        rmse = float(np.sqrt((err**2).mean()))
        mae = float(err.abs().mean())
        bias = float(err.mean())
        rows.append(
            dict(
                Species=sp,
                N=n,
                Theo_FC=f"{theo:.2f}",
                RMSE=rmse,
                MAE=mae,
                Bias=bias,
                Detection_Rate=float(sub["observed_regulated"].mean()),
            )
        )
    return overall, pd.DataFrame(rows)


def error_metrics_stable(
    res: pd.DataFrame,
    true_fc: Dict[str, float],
    species_map: Dict[str, str],
    stable_thr: float = 0.5,
) -> Tuple[Dict, pd.DataFrame]:
    """False positive metrics for stable proteome (|true log2FC| < stable_thr)."""
    df = res.copy()
    df = df[df["regulation"] != "not_tested"]

    df["species_key"] = df.index.map(lambda x: species_map.get(x, "Unknown"))
    df["true_log2fc"] = df["species_key"].map(lambda s: true_fc.get(s, np.nan))
    df = df.dropna(subset=["true_log2fc"])
    if df.empty:
        return {}, pd.DataFrame()

    df = df[np.abs(df["true_log2fc"]) < stable_thr].copy()
    if df.empty:
        return {}, pd.DataFrame()

    df["observed_regulated"] = df["regulation"].isin(["up", "down"])
    fp = int(df["observed_regulated"].sum())
    tn = int((~df["observed_regulated"]).sum())
    total = len(df)

    fpr = fp / total if total > 0 else 0.0
    tnr = tn / total if total > 0 else 0.0

    def wilson(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
        if trials == 0:
            return (0.0, 0.0)
        p = successes / trials
        denom = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denom
        adj = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
        return (max(0, center - adj), min(1, center + adj))

    fpr_ci = wilson(fp, total)
    tnr_ci = wilson(tn, total)

    overall = dict(
        Total_Stable=total,
        False_Positives=fp,
        True_Negatives=tn,
        False_Positive_Rate=fpr,
        FPR_CI=fpr_ci,
        True_Negative_Rate=tnr,
        TNR_CI=tnr_ci,
    )

    rows: List[dict] = []
    for sp in df["species_key"].unique():
        sub = df[df["species_key"] == sp].copy()
        n = len(sub)
        fp_s = int(sub["observed_regulated"].sum())
        tn_s = int((~sub["observed_regulated"]).sum())
        fpr_s = fp_s / n if n > 0 else 0.0
        mae = float(np.abs(sub["log2fc"]).mean())
        rows.append(
            dict(
                Species=sp,
                N=n,
                False_Positives=fp_s,
                True_Negatives=tn_s,
                FP_Rate=fpr_s,
                Mean_Abs_Error=mae,
            )
        )
    return overall, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(page_title="Differential Abundance", page_icon="🔬", layout="wide")
st.title("🔬 Differential Abundance Analysis")
st.markdown("Limma empirical Bayes on A vs B with spike-in based validation.")
st.markdown("---")

# ---------------------------------------------------------------------
# DATA AVAILABILITY
# ---------------------------------------------------------------------

if "df_imputed" not in st.session_state or st.session_state.df_imputed is None:
    st.error("No imputed data found. Please finish the Missing Value Imputation step first.")
    st.stop()

df_pl: pl.DataFrame = st.session_state.df_imputed
df = df_pl.to_pandas()  # for limma-style operations
numeric_cols: List[str] = st.session_state.numeric_cols
species_col: str = st.session_state.species_col
sample_to_condition: Dict[str, str] = st.session_state.sample_to_condition

conditions = sorted(set(sample_to_condition.values()))
cond_samples: Dict[str, List[str]] = {}
for s, c in sample_to_condition.items():
    if s in numeric_cols:
        cond_samples.setdefault(c, []).append(s)

st.info(
    f"Data: {df.shape[0]:,} proteins × {len(numeric_cols)} samples · "
    f"Conditions: {', '.join(conditions)}"
)

# ---------------------------------------------------------------------
# 1. CHOOSE A VS B
# ---------------------------------------------------------------------

st.subheader("1️⃣ Comparison Setup (A vs B)")

col1, col2 = st.columns(2)
with col1:
    ref_cond = st.selectbox("Condition A (reference)", options=conditions, index=0)
with col2:
    treat_cond = st.selectbox(
        "Condition B (treatment)",
        options=[c for c in conditions if c != ref_cond],
        index=0 if len(conditions) > 1 else 0,
    )

if ref_cond == treat_cond:
    st.error("Choose two different conditions.")
    st.stop()

ref_samples = cond_samples[ref_cond]
treat_samples = cond_samples[treat_cond]

st.markdown(
    f"- Log2FC is **A/B = {ref_cond}/{treat_cond}**\n"
    "- Positive log2FC → higher in A; negative → higher in B."
)
st.markdown("---")

# ---------------------------------------------------------------------
# 2. COMPOSITION-BASED EXPECTED FC
# ---------------------------------------------------------------------

st.subheader("2️⃣ Spike-in Composition (optional)")

use_comp = st.checkbox(
    "Provide % composition per species per condition to define expected log2FC",
    value=False,
)

theoretical_fc: Dict[str, float] = {}
species_values = sorted([s for s in df[species_col].unique() if isinstance(s, str)])

if use_comp:
    st.markdown("Enter percentage composition (will be normalized to 100% within each condition).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{ref_cond} (A)**")
        comp_a = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%) in {ref_cond}",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / max(len(species_values), 1),
                step=5.0,
                key=f"a_{sp}",
            )
            comp_a[sp] = val
        ta = sum(comp_a.values()) or 1.0
        comp_a = {k: v * 100 / ta for k, v in comp_a.items()}

    with c2:
        st.markdown(f"**{treat_cond} (B)**")
        comp_b = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%) in {treat_cond}",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / max(len(species_values), 1),
                step=5.0,
                key=f"b_{sp}",
            )
            comp_b[sp] = val
        tb = sum(comp_b.values()) or 1.0
        comp_b = {k: v * 100 / tb for k, v in comp_b.items()}

    # log2(A/B) per species
    rows = []
    for sp in species_values:
        pa = comp_a.get(sp, 0.0)
        pb = comp_b.get(sp, 0.0)
        if pa == 0 and pb == 0:
            log2fc = 0.0
        elif pb == 0:
            log2fc = 10.0
        elif pa == 0:
            log2fc = -10.0
        else:
            log2fc = float(np.log2(pa / pb))
        theoretical_fc[sp] = log2fc
        rows.append(
            dict(
                Species=sp,
                **{f"{ref_cond} (%)": f"{pa:.1f}", f"{treat_cond} (%)": f"{pb:.1f}"},
                Log2FC=f"{log2fc:.3f}",
                Linear_FC=f"{2**log2fc:.2f}x",
            )
        )

    st.markdown("**Expected log2FC from composition (A/B)**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # single combined visualization (remove “useless” right plot)
    st.markdown("**Composition per condition (stacked) and species-specific expected log2FC**")
    l, r = st.columns(2)
    with l:
        plot_rows = []
        for sp in species_values:
            plot_rows.append(dict(Condition=ref_cond, Species=sp, Percentage=comp_a.get(sp, 0.0)))
            plot_rows.append(dict(Condition=treat_cond, Species=sp, Percentage=comp_b.get(sp, 0.0)))
        comp_df = pd.DataFrame(plot_rows)
        figc = px.bar(
            comp_df,
            x="Condition",
            y="Percentage",
            color="Species",
            barmode="stack",
            labels={"Percentage": "Composition (%)"},
            title="Percentage composition per condition",
            height=450,
        )
        figc.update_yaxes(range=[0, 100])
        st.plotly_chart(figc, use_container_width=True)

    with r:
        fc_plot = pd.DataFrame(
            dict(Species=list(theoretical_fc.keys()), Log2FC=list(theoretical_fc.values()))
        )
        figf = px.bar(
            fc_plot,
            x="Species",
            y="Log2FC",
            color="Species",
            title=f"Expected log2FC (A/B = {ref_cond}/{treat_cond})",
            labels={"Log2FC": "Expected log2 fold change"},
            height=450,
        )
        figf.add_hline(y=0.0, line_color="red")
        st.plotly_chart(figf, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------
# 3. STATISTICAL SETTINGS (NO MIN FC)
# ---------------------------------------------------------------------

st.subheader("3️⃣ Statistical Settings")

c1, c2 = st.columns(2)
with c1:
    p_thr = st.selectbox(
        "FDR significance threshold",
        options=[0.001, 0.01, 0.05, 0.1],
        index=2,
        format_func=lambda x: f"{x*100:.1f} %",
    )
with c2:
    use_fdr = st.checkbox("Use FDR correction (BH)", value=True)

stable_thr = 0.5  # fixed: only for stable vs variable classification in metrics
st.caption(
    f"Stable proteome is defined as |expected log2FC| < {stable_thr}; only those contribute to false positives."
)

st.markdown("---")

# ---------------------------------------------------------------------
# 4. RUN LIMMA A VS B
# ---------------------------------------------------------------------

st.subheader("4️⃣ Run limma (A vs B)")

if st.button("Run analysis", type="primary"):
    with st.spinner("Fitting linear models and empirical Bayes..."):
        # log2 transform if not already on log scale
        df_num = df[numeric_cols]
        if (df_num > 50).any().any():
            df_log2 = np.log2(df_num + 1.0)
        else:
            df_log2 = df_num.copy()

        fit = fit_linear_model(df_log2, ref_samples, treat_samples)
        limma_res = empirical_bayes_moderation(fit)

        if use_fdr:
            limma_res["fdr"] = benjamini_hochberg_fdr(limma_res["pvalue"])
            test_col = "fdr"
        else:
            limma_res["fdr"] = limma_res["pvalue"]
            test_col = "pvalue"

        # significance purely by p/FDR (no FC cutoff)
        limma_res["regulation"] = limma_res.apply(
            lambda r: (
                "up"
                if (r[test_col] < p_thr and r["log2fc"] > 0)
                else "down"
                if (r[test_col] < p_thr and r["log2fc"] < 0)
                else "not_significant"
            ),
            axis=1,
        )

        limma_res["neg_log10_p"] = -np.log10(limma_res[test_col].replace(0, 1e-300))
        limma_res["species"] = limma_res.index.to_series().map(df[species_col])

        st.session_state.dea_results = limma_res
        st.session_state.dea_ref = ref_cond
        st.session_state.dea_treat = treat_cond
        st.session_state.dea_p_thr = p_thr
        st.session_state.dea_theoretical_fc = theoretical_fc

    st.success("Analysis finished.")

# ---------------------------------------------------------------------
# 5. RESULTS: MA/VOLCANO COLORED BY SPECIES
# ---------------------------------------------------------------------

if "dea_results" in st.session_state:
    res = st.session_state.dea_results
    ref_cond = st.session_state.dea_ref
    treat_cond = st.session_state.dea_treat
    p_thr = st.session_state.dea_p_thr
    theoretical_fc = st.session_state.dea_theoretical_fc

    st.markdown("---")
    st.subheader("5️⃣ Results overview")

    n_total = len(res)
    n_up = int((res["regulation"] == "up").sum())
    n_down = int((res["regulation"] == "down").sum())
    n_sig = n_up + n_down

    m1, m2, m3 = st.columns(3)
    m1.metric("Total proteins", f"{n_total:,}")
    m2.metric("Significant (p/FDR)", f"{n_sig:,}")
    m3.metric("Up / Down", f"{n_up} / {n_down}")

    # Volcano (colored by species)
    st.markdown("### Volcano plot (colored by species)")
    volc = res[res["regulation"] != "not_tested"].copy()
    fig_v = px.scatter(
        volc,
        x="log2fc",
        y="neg_log10_p",
        color="species",
        hover_data=["regulation"],
        labels={
            "log2fc": f"log2FC (A/B = {ref_cond}/{treat_cond})",
            "neg_log10_p": "-log10(FDR)" if use_fdr else "-log10(p)",
        },
        title="Volcano plot",
        height=550,
    )
    fig_v.add_hline(y=-np.log10(p_thr), line_dash="dash", line_color="gray")
    st.plotly_chart(fig_v, use_container_width=True)

    # MA plot (colored by species)
    st.markdown("### MA plot (colored by species)")
    ma = res[res["regulation"] != "not_tested"].copy()
    ma["A"] = (ma["mean_g1"] + ma["mean_g2"]) / 2
    ma["M"] = ma["log2fc"]
    fig_ma = px.scatter(
        ma,
        x="A",
        y="M",
        color="species",
        hover_data=["regulation"],
        labels={"A": "Mean log2 intensity", "M": "log2FC (A/B)"},
        title="MA plot",
        height=550,
    )
    fig_ma.add_hline(y=0.0, line_color="red")
    st.plotly_chart(fig_ma, use_container_width=True)

    # top table
    st.markdown("### Top significant proteins")
    top = res[res["regulation"].isin(["up", "down"])].sort_values("fdr").head(100)
    st.dataframe(
        top[["log2fc", "t_stat", "pvalue", "fdr", "regulation", "species"]].round(4),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # 6. VALIDATION METRICS (ONLY IF THEORETICAL FC GIVEN)
    # -----------------------------------------------------------------
    if theoretical_fc:
        st.markdown("---")
        st.subheader("6️⃣ Spike-in based error metrics")

        species_map = dict(zip(res.index.astype(str), res["species"].astype(str)))

        ov_var, sp_var = error_metrics_variable(
            res, theoretical_fc, species_map, stable_thr=stable_thr, p_thr=p_thr
        )
        ov_stab, sp_stab = error_metrics_stable(
            res, theoretical_fc, species_map, stable_thr=stable_thr
        )

        if ov_stab:
            st.markdown("**Stable proteome (false positives)**")
            c1, c2 = st.columns(2)
            c1.metric(
                "False positive rate",
                f"{ov_stab['False_Positive_Rate']:.1%}",
                help=f"CI {ov_stab['FPR_CI'][0]:.1%}–{ov_stab['FPR_CI'][1]:.1%}",
            )
            c2.metric("Stable proteins used", f"{ov_stab['Total_Stable']:,}")
            if not sp_stab.empty:
                st.dataframe(sp_stab, use_container_width=True)

        if ov_var:
            st.markdown("**Variable proteome (detection performance)**")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Sensitivity",
                f"{ov_var['Sensitivity']:.1%}",
                help=f"CI {ov_var['Sensitivity_CI'][0]:.1%}–{ov_var['Sensitivity_CI'][1]:.1%}",
            )
            c2.metric(
                "Specificity",
                f"{ov_var['Specificity']:.1%}",
                help=f"CI {ov_var['Specificity_CI'][0]:.1%}–{ov_var['Specificity_CI'][1]:.1%}",
            )
            c3.metric("Precision", f"{ov_var['Precision']:.1%}")
            if not sp_var.empty:
                st.dataframe(sp_var.round(3), use_container_width=True)

    # -----------------------------------------------------------------
    # 7. EXPORT
    # -----------------------------------------------------------------
    st.markdown("---")
    st.subheader("7️⃣ Export")

    st.download_button(
        "Download limma results (CSV)",
        data=res.to_csv().encode("utf-8"),
        file_name=f"limma_{ref_cond}_vs_{treat_cond}.csv",
        mime="text/csv",
    )
else:
    st.info("Configure settings and run the analysis to see results.")
