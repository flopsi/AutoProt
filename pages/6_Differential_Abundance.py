roteins)")
        
        res_sorted = res.dropna(subset=[pval_col]).sort_values(pval_col)
        n_top = min(15, len(res_sorted))
        
        if n_top > 0:
            top_proteins = res_sorted.head(n_top).index
            hm_data = df.loc[top_proteins, ref_samples + treat_samples]
            
            # Z-score normalize
            hm_z = (hm_data.T - hm_data.T.mean()) / (hm_data.T.std() + 1e-6)
            
            fig = go.Figure(data=go.Heatmap(
                z=hm_z.T.values,
                x=hm_z.T.columns,
                y=hm_z.T.index,
                colorscale="RdBu_r",
                zmid=0,
            ))
            fig.update_layout(height=600, width=1000)
            st.plotly_chart(fig, width="stretch")
    
    # P-value distribution
    if "P-value Distribution" in viz_options:
        st.markdown("### 📈 P-value Distribution")
        
        pvals = res[pval_col].dropna()
        
        fig = px.histogram(
            {"p": pvals},
            x="p",
            nbins=50,
            height=500,
            labels={"p": pval_col}
        )
        fig.add_vline(x=p_thr, line_dash="dash", line_color="red")
        st.plotly_chart(fig, width="stretch")
    
    # ROC Curve
    if "ROC Curve" in viz_options and theoretical_fc:
        st.markdown("### 📉 ROC Curve (Spike-in)")
        
        fpr_list, tpr_list, _ = compute_roc_curve(res, theoretical_fc, fc_thr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_list, y=tpr_list, mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600,
        )
        st.plotly_chart(fig, width="stretch")
    
    # Results table
    st.markdown("---")
    st.subheader("5️⃣ Detailed Results")
    
    with st.expander("📋 View All Results"):
        display_cols = ["log2fc", "pvalue", "fdr", "mean_g1", "mean_g2", "n_g1", "n_g2", "species", "regulation"]
        display_cols = [c for c in display_cols if c in res.columns]
        
        display_df = res[display_cols].copy()
        display_df = display_df.round(6)
        display_df = display_df.sort_values(pval_col)
        
        st.dataframe(display_df, height=600, use_container_width=True)
        
        csv = display_df.to_csv(index=True)
        st.download_button("📥 Download CSV", csv, f"dea_{ref_cond}_vs_{treat_cond}.csv", "text/csv")

else:
    st.info("👆 Configure and run analysis above")
'''
