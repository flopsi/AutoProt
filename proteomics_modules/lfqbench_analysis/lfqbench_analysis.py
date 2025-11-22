"""
LFQbench Analysis Module - Core Analysis Engine
Implements benchmark proteomics analysis with fold-change calculations,
performance metrics, and statistical testing following LFQbench methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA


@dataclass
class BenchmarkConfig:
    """Configuration for LFQbench analysis"""
    expected_fc_human: float = 0.0
    expected_fc_yeast: float = 1.0
    expected_fc_ecoli: float = -2.0
    expected_fc_celegans: float = -1.0
    
    limit_mv: float = 2/3
    limit_cv: float = 20.0
    limit_fc: float = 0.5
    
    alpha_limma: float = 0.01
    
    color_human: str = "#199d76"
    color_yeast: str = "#d85f02"
    color_ecoli: str = "#7570b2"
    color_celegans: str = "darkred"


class LFQbenchAnalyzer:
    """Main LFQbench analysis engine"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.expected_fc_map = {
            'Human': self.config.expected_fc_human,
            'Yeast': self.config.expected_fc_yeast,
            'E. coli': self.config.expected_fc_ecoli,
            'C.elegans': self.config.expected_fc_celegans
        }
    
    def calculate_cv(self, values: np.ndarray) -> float:
        """Calculate coefficient of variation"""
        if len(values) < 2 or np.all(np.isnan(values)):
            return np.nan
        
        clean_values = values[~np.isnan(values)]
        if len(clean_values) < 2:
            return np.nan
        
        mean = np.mean(clean_values)
        if mean == 0:
            return np.nan
        
        std = np.std(clean_values, ddof=1)
        return (std / mean) * 100
    
    def calculate_log2_fc(self, exp_values: np.ndarray, ctr_values: np.ndarray) -> float:
        """Calculate log2 fold-change"""
        exp_mean = np.nanmean(exp_values)
        ctr_mean = np.nanmean(ctr_values)
        
        if exp_mean <= 0 or ctr_mean <= 0 or np.isnan(exp_mean) or np.isnan(ctr_mean):
            return np.nan
        
        return np.log2(exp_mean / ctr_mean)
    
    def filter_by_completeness(self, df: pd.DataFrame, exp_cols: List[str], ctr_cols: List[str]) -> pd.DataFrame:
        """Filter proteins by data completeness"""
        min_valid = int(np.ceil(min(len(exp_cols), len(ctr_cols)) * (1 - self.config.limit_mv)))
        
        df['exp_valid'] = df[exp_cols].notna().sum(axis=1)
        df['ctr_valid'] = df[ctr_cols].notna().sum(axis=1)
        
        filtered = df[(df['exp_valid'] >= min_valid) & (df['ctr_valid'] >= min_valid)].copy()
        filtered.drop(columns=['exp_valid', 'ctr_valid'], inplace=True)
        
        return filtered
    
    def calculate_cvs(self, df: pd.DataFrame, exp_cols: List[str], ctr_cols: List[str]) -> pd.DataFrame:
        """Calculate CVs for experimental and control conditions"""
        df['exp_cv'] = df[exp_cols].apply(lambda row: self.calculate_cv(row.values), axis=1)
        df['ctr_cv'] = df[ctr_cols].apply(lambda row: self.calculate_cv(row.values), axis=1)
        return df
    
    def filter_by_cv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter proteins by CV threshold"""
        return df[(df['exp_cv'] <= self.config.limit_cv) & (df['ctr_cv'] <= self.config.limit_cv)].copy()
    
    def calculate_fold_changes(self, df: pd.DataFrame, exp_cols: List[str], ctr_cols: List[str]) -> pd.DataFrame:
        """Calculate fold-changes for all proteins"""
        df['exp_mean'] = df[exp_cols].mean(axis=1)
        df['ctr_mean'] = df[ctr_cols].mean(axis=1)
        
        df['log2_fc'] = df.apply(lambda row: self.calculate_log2_fc(row[exp_cols].values, row[ctr_cols].values), axis=1)
        
        df['expected_log2_fc'] = df['Species'].map(self.expected_fc_map)
        df['fc_deviation'] = np.abs(df['log2_fc'] - df['expected_log2_fc'])
        
        return df
    
    def perform_limma_test(self, df: pd.DataFrame, exp_cols: List[str], ctr_cols: List[str]) -> pd.DataFrame:
        """Perform differential expression analysis"""
        
        p_values = []
        t_stats = []
        
        for idx, row in df.iterrows():
            exp_vals = pd.to_numeric(row[exp_cols], errors='coerce').dropna().values.astype(float)
            ctr_vals = pd.to_numeric(row[ctr_cols], errors='coerce').dropna().values.astype(float)
            
            if len(exp_vals) < 2 or len(ctr_vals) < 2:
                p_values.append(1.0)
                t_stats.append(0.0)
                continue
            
            try:
                t_stat, p_val = stats.ttest_ind(exp_vals, ctr_vals, equal_var=False)
                p_values.append(float(p_val) if not pd.isna(p_val) else 1.0)
                t_stats.append(float(t_stat) if not pd.isna(t_stat) else 0.0)
            except:
                p_values.append(1.0)
                t_stats.append(0.0)
        
        df['p_value'] = p_values
        df['t_statistic'] = t_stats
        df['p_adj'] = self._benjamini_hochberg(df['p_value'].values)
        df['is_significant'] = (df['p_adj'] < self.config.alpha_limma) & (np.abs(df['log2_fc']) > self.config.limit_fc)
        
        return df
    
    def _benjamini_hochberg(self, p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction"""
        p_values = np.array(p_values)
        n = len(p_values)
        
        mask = ~np.isnan(p_values)
        adjusted = np.full(n, np.nan)
        
        if np.sum(mask) == 0:
            return adjusted
        
        sorted_idx = np.argsort(p_values[mask])
        sorted_p = p_values[mask][sorted_idx]
        
        adjusted_p = np.minimum.accumulate(sorted_p * n / np.arange(1, len(sorted_p) + 1)[::-1])[::-1]
        adjusted[mask] = adjusted_p[np.argsort(sorted_idx)]
        
        return adjusted
    
    def classify_de_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify results as TP, TN, FP, FN"""
        
        def classify_row(row):
            expected_de = np.abs(row['expected_log2_fc']) > self.config.limit_fc
            observed_de = row['is_significant']
            
            if expected_de and observed_de:
                return 'true positive'
            elif expected_de and not observed_de:
                return 'false negative'
            elif not expected_de and observed_de:
                return 'false positive'
            else:
                return 'true negative'
        
        df['de_result'] = df.apply(classify_row, axis=1)
        return df
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        tp = len(df[df['de_result'] == 'true positive'])
        tn = len(df[df['de_result'] == 'true negative'])
        fp = len(df[df['de_result'] == 'false positive'])
        fn = len(df[df['de_result'] == 'false negative'])
        
        sensitivity = float((tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0)
        specificity = float((tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0)
        de_fdr = float((fp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0)
        
        accuracy = float(df['fc_deviation'].mean() if 'fc_deviation' in df.columns and len(df['fc_deviation'].dropna()) > 0 else 0.0)
        
        trueness_values = []
        for species in self.expected_fc_map.keys():
            species_data = df[df['Species'] == species]
            if len(species_data) > 0 and 'log2_fc' in species_data.columns:
                median_fc = float(species_data['log2_fc'].median())
                if not pd.isna(median_fc):
                    expected_fc = self.expected_fc_map[species]
                    trueness_values.append(abs(median_fc - expected_fc))
        
        trueness = float(np.sum(trueness_values) if trueness_values else 0.0)
        
        cv_exp_mean = df['exp_cv'].mean() if 'exp_cv' in df.columns else 0.0
        cv_ctr_mean = df['ctr_cv'].mean() if 'ctr_cv' in df.columns else 0.0
        cv_mean = float((cv_exp_mean + cv_ctr_mean) / 2)
        
        cv_exp_median = df['exp_cv'].median() if 'exp_cv' in df.columns else 0.0
        cv_ctr_median = df['ctr_cv'].median() if 'ctr_cv' in df.columns else 0.0
        cv_median = float((cv_exp_median + cv_ctr_median) / 2)
        
        return {
            'tp': float(tp),
            'tn': float(tn),
            'fp': float(fp),
            'fn': float(fn),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'de_fdr': de_fdr,
            'accuracy': accuracy,
            'trueness': trueness,
            'cv_mean': cv_mean,
            'cv_median': cv_median,
            'n_proteins': float(len(df))
        }
    
    def calculate_asymmetry_factor(self, log2_fc_values: np.ndarray) -> float:
        """Calculate asymmetry factor using quartile method"""
        log2_fc_values = log2_fc_values[~np.isnan(log2_fc_values)]
        
        if len(log2_fc_values) < 10:
            return np.nan
        
        q1 = float(np.percentile(log2_fc_values, 25))
        q2 = float(np.percentile(log2_fc_values, 50))
        q3 = float(np.percentile(log2_fc_values, 75))
        
        q2_q1 = abs(q2 - q1)
        q3_q2 = abs(q3 - q2)
        
        if q3_q2 == 0:
            return np.nan
        
        asymmetry = q2_q1 / q3_q2
        return float(asymmetry)
    
    def calculate_asymmetry_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate asymmetry factors for each species"""
        
        asymmetry_data = []
        
        for species in self.expected_fc_map.keys():
            species_data = df[df['Species'] == species]
            
            if len(species_data) >= 1:  # CHANGED FROM >= 10
                fc_values = species_data['log2_fc'].dropna().values
                
                if len(fc_values) >= 2:  # Need at least 2 values for percentile
                    asymmetry = self.calculate_asymmetry_factor(fc_values)
                    
                    asymmetry_data.append({
                        'Species': species,
                        'Asymmetry Factor': float(asymmetry) if not np.isnan(asymmetry) else np.nan,
                        'N Proteins': int(len(species_data))
                    })
        
        result_df = pd.DataFrame(asymmetry_data)
        print(f"DEBUG: Asymmetry DF created with {len(result_df)} rows")
        return result_df

    
    def run_complete_analysis(self, df: pd.DataFrame, exp_cols: List[str], ctr_cols: List[str]) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Run complete LFQbench analysis pipeline"""
        
        print("Step 1: Converting columns to numeric...")
        for col in exp_cols + ctr_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("Step 2: Filtering by data completeness...")
        df_filtered = self.filter_by_completeness(df, exp_cols, ctr_cols)
        print(f"  {len(df_filtered)}/{len(df)} proteins passed")
        
        print("Step 3: Calculating CVs...")
        df_filtered = self.calculate_cvs(df_filtered, exp_cols, ctr_cols)
        
        print("Step 4: Filtering by CV...")
        df_cv_filtered = self.filter_by_cv(df_filtered)
        print(f"  {len(df_cv_filtered)}/{len(df_filtered)} proteins passed")
        
        print("Step 5: Calculating fold-changes...")
        df_fc = self.calculate_fold_changes(df_cv_filtered, exp_cols, ctr_cols)
        
        print("Step 6: Performing t-tests...")
        df_de = self.perform_limma_test(df_fc, exp_cols, ctr_cols)
        
        print("Step 7: Classifying results...")
        df_classified = self.classify_de_results(df_de)
        
        print("Step 8: Calculating metrics...")
        metrics = self.calculate_performance_metrics(df_classified)
        
        print("Step 9: Calculating asymmetry...")
        asymmetry_df = self.calculate_asymmetry_metrics(df_classified)
        print(f"  Asymmetry: {len(asymmetry_df)} species calculated")
        
        print("✅ Analysis complete!")
        
        return df_classified, metrics, asymmetry_df
    
    def perform_pca(self, df: pd.DataFrame, sample_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform PCA on sample data"""
        data_matrix = df[sample_cols].T.dropna(axis=1)
        data_matrix = data_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        pca = PCA(n_components=min(2, data_scaled.shape[1]))
        pca_result = pca.fit_transform(data_scaled)
        
        return pca_result, pca.explained_variance_ratio_


def get_lfqbench_analyzer(config: Optional[BenchmarkConfig] = None) -> LFQbenchAnalyzer:
    """Get LFQbench analyzer instance"""
    return LFQbenchAnalyzer(config)
