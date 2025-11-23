# DIA Proteomics Analysis Framework - File Structure

## 📁 Project Structure

```
project_root/
├── Home.py                          # Main homepage (entrypoint)
├── pages/
│   ├── 1_🔬_Protein_Upload.py      # Protein-level data upload
│   ├── 2_🧪_Peptide_Upload.py      # Peptide-level data upload (optional)
│   └── 3_📊_Analysis.py             # Statistical analysis & visualization
├── models.py                         # Data classes and enums
├── config.py                         # Column detection & trimming utilities
└── README.md                         # This file
```

## 🎯 Key Improvements Implemented

### 1. **Multipage Architecture**
✅ Follows Streamlit best practices with `pages/` directory
✅ Homepage (`Home.py`) provides overview and navigation
✅ Separate pages for protein and peptide data
✅ Analysis page for statistical processing

### 2. **Data Classes & Enums**
✅ `DataLevel` enum - PROTEIN/PEPTIDE
✅ `Condition` enum - CONTROL/TREATMENT  
✅ `StatisticalTest` enum with smart properties:
   - `description` - Human-readable test description
   - `requires_normality` - Whether test assumes normal distribution
   - `min_groups` - Minimum groups required
✅ `NormalizationMethod` enum - None/Log2/Median/Quantile/Z-Score
✅ `ImputationMethod` enum - None/Zero/Min/Mean/Median/KNN
✅ `ColumnMetadata` dataclass - Column information
✅ `DatasetConfig` dataclass - Configuration with validation
✅ `ProteomicsDataset` dataclass - Complete dataset wrapper
✅ `AnalysisParams` dataclass - Analysis parameters with validation
✅ `SessionKeys` enum - Type-safe session state keys

### 3. **Proper Button Behavior**
✅ Uses `st.button()` without session state (Streamlit recommended pattern)
✅ Buttons trigger actions immediately
✅ Results stored in session state after action
✅ `type="primary"` for main actions
✅ `use_container_width=True` for full-width buttons

### 4. **Enhanced Error Handling**
✅ Validation in dataclass `__post_init__` methods
✅ `DatasetConfig.validate()` returns list of warnings
✅ Duplicate column name detection and auto-fixing
✅ Type checking via dataclasses

## 📊 Statistical Test Enum Example

```python
# Example usage of StatisticalTest enum
test = StatisticalTest.TTEST

print(test.description)
# "Parametric test for two groups (assumes normal distribution)"

print(test.requires_normality)
# True

print(test.min_groups)
# 2

# Iterate over all tests
for test in StatisticalTest:
    print(f"{test.value}: {test.description}")
```

## 🔄 Data Flow

```
1. User uploads file → Protein/Peptide Upload page
2. Preview data → Show 100 rows for column selection
3. User selects columns → Interactive data editor
4. User assigns roles → Protein Group, Species, Control/Treatment
5. Click "Load Full Dataset" → Create ProteomicsDataset
6. Store in session state → SessionKeys.PROTEIN_DATASET/PEPTIDE_DATASET
7. Navigate to Analysis → Configure parameters (AnalysisParams)
8. Run analysis → Results stored in SessionKeys.RESULTS
```

## 🎨 Theme Support

CSS automatically adapts to light/dark mode using:
```css
@media (prefers-color-scheme: dark) { ... }
@media (prefers-color-scheme: light) { ... }
```

## 📝 Next Steps to Complete

1. **Create `2_🧪_Peptide_Upload.py`** (similar to protein upload)
2. **Create `3_📊_Analysis.py`** (statistical analysis page)
3. **Implement analysis functions** using StatisticalTest enum
4. **Add visualization functions** for results

## 🚀 Running the App

```bash
streamlit run Home.py
```

## 💡 Key Features

- ✅ **Type Safety** - Dataclasses and enums prevent errors
- ✅ **Validation** - Built-in validation in data models
- ✅ **Separation of Concerns** - Models, config, and UI separated
- ✅ **Extensibility** - Easy to add new tests/methods via enums
- ✅ **Maintainability** - Clear structure and type hints
- ✅ **Best Practices** - Follows all Streamlit guidelines

---

**Status:** Ready for peptide upload page and analysis implementation!
