# LFQ Proteomics Analysis Platform

**Thermo Fisher Scientific** - Mass Spectrometry Data Analysis Application

## Overview

Professional Streamlit-based application for analyzing, visualizing, and reporting Label-Free Quantification (LFQ) proteomics data from mass spectrometry experiments.

## Features

### Current (MVP v1.0)
- **Protein Data Upload**: CSV/TSV/Excel ingestion with auto-detection
- **Peptide Data Upload**: Linked peptide-level analysis
- **Species Detection**: Automatic identification of HUMAN, ECOLI, YEAST
- **Condition Assignment**: Semi-automated A/B condition mapping
- **Interactive Visualizations**: Species-specific count charts
- **Data Quality Module**: Template for future QC metrics

### Supported Workflows
- LFQ Bench: Two-condition comparison (A vs B)

## Installation

### 1. Clone Repository
```bash
git clone <repository_url>
cd proteomics_app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Quick Start

### Using Demo Data
1. Navigate to "Protein Upload" page
2. Check "Use demo data" checkbox
3. Follow on-screen instructions for column annotation
4. Confirm condition assignments (A vs B)
5. Review summary charts

### Using Your Own Data

#### Data Format Requirements

**Protein Data**:
```csv
Protein.Group,PG.ProteinNames,Sample_A1,Sample_A2,Sample_B1,Sample_B2
P12345,PROTEIN_HUMAN,1234.56,1456.78,890.12,923.45
...
```

**Peptide Data**:
```csv
Protein.Group,Peptide.Sequence,Sample_A1,Sample_A2,Sample_B1,Sample_B2
P12345,PEPTIDEK,567.89,589.12,234.56,245.67
...
```

**Requirements**:
- At least one metadata column (protein identifiers)
- One column containing species suffixes (`_HUMAN`, `_ECOLI`, `_YEAST`)
- Even number of quantitative columns
- Numeric columns may be stored as strings (auto-converted)

## Workflow

```
1. Upload Protein Data
   ↓
2. Select Species Column (auto-detected)
   ↓
3. Choose Workflow (LFQ Bench)
   ↓
4. Assign Conditions (A vs B, auto-detected)
   ↓
5. Review Summary Charts
   ↓
6. Upload Peptide Data (optional)
   ↓
7. [Future: Quality Assessment]
   ↓
8. [Future: Statistical Analysis]
```

## Project Structure

```
proteomics_app/
├── app.py                          # Main entry point
├── config/                         # Configuration
│   ├── colors.py                   # Thermo Fisher color scheme
│   ├── species.py                  # Species enum
│   └── workflows.py                # Workflow definitions
├── models/                         # Data models
│   └── proteomics_data.py          # Core data classes
├── pages/                          # Streamlit pages
│   ├── 1_🏠_Home.py
│   ├── 2_📊_Protein_Upload.py
│   ├── 3_🔬_Peptide_Upload.py
│   └── 4_✓_Data_Quality.py
├── utils/                          # Utilities
│   ├── file_handlers.py
│   ├── species_detector.py
│   ├── condition_detector.py
│   └── column_mapper.py
├── components/                     # UI components
│   ├── header.py
│   ├── charts.py
│   └── condition_selector.py
├── styles/                         # Custom CSS
│   └── thermo_fisher.css
└── demo_data/                      # Demo datasets
    └── test3_pg_matrix.csv
```

## Architecture

### Data Model

The app uses a **dataclass-based architecture** with separation of metadata and quantitative data:

```python
ProteomicsDataset
├── raw_df: Original uploaded data
├── metadata: Non-numeric columns
├── quant_data: Numeric columns only
├── species_map: Row-level species assignments
├── condition_mapping: Column renaming (A1, A2, B1, B2)
└── protein_groups: Full protein group strings
```

### Color Scheme

Consistent colors across all visualizations:

**Species Colors**:
- HUMAN: Navy (#262262)
- ECOLI: Purple (#8B4789)
- YEAST: Orange (#EA7600)

**Condition Colors**:
- Condition A: Navy (#262262)
- Condition B: Sky (#9BD3DD)

## Extending the App

### Adding New Pages

1. Create file in `pages/` folder:
```python
# pages/5_📈_New_Analysis.py
import streamlit as st
from components.header import render_header

render_header()
st.title("New Analysis Module")

# Access uploaded data
if st.session_state.get('protein_uploaded'):
    data = st.session_state.protein_data
    # Your analysis here
```

2. Page automatically appears in sidebar navigation

### Adding New Species

Edit `config/species.py`:
```python
class Species(Enum):
    HUMAN = "human"
    ECOLI = "ecoli"
    YEAST = "yeast"
    MOUSE = "mouse"  # Add new species
```

Update color mapping in `config/colors.py`

### Adding New Workflows

Edit `config/workflows.py`:
```python
class WorkflowType(Enum):
    LFQ_BENCH = "LFQ Bench"
    TMT_ANALYSIS = "TMT Analysis"  # Add new workflow
```

## Development

### Code Style
- Follow PEP 8
- Use type hints
- Docstrings for all functions/classes
- Modular, DRY (Don't Repeat Yourself) code

### Testing
```bash
# Test with demo data
streamlit run app.py

# Load demo data via checkbox in Protein Upload page
```

### Performance Considerations
- Large datasets (>20,000 rows) handled efficiently
- Quantitative data separated for optimized operations
- Session state management for data persistence

## Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: "File upload failed"
- Check file format (CSV, TSV, or Excel)
- Ensure numeric columns are readable
- Verify column headers are unique

### Issue: "Species not detected"
- Ensure protein names contain `_HUMAN`, `_ECOLI`, or `_YEAST` suffixes
- Manually select species column if auto-detection fails

### Issue: "Charts not displaying"
Check browser console for errors and verify Plotly is installed:
```bash
pip install plotly --upgrade
```

## Roadmap

### Version 1.1 (Next Release)
- [ ] Data Quality metrics (CV%, missing values)
- [ ] Intensity distribution plots
- [ ] PCA/sample clustering

### Version 1.2
- [ ] Statistical testing (t-test, ANOVA)
- [ ] Volcano plots
- [ ] Differential expression analysis

### Version 2.0
- [ ] TMT workflow support
- [ ] SILAC workflow support
- [ ] Automated report generation
- [ ] Export to PDF/Excel

## Support

For issues, questions, or feature requests, contact your Thermo Fisher representative.

## License

Proprietary - Thermo Fisher Scientific

© 2025 Thermo Fisher Scientific Inc. All rights reserved.

---

**Version**: 1.0.0  
**Last Updated**: November 23, 2025  
**Developed by**: Thermo Fisher Scientific Data Science Team
