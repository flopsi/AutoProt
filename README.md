# DIA Proteomics Analysis App

A modular Streamlit app for automated proteomics data import, annotation, and quality control visualization – with Thermo Fisher Scientific branding.

## 🚀 Features

- **Automated Column Detection**
Metadata columns and quantitative (intensity) columns detected by type.
- **Name Trimming**
Quantitative columns automatically cleaned of prefixes/suffixes.
- **Automatic Condition Assignment**
Even split assigns half Control, half Treatment; can override in app.
- **Interactive Annotation**
Modify assignments easily and propagate globally.
- **Modern Visualizations**
Intensity distribution, Coefficient of Variation (CV), and PCA sample clustering.
- **Modular Architecture**
Codebase structured for easy addition of new statistics and plots.

## 🎨 Color Scheme (Thermo Fisher Style)

<aside>
💡

Primary Red: #E71316
Dark Red: #A6192E
Sky Blue: #9BD3DD
Primary Gray: #54585A
Light Gray: #E2E3E4
Green: #B5BD00
Orange: #EA7600

</aside>

## 🏗️ Folder Structure

<aside>
💡

├── [app.py](http://app.py/) # Main Streamlit entrypoint
├── [config.py](http://config.py/) # Utilities: column detection, trimming, assignment, color
├── requirements.txt # Environment for deployment
└── ... # Additional modules easily added

</aside>

## ➕ How to Use

1. **Clone the repo:**
git clone [https://github.com/yourusername/dia-proteomics-app.git](https://github.com/yourusername/dia-proteomics-app.git)
cd dia-proteomics-app
2. **Install dependencies:**
pip install -r requirements.txt
3. **Run the app:**
streamlit run [app.py](http://app.py/)
4. **Upload** a CSV/TSV data matrix.
    
    Columns are automatically split and annotated.
    
5. **Review visualizations.**
    
    Interactive plot tabs update automatically as data/annotation change.
    

## 📝 Module Expansion

To add new stats/plots, simply add a new Python file (e.g., `modules/stats_newplot.py`) and connect in `app.py`. All utilities and styling are global.

## 🛡️ License

Proprietary & Confidential | For Internal Use Only

© 2025 Thermo Fisher Scientific Inc. All rights reserved.

---

**Support:**

- See code comments
- Open issues for technical help or feature requests