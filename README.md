# ProteoFlow - Intelligent Proteomics Analysis

A Streamlit-based web application for analyzing proteomics data with AI-powered insights using Google's Gemini.

## Features

- 📤 **Data Upload**: Import CSV/TSV files from MaxQuant or FragPipe
- 📊 **Interactive Volcano Plot**: Visualize differential protein expression
- 📋 **Significance Filtering**: Adjustable p-value and fold-change thresholds
- 🤖 **AI-Powered Analysis**: Generate comprehensive biological reports using Gemini
- 💬 **Interactive Chat**: Ask questions about your data
- 📈 **Real-time Statistics**: Track upregulated, downregulated, and significant proteins

## Installation

### 1. Clone or download this repository

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Gemini API (Optional but recommended)

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

Set the environment variable:

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Or create a `.env` file:**
```
GEMINI_API_KEY=your-api-key-here
```

## Project Structure

```
proteomics-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── utils/
│   ├── __init__.py
│   ├── data_generator.py       # Mock data generation
│   └── analysis.py             # Data processing functions
├── components/
│   ├── __init__.py
│   ├── plots.py                # Volcano plot visualization
│   ├── tables.py               # Data table component
│   └── stats.py                # Statistics cards
└── services/
    ├── __init__.py
    └── gemini_service.py       # AI integration

```

## Usage

### 1. Run the application

```bash
streamlit run app.py
```

### 2. Upload your data or load demo dataset

- Click "Load Demo Dataset" to explore with sample data
- Or upload your own CSV/TSV file with columns: Gene, Fold Change, P-value

### 3. Explore the dashboard

- Adjust significance thresholds using the sidebar sliders
- Click on points in the volcano plot to view protein details
- View the sorted table of significant proteins

### 4. Generate AI insights

- Click "Generate Full Report" to get AI-powered biological interpretation
- Use the chat interface to ask specific questions about your data

## Data Format

Your input file should contain at minimum:

- **Gene**: Gene symbol or protein ID
- **Fold Change**: Numerical fold change values (or log2 fold change)
- **P-value**: Statistical significance values

Optional columns:
- Description
- Intensities (sample and control)
- Additional annotations

## Features Detail

### Volcano Plot
- Interactive visualization of differential expression
- Color-coded by significance (red=up, blue=down, gray=NS)
- Adjustable thresholds with visual guides
- Click-to-select functionality

### AI Analysis
- Automated pathway identification
- Functional categorization
- Biological interpretation
- Experimental recommendations

### Chat Interface
- Ask questions about specific proteins
- Query pathway involvement
- Get functional explanations
- Context-aware responses

## Configuration

### Default Thresholds
- P-value cutoff: 1.3 (-log10 scale, equivalent to p=0.05)
- Fold change cutoff: 1.0 (log2 scale, equivalent to 2-fold)

### Customization
Modify `app.py` to change:
- Color schemes
- Plot dimensions
- Statistical methods
- UI layout

## Troubleshooting

### Gemini API not working
- Ensure API key is correctly set
- Check internet connection
- Verify API key has appropriate permissions
- The app will work in demo mode without API key (limited features)

### Data upload issues
- Ensure file is CSV or TSV format
- Check column names match expected format
- Verify numeric columns contain valid numbers

### Performance issues
- Large datasets (>10,000 proteins) may be slow
- Consider filtering data before upload
- Use Chrome or Firefox for best performance

## Demo Mode

Without Gemini API configured, the app runs in demo mode:
- ✅ Full visualization and filtering capabilities
- ✅ Data table and statistics
- ⚠️ Limited AI analysis (template reports)
- ⚠️ Basic chat responses

## License

MIT License - feel free to use and modify

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Credits

- Built with [Streamlit](https://streamlit.io/)
- AI powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- Visualization with [Plotly](https://plotly.com/)
- Original React version inspiration

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on GitHub
