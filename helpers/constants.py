"""
helpers/constants.py
Color schemes, themes, and global constants for AutoProt
Single source of truth for all visual styling
"""

from typing import Dict

# ============================================================================
# THEME DEFINITIONS
# ============================================================================

# Light theme (default)
THEME_LIGHT = {
    "name": "Light",
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8f9fa",
    "text_primary": "#1a1a1a",
    "text_secondary": "#54585A",
    "color_human": "#199d76",      # Green - from company guide
    "color_yeast": "#d85f02",      # Orange - from company guide
    "color_ecoli": "#7570b2",      # Purple - from company guide
    "color_up": "#d85f02",         # Orange for upregulated
    "color_down": "#199d76",       # Green for downregulated
    "color_ns": "#cccccc",         # Gray for not significant
    "color_nt": "#999999",         # Dark gray for not tested
    "grid": "#e0e0e0",
    "border": "#d0d0d0",
    "accent": "#199d76",
    "paper_bg": "rgba(0,0,0,0)",
}

# Dark theme
THEME_DARK = {
    "name": "Dark",
    "bg_primary": "#1a1a1a",
    "bg_secondary": "#2d2d2d",
    "text_primary": "#ffffff",
    "text_secondary": "#cccccc",
    "color_human": "#4ccc9f",      # Lighter green
    "color_yeast": "#ff9933",      # Lighter orange
    "color_ecoli": "#9e94d4",      # Lighter purple
    "color_up": "#ff9933",
    "color_down": "#4ccc9f",
    "color_ns": "#666666",
    "color_nt": "#444444",
    "grid": "#404040",
    "border": "#505050",
    "accent": "#4ccc9f",
    "paper_bg": "rgba(10,10,10,0.8)",
}

# Colorblind-friendly theme (deuteranopia - red-green colorblind)
THEME_COLORBLIND = {
    "name": "Colorblind-Friendly",
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8f9fa",
    "text_primary": "#1a1a1a",
    "text_secondary": "#54585A",
    "color_human": "#0173b2",      # Blue
    "color_yeast": "#cc78bc",      # Magenta
    "color_ecoli": "#ca9161",      # Brown
    "color_up": "#cc78bc",         # Magenta for up
    "color_down": "#0173b2",       # Blue for down
    "color_ns": "#cccccc",
    "color_nt": "#999999",
    "grid": "#e0e0e0",
    "border": "#d0d0d0",
    "accent": "#0173b2",
    "paper_bg": "rgba(0,0,0,0)",
}

# Journal submission theme (black/white/gray only)
THEME_JOURNAL = {
    "name": "Journal (B&W)",
    "bg_primary": "#ffffff",
    "bg_secondary": "#f5f5f5",
    "text_primary": "#000000",
    "text_secondary": "#333333",
    "color_human": "#404040",      # Dark gray
    "color_yeast": "#000000",      # Black
    "color_ecoli": "#808080",      # Medium gray
    "color_up": "#000000",
    "color_down": "#404040",
    "color_ns": "#c0c0c0",
    "color_nt": "#e0e0e0",
    "grid": "#d0d0d0",
    "border": "#a0a0a0",
    "accent": "#000000",
    "paper_bg": "rgba(0,0,0,0)",
}

# Map theme names to configs
THEMES: Dict[str, Dict] = {
    "light": THEME_LIGHT,
    "dark": THEME_DARK,
    "colorblind": THEME_COLORBLIND,
    "journal": THEME_JOURNAL,
}

# ============================================================================
# ANALYSIS CONSTANTS
# ============================================================================

# Default statistical thresholds
DEFAULT_FC_THRESHOLD = 1.0         # log2FC = 1.0 equals 2-fold change
DEFAULT_PVAL_THRESHOLD = 0.05      # p-value cutoff
DEFAULT_MIN_VALID = 2              # Min values per group for t-test

# Species to track
SPECIES_LIST = ["HUMAN", "YEAST", "ECOLI", "MOUSE"]

# Transform options
# In constants.py, add new transforms:
# In helpers/constants.py, update TRANSFORMS list:

TRANSFORMS = [
    "log2", 
    "log10", 
    "sqrt", 
    "cbrt", 
    "yeo_johnson", 
    "quantile",
    "robust",      # NEW
    "zscore",      # NEW
    "minmax"       # NEW
]



# ============================================================================
# UI CONSTANTS
# ============================================================================

# Font configurations
FONT_FAMILY = "Arial"
FONT_SIZE_BASE = 14
FONT_SIZE_TITLE = 16
FONT_SIZE_SMALL = 12

# Plot dimensions
PLOT_HEIGHT_DEFAULT = 600
PLOT_HEIGHT_SMALL = 400
PLOT_WIDTH_DEFAULT = 900

# ============================================================================
# FILE I/O CONSTANTS
# ============================================================================

SUPPORTED_FORMATS = ["csv", "tsv", "txt", "xlsx"]
MAX_FILE_SIZE_MB = 100

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_theme(theme_name: str = "light") -> Dict:
    """
    Get theme configuration by name.
    
    Args:
        theme_name: "light", "dark", "colorblind", or "journal"
        
    Returns:
        Dictionary with theme colors and settings
    """
    return THEMES.get(theme_name.lower(), THEME_LIGHT)

def get_theme_names() -> list:
    """Get list of available theme names."""
    return list(THEMES.keys())
