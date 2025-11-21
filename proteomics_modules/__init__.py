"""
Proteomics Modules Package
Modular proteomics data analysis toolkit
"""

__version__ = "0.1.0"
__author__ = "Florian Marty"

# Lazy import to avoid circular dependency
def get_upload_module():
    from .data_upload import DataUploadModule, run_upload_module
    return DataUploadModule, run_upload_module

# For backward compatibility
try:
    from .data_upload import DataUploadModule, run_upload_module
    __all__ = ['DataUploadModule', 'run_upload_module']
except ImportError:
    __all__ = []
