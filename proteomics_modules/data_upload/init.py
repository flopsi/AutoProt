"""
Data Upload Module
Handles file upload, validation, column detection, and species annotation.
"""

from .module import DataUploadModule, run_upload_module
from .config import get_config, DataUploadConfig
from .column_detector import get_column_detector, get_species_manager

__all__ = [
    'DataUploadModule',
    'run_upload_module',
    'get_config',
    'DataUploadConfig',
    'get_column_detector',
    'get_species_manager'
]

__version__ = '1.0.0'
