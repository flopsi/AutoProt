"""
helpers/audit.py

Robust audit logging with JSON serialization safety
Tracks user actions and data transformations for reproducibility
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import streamlit as st
from typing import Dict, Any, Optional

# ============================================================================
# AUDIT CONFIGURATION
# File paths and limits for log management
# ============================================================================

AUDIT_DIR = Path("audit_logs")
AUDIT_FILE = AUDIT_DIR / "audit_log.jsonl"
MAX_LOG_SIZE_MB = 10  # Rotate when file exceeds this size

# ============================================================================
# UTILITY FUNCTIONS
# JSON serialization and file management
# ============================================================================

def _to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy/pandas types to JSON-serializable Python types.
    Handles NaN, Int64, float64, arrays, DataFrames, etc.
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON-serializable version
    """
    import numpy as np
    import pandas as pd
    
    if pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj_dict = obj.to_dict()
        if len(str(obj_dict)) > 500:
            obj_dict = {"truncated": f"Size: {len(obj)} items"}
        return obj_dict
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj

def _ensure_audit_dir():
    """Create audit directory if it doesn't exist."""
    AUDIT_DIR.mkdir(exist_ok=True)

def _rotate_log_if_needed():
    """Rotate log file if it exceeds max size."""
    if AUDIT_FILE.exists() and AUDIT_FILE.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
        backup_file = AUDIT_FILE.with_suffix(f".backup.{int(time.time())}")
        AUDIT_FILE.rename(backup_file)
        st.info(f"🔄 Audit log rotated: {backup_file.name}")

# ============================================================================
# MAIN AUDIT FUNCTIONS
# Event logging and retrieval
# ============================================================================

def log_event(
    page: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Log an audit event to JSONL file.
    
    Args:
        page: Page where event occurred (e.g., "1_Data_Upload")
        action: Action performed (e.g., "file_uploaded", "transform_selected")
        details: Additional event details
        user_id: User identifier
        session_id: Streamlit session ID
    """
    try:
        _ensure_audit_dir()
        _rotate_log_if_needed()
        
        # Prepare event data
        event = {
            "timestamp": datetime.now().isoformat(),
            "page": page,
            "action": action,
            "user_id": user_id,
            "session_id": session_id or st.session_state.get("session_id", "unknown"),
            "details": details or {}
        }
        
        # Convert all values to JSON-serializable types
        event_safe = {
            k: _to_json_serializable(v)
            for k, v in event.items()
        }
        
        # Write to JSONL file (one JSON object per line)
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_safe) + "\n")
    
    except Exception as e:
        # Silent fail - don't break the app
        print(f"Audit logging failed: {e}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# Pre-configured event loggers for common actions
# ============================================================================

def log_file_upload(filename: str, file_size: int, n_rows: int, n_cols: int, numeric_cols: int):
    """Log file upload event."""
    details = {
        "filename": filename,
        "file_size_bytes": int(file_size),
        "rows": int(n_rows),
        "columns": int(n_cols),
        "numeric_columns": int(numeric_cols)
    }
    log_event("1_Data_Upload", "file_uploaded", details)

def log_transformation_selected(transform_name: str, shapiro_w: float):
    """Log transformation selection."""
    details = {
        "transform": transform_name,
        "shapiro_w": float(shapiro_w)
    }
    log_event("2_Visual_EDA", "transformation_selected", details)

def log_species_filter(species_selected: list, proteins_filtered: int):
    """Log species filtering."""
    details = {
        "species": species_selected,
        "proteins_after_filter": int(proteins_filtered)
    }
    log_event("2_Visual_EDA", "species_filtered", details)

# ============================================================================
# AUTOMATIC SESSION TRACKING
# Initialize session-level logging
# ============================================================================

def init_audit_session():
    """Initialize audit session tracking."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
        log_event(
            page="app_start",
            action="session_started",
            session_id=st.session_state.session_id
        )
