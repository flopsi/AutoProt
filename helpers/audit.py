"""
helpers/audit.py
Robust audit logging with JSON serialization safety
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
# ============================================================================

AUDIT_DIR = Path("audit_logs")
AUDIT_FILE = AUDIT_DIR / "audit_log.jsonl"
MAX_LOG_SIZE_MB = 10  # Rotate when file exceeds this size

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy/pandas types to JSON-serializable Python types.
    Handles NaN, Int64, float64, arrays, DataFrames, etc.
    """
    import numpy as np
    import pandas as pd
    
    if pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
        return None
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(obj):
            return None
        return float(obj)
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        # Convert to dict and truncate long content
        obj_dict = obj.to_dict()
        # Limit dict size for logging
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
    
    Parameters:
    -----------
    page : str
        Page where event occurred (e.g., "1_Data_Upload")
    action : str
        Action performed (e.g., "file_uploaded", "transform_selected")
    details : dict, optional
        Additional event details
    user_id : str, optional
        User identifier
    session_id : str, optional
        Streamlit session ID
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


def log_file_upload(
    filename: str,
    file_size: int,
    n_rows: int,
    n_cols: int,
    numeric_cols: int
):
    """Log file upload event."""
    details = {
        "filename": filename,
        "file_size_bytes": int(file_size),
        "rows": int(n_rows),
        "columns": int(n_cols),
        "numeric_columns": int(numeric_cols)
    }
    log_event("1_Data_Upload", "file_uploaded", details)


def log_transformation_selected(
    transform_name: str,
    shapiro_w: float
):
    """Log transformation selection."""
    details = {
        "transform": transform_name,
        "shapiro_w": float(shapiro_w)
    }
    log_event("2_Visual_EDA", "transformation_selected", details)


def log_species_filter(
    species_selected: list,
    proteins_filtered: int
):
    """Log species filtering."""
    details = {
        "species": species_selected,
        "proteins_after_filter": int(proteins_filtered)
    }
    log_event("2_Visual_EDA", "species_filtered", details)


def get_recent_events(limit: int = 50) -> list[Dict[str, Any]]:
    """
    Get recent audit events for display.
    
    Parameters:
    -----------
    limit : int
        Maximum number of events to return
    
    Returns:
    --------
    list of event dictionaries
    """
    try:
        if not AUDIT_FILE.exists():
            return []
        
        events = []
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if line_num >= limit:
                    break
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        return events[::-1]  # Return most recent first
        
    except Exception:
        return []


def display_audit_log():
    """Display recent audit events in Streamlit."""
    st.subheader("📋 Recent Activity")
    
    events = get_recent_events(20)
    
    if not events:
        st.info("No audit events recorded yet")
        return
    
    # Create summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Events", len(events))
    with col2:
        uploads = sum(1 for e in events if e.get("action") == "file_uploaded")
        st.metric("File Uploads", uploads)
    with col3:
        transforms = sum(1 for e in events if "transformation" in e.get("action", ""))
        st.metric("Transformations", transforms)
    
    # Show event table
    df_events = pd.DataFrame(events)
    if not df_events.empty:
        # Format columns for display
        display_cols = ["timestamp", "page", "action", "user_id"]
        if "details" in df_events.columns:
            display_cols.append("details")
        
        # Truncate long details
        if "details" in df_events.columns:
            df_events["details"] = df_events["details"].apply(
                lambda x: {k: str(v)[:50] + "..." if len(str(v)) > 50 else v 
                          for k, v in (x or {}).items()}
            )
        
        st.dataframe(
            df_events[display_cols].tail(10),
            width="stretch",
            height=300
        )


# ============================================================================
# AUTOMATIC SESSION TRACKING
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
