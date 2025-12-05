"""
helpers/audit.py
Audit trail logging for data manipulation tracking
Records all operations: uploads, filters, transformations, analyses
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from helpers.dataclasses import AuditEvent, AuditTrail

# ============================================================================
# AUDIT FILE MANAGEMENT
# ============================================================================

AUDIT_LOG_PATH = "data/audit.log"


def ensure_audit_dir() -> None:
    """Create data directory if it doesn't exist."""
    os.makedirs("data", exist_ok=True)


def log_event(page: str, action: str, details: dict = None):
    """Log an event to the audit trail."""    """
    Log an event to the audit trail.
    
    Args:
        page: Page name where event occurred
        action: Human-readable action description
        details: Additional metadata (dict)
        
    Example:
        log_event("Data Upload", "Uploaded protein data", {"n_proteins": 5000, "file": "data.csv"})
        log_event("Filtering", "Removed low CV proteins", {"removed": 200, "threshold": 0.5})
        log_event("Analysis", "Ran t-test", {"n_significant": 150, "fc_threshold": 1.0})
    """
    ensure_audit_dir()
    
    event = AuditEvent(
        timestamp=datetime.now().isoformat(),
        page=page,
        action=action,
        details=details or {},
    )
    
    # Append to JSON log file
    try:
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    except Exception as e:
        print(f"Failed to log event: {e}")


def read_audit_log() -> List[Dict]:
    """
    Read all events from audit log file.
    
    Returns:
        List of event dictionaries
    """
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    
    events = []
    try:
        with open(AUDIT_LOG_PATH, "r") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    except Exception as e:
        print(f"Failed to read audit log: {e}")
    
    return events


def clear_audit_log() -> None:
    """Clear the audit log file."""
    ensure_audit_dir()
    try:
        open(AUDIT_LOG_PATH, "w").close()
    except Exception as e:
        print(f"Failed to clear audit log: {e}")

    # Convert numpy types to Python types for JSON serialization
    if details:
        details_cleaned = {}
        for key, value in details.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                details_cleaned[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                details_cleaned[key] = float(value)
            elif isinstance(value, np.bool_):
                details_cleaned[key] = bool(value)
            elif isinstance(value, np.ndarray):
                details_cleaned[key] = value.tolist()
            else:
                details_cleaned[key] = value
        details = details_cleaned


# ============================================================================
# AUDIT SUMMARY & STATISTICS
# ============================================================================

def get_audit_summary() -> Dict:
    """
    Generate summary statistics from audit log.
    
    Returns:
        Dict with counts by page, timeline, etc.
    """
    events = read_audit_log()
    
    if not events:
        return {
            "total_events": 0,
            "pages": {},
            "first_event": None,
            "last_event": None,
        }
    
    # Count by page
    page_counts = {}
    for event in events:
        page = event.get("page", "Unknown")
        page_counts[page] = page_counts.get(page, 0) + 1
    
    # Timeline
    first_time = events[0].get("timestamp", "")
    last_time = events[-1].get("timestamp", "")
    
    return {
        "total_events": len(events),
        "pages": page_counts,
        "first_event": first_time,
        "last_event": last_time,
        "session_duration": calculate_duration(first_time, last_time),
    }


def calculate_duration(start_iso: str, end_iso: str) -> str:
    """
    Calculate duration between two ISO timestamps.
    
    Args:
        start_iso: ISO format timestamp
        end_iso: ISO format timestamp
        
    Returns:
        Human-readable duration string
    """
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        duration = end - start
        
        minutes = duration.total_seconds() / 60
        if minutes < 1:
            return "< 1 min"
        elif minutes < 60:
            return f"{int(minutes)} min"
        else:
            hours = minutes / 60
            return f"{hours:.1f} hr"
    except Exception:
        return "Unknown"


# ============================================================================
# AUDIT TRAIL HISTORY
# ============================================================================

def format_audit_trail_for_display() -> List[Dict]:
    """
    Format audit log for human-readable display.
    
    Returns:
        List of formatted event dicts suitable for st.dataframe()
    """
    events = read_audit_log()
    
    formatted = []
    for event in events:
        # Parse timestamp
        ts = event.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = ts
        
        # Format details as string
        details = event.get("details", {})
        details_str = ", ".join(f"{k}={v}" for k, v in details.items()) if details else "-"
        
        formatted.append({
            "Time": time_str,
            "Page": event.get("page", ""),
            "Action": event.get("action", ""),
            "Details": details_str,
        })
    
    return formatted


# ============================================================================
# COMMON LOGGING PATTERNS (Copy-paste into pages)
# ============================================================================

"""
# === PATTERN 1: Log data upload ===
from helpers.audit import log_event

log_event(
    "Data Upload",
    f"Uploaded {file.name}",
    {
        "filename": file.name,
        "n_proteins": len(df),
        "n_samples": len(numeric_cols),
        "file_size_mb": file.size / 1e6,
    }
)

# === PATTERN 2: Log filtering ===
log_event(
    "Preprocessing",
    "Removed proteins by CV threshold",
    {
        "removed_count": n_removed,
        "remaining_count": len(filtered_df),
        "threshold": cv_threshold,
        "filter_rate_pct": (n_removed / n_total * 100),
    }
)

# === PATTERN 3: Log transformation ===
log_event(
    "Preprocessing",
    f"Applied {transform_key} transformation",
    {
        "transform": transform_key,
        "proteins": len(transformed_df),
        "samples": len(numeric_cols),
    }
)

# === PATTERN 4: Log analysis ===
log_event(
    "Analysis",
    "Ran differential expression analysis",
    {
        "n_proteins": len(results_df),
        "n_significant_up": n_up,
        "n_significant_down": n_down,
        "fc_threshold": fc_threshold,
        "pval_threshold": pval_threshold,
    }
)

# === PATTERN 5: Log error metrics ===
log_event(
    "Analysis",
    "Calculated error metrics",
    {
        "rmse": f"{rmse:.3f}",
        "mae": f"{mae:.3f}",
        "sensitivity": f"{sensitivity:.2%}",
        "specificity": f"{specificity:.2%}",
    }
)
"""
