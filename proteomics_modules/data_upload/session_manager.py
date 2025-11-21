"""
Session management for proteomics data upload.
Handles file storage, session IDs, and cleanup.
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import streamlit as st

from .config import get_config


class SessionManager:
    """Manages user sessions and file storage"""
    
    def __init__(self):
        self.config = get_config()
        self.base_dir = Path(self.config.TEMP_DATA_DIR)
        self._ensure_base_dir()
    
    def _ensure_base_dir(self):
        """Create base directory if it doesn't exist"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_or_create_session_id(self) -> str:
        """Get existing session ID or create new one"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.session_created = datetime.now()
        return st.session_state.session_id
    
    def get_session_dir(self, session_id: Optional[str] = None) -> Path:
        """Get session directory path"""
        if session_id is None:
            session_id = self.get_or_create_session_id()
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def get_upload_dir(self, session_id: Optional[str] = None) -> Path:
        """Get upload directory for raw uploaded files"""
        upload_dir = self.get_session_dir(session_id) / 'uploaded'
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir
    
    def get_processed_dir(self, session_id: Optional[str] = None) -> Path:
        """Get directory for processed files"""
        processed_dir = self.get_session_dir(session_id) / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir
    
    def get_results_dir(self, session_id: Optional[str] = None) -> Path:
        """Get directory for analysis results"""
        results_dir = self.get_session_dir(session_id) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def save_uploaded_file(self, uploaded_file, filename: Optional[str] = None) -> Path:
        """
        Save uploaded file to session directory
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            filename: Optional custom filename (uses original name if None)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = uploaded_file.name
        
        upload_dir = self.get_upload_dir()
        file_path = upload_dir / filename
        
        # Write file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Store in session state
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
        
        st.session_state.uploaded_files[filename] = {
            'path': str(file_path),
            'size': uploaded_file.size,
            'timestamp': datetime.now().isoformat(),
            'type': uploaded_file.type
        }
        
        return file_path
    
    def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get information about uploaded file"""
        if 'uploaded_files' in st.session_state:
            return st.session_state.uploaded_files.get(filename)
        return None
    
    def list_uploaded_files(self) -> list:
        """List all uploaded files in current session"""
        upload_dir = self.get_upload_dir()
        if upload_dir.exists():
            return list(upload_dir.glob('*'))
        return []
    
    def clear_session(self, session_id: Optional[str] = None):
        """Clear all files for a session"""
        if session_id is None:
            session_id = self.get_or_create_session_id()
        
        session_dir = self.base_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        # Clear session state
        if 'uploaded_files' in st.session_state:
            del st.session_state.uploaded_files
    
    def cleanup_old_sessions(self):
        """Remove sessions older than timeout period"""
        if not self.config.AUTO_CLEANUP:
            return
        
        timeout = timedelta(hours=self.config.SESSION_TIMEOUT_HOURS)
        cutoff_time = datetime.now() - timeout
        
        if not self.base_dir.exists():
            return
        
        for session_dir in self.base_dir.iterdir():
            if session_dir.is_dir():
                # Check modification time
                mod_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
                if mod_time < cutoff_time:
                    try:
                        shutil.rmtree(session_dir)
                    except Exception as e:
                        # Log but don't fail
                        print(f"Failed to cleanup session {session_dir.name}: {e}")
    
    def get_session_size(self, session_id: Optional[str] = None) -> int:
        """Get total size of session directory in bytes"""
        session_dir = self.get_session_dir(session_id)
        total_size = 0
        
        for path in session_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


# Singleton instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get or create session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
