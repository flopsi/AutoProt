"""
Session management for data upload module.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid
import shutil
from typing import Optional

from .config import get_config


class SessionManager:
    """Manage upload sessions and temporary files"""
    
    def __init__(self):
        self.config = get_config()
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Ensure upload directory exists"""
        self.config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_or_create_session_id(self) -> str:
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def get_session_dir(self) -> Path:
        """Get session-specific directory"""
        session_id = self.get_or_create_session_id()
        session_dir = self.config.UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def save_uploaded_file(self, uploaded_file, filename: Optional[str] = None) -> Path:
        """Save uploaded file to session directory"""
        session_dir = self.get_session_dir()
        
        if filename is None:
            filename = uploaded_file.name
        
        file_path = session_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def cleanup_session(self):
        """Remove session directory and files"""
        session_dir = self.get_session_dir()
        if session_dir.exists():
            shutil.rmtree(session_dir)
    
    def list_session_files(self) -> list:
        """List files in current session"""
        session_dir = self.get_session_dir()
        return list(session_dir.glob('*'))


def get_session_manager() -> SessionManager:
    """Get session manager instance"""
    return SessionManager()
