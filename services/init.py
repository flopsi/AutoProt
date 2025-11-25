"""
Services package for external integrations
"""

from .gemini_service import analyze_proteins, chat_with_data

__all__ = [
    'analyze_proteins',
    'chat_with_data'
]
# Services packagefrom .gemini_service import analyze_proteins, chat_with_data
__all__ = ['analyze_proteins', 'chat_with_data']
