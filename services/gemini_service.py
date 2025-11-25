"""
Gemini AI service for proteomics analysis insights
"""

import google.generativeai as genai
import os
import pandas as pd
from typing import Optional


def configure_gemini():
    """Configure Gemini API with key from environment"""
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False


def analyze_proteins(df: pd.DataFrame, condition_a: str = "Condition A", 
                    condition_b: str = "Condition B") -> Optional[str]:
    """
    Generate AI-powered analysis summary of proteomics results
    
    Args:
        df: DataFrame with protein analysis results
        condition_a: Name of first condition
        condition_b: Name of second condition
        
    Returns:
        Analysis text or None if API not configured
    """
    if not configure_gemini():
        return None
    
    try:
        # Prepare summary statistics
        total = len(df)
        up = (df['significance'] == 'UP').sum() if 'significance' in df.columns else 0
        down = (df['significance'] == 'DOWN').sum() if 'significance' in df.columns else 0
        
        # Get top proteins
        if 'significance' in df.columns and 'log2FoldChange' in df.columns:
            top_up = df[df['significance'] == 'UP'].nlargest(5, 'log2FoldChange')
            top_down = df[df['significance'] == 'DOWN'].nsmallest(5, 'log2FoldChange')
        else:
            top_up = pd.DataFrame()
            top_down = pd.DataFrame()
        
        # Create prompt
        prompt = f"""
        Analyze the following proteomics experiment results comparing {condition_a} vs {condition_b}:

        Summary Statistics:
        - Total proteins analyzed: {total}
        - Significantly upregulated: {up}
        - Significantly downregulated: {down}

        Top upregulated proteins:
        {top_up[['gene', 'log2FoldChange', 'pValue']].to_string() if not top_up.empty else 'None'}

        Top downregulated proteins:
        {top_down[['gene', 'log2FoldChange', 'pValue']].to_string() if not top_down.empty else 'None'}

        Provide:
        1. Brief interpretation of the overall results
        2. Key biological pathways that might be affected
        3. Suggestions for follow-up experiments
        
        Keep the response concise and scientific (max 300 words).
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def chat_with_data(df: pd.DataFrame, user_question: str) -> Optional[str]:
    """
    Answer user questions about proteomics data using Gemini
    
    Args:
        df: DataFrame with protein data
        user_question: User's question
        
    Returns:
        Answer text or None if API not configured
    """
    if not configure_gemini():
        return None
    
    try:
        # Prepare data summary
        summary = f"""
        Dataset contains {len(df)} proteins.
        Columns: {', '.join(df.columns.tolist())}
        
        Sample data (first 10 rows):
        {df.head(10).to_string()}
        
        Summary statistics:
        {df.describe().to_string()}
        """
        
        prompt = f"""
        Based on this proteomics dataset:
        
        {summary}
        
        User question: {user_question}
        
        Provide a clear, scientific answer. If the question cannot be answered from the data, 
        explain what additional information would be needed.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error processing question: {str(e)}"
