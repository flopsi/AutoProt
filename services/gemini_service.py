import os
from typing import List, Dict
import time
# Note: You'll need to install google-generativeai# pip install google-generativeaitry:
    import google.generativeai as genai
    GEMINI_AVAILABLE = Trueexcept ImportError:
    GEMINI_AVAILABLE = False    print("Warning: google-generativeai not installed. Using mock responses.")
def initialize_gemini():
    """Initialize Gemini API with API key from environment"""    if not GEMINI_AVAILABLE:
        return None    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Warning: No GEMINI_API_KEY found in environment. Using mock responses.")
        return None    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')
def analyze_proteins(proteins: List[Dict], context: str = "") -> str:
    """    Analyze significant proteins using Gemini AI    Args:        proteins: List of protein dictionaries with gene, log2FC, pValue, etc.        context: Experimental context description    Returns:        Markdown-formatted analysis report    """    model = initialize_gemini()
    # Prepare protein list for prompt    protein_summary = "\n".join([
        f"- **{p['gene']}** (Log2FC: {p['log2FoldChange']:.2f}, -Log10P: {p['negLog10PValue']:.2f})"        for p in proteins[:15]
    ])
    prompt = f"""You are a bioinformatics expert analyzing proteomics data.**Experimental Context:** {context if context else "Not provided"}**Top Significant Proteins:**{protein_summary}Please provide a comprehensive biological interpretation including:1. **Overview**: Summarize the main findings2. **Key Pathways**: Identify major biological pathways affected3. **Functional Categories**: Group proteins by function4. **Biological Significance**: Discuss the implications5. **Recommendations**: Suggest follow-up experimentsFormat your response in clear markdown with headers and bullet points."""    if model:
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return generate_mock_report(proteins, context)
    else:
        return generate_mock_report(proteins, context)
def chat_with_data(history: List[Dict], query: str, context: str) -> str:
    """    Interactive chat about proteomics data    Args:        history: Previous chat messages        query: Current user question        context: Data context (stats, selected protein, etc.)    Returns:        AI response text    """    model = initialize_gemini()
    # Build conversation history    conversation = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['text']}"        for msg in history
    ])
    prompt = f"""You are a helpful assistant analyzing proteomics data.**Data Context:** {context}**Conversation History:**{conversation if conversation else "No previous messages"}**User Question:** {query}Provide a clear, concise answer based on the data context. If you don't have enough information, say so."""    if model:
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return generate_mock_chat_response(query)
    else:
        return generate_mock_chat_response(query)
def generate_mock_report(proteins: List[Dict], context: str) -> str:
    """Generate a mock analysis report when Gemini is unavailable"""    top_up = [p for p in proteins if p.get('significance') == 'UP'][:5]
    top_down = [p for p in proteins if p.get('significance') == 'DOWN'][:5]
    report = f"""# Proteomics Analysis Report## OverviewThis analysis examines differential protein expression from the following experiment:**Context:** {context if context else "No experimental context provided"}A total of **{len(proteins)} significant proteins** were identified based on the statistical thresholds applied.## Key Findings### Upregulated ProteinsThe following proteins show significant upregulation:"""    for p in top_up:
        report += f"- **{p['gene']}** (Log2FC: {p['log2FoldChange']:.2f}, -Log10P: {p['negLog10PValue']:.2f})\n"    report += """\n### Downregulated ProteinsThe following proteins show significant downregulation:"""    for p in top_down:
        report += f"- **{p['gene']}** (Log2FC: {p['log2FoldChange']:.2f}, -Log10P: {p['negLog10PValue']:.2f})\n"    report += """## Biological PathwaysBased on the identified proteins, several key biological pathways appear to be affected:1. **Metabolic Regulation**: Multiple proteins involved in cellular metabolism show altered expression2. **Signal Transduction**: Key signaling molecules demonstrate significant changes3. **Cell Cycle Control**: Regulatory proteins affecting cell division are differentially expressed## Recommendations1. **Validation**: Confirm findings using Western blot or qPCR for key proteins2. **Pathway Analysis**: Conduct detailed pathway enrichment analysis (e.g., KEGG, GO)3. **Functional Studies**: Perform functional assays to validate biological impact4. **Time Course**: Consider time-series experiments to understand temporal dynamics## Notes⚠️ This is a **demonstration report**. For production use, configure the Gemini API key for AI-powered analysis.---*Report generated with ProteoFlow v1.0*"""    return report
def generate_mock_chat_response(query: str) -> str:
    """Generate a mock chat response when Gemini is unavailable"""    query_lower = query.lower()
    if 'function' in query_lower or 'what is' in query_lower or 'what does' in query_lower:
        return "I can help you understand protein functions! However, for detailed biological insights, please configure the Gemini API key. In the meantime, I can tell you that the selected protein plays important roles in cellular processes."    elif 'mitochondrial' in query_lower or 'mitochondria' in query_lower:
        return "To identify mitochondrial proteins in your dataset, I would need to analyze the gene ontology annotations. With the Gemini API configured, I can provide detailed analysis of subcellular localization."    elif 'pathway' in query_lower:
        return "Pathway analysis requires integration with biological databases. Configure the Gemini API key to get AI-powered pathway insights based on your significant proteins."    else:
        return f"I received your question: '{query}'. To provide detailed biological insights, please configure the Gemini API key in your environment (GEMINI_API_KEY or GOOGLE_API_KEY). For now, I'm running in demo mode with limited capabilities."
