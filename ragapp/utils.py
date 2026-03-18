import os
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_file(path):
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    
    try:
        text_extensions = [".txt", ".py", ".js", ".html", ".css", ".json", ".md"]
        if ext in text_extensions:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        
        if ext == ".pdf":
            reader = PdfReader(path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
            
        if ext in [".docx", ".doc"]:
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])

    except Exception as e:
        print(f"Error: {e}")
        return ""

    return ""
