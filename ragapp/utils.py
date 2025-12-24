# ragapp/utils.py
from PyPDF2 import PdfReader
from docx import Document
import os

def extract_text_from_file(path):
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".txt", ".py", ".js", ".html", ".css", ".json", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext == ".pdf":
            reader = PdfReader(path)
            text = []
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
        if ext in [".docx", ".doc"]:
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return ""
    return ""
