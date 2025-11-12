import re
from typing import List

def clean_text(text: str) -> str:
    """Cleans the given text by removing unnecessary spaces and line breaks."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_text_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Splits text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks
