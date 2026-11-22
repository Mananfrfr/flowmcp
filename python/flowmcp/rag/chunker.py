
import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: str = "fixed",
) -> List[str]:
    """Split text using fixed-size or Markdown-aware chunking."""

    if strategy not in ("fixed", "markdown"):
        raise ValueError("strategy must be either 'fixed' or 'markdown'")

    if strategy == "markdown":
        heading_pattern = re.compile(r"^#{1,6}(?:\s|$)", re.MULTILINE)

        if heading_pattern.search(text):
            sections = re.split(r"(?=^#{1,6}(?:\s|$))", text, flags=re.MULTILINE)
            return [section.strip() for section in sections if section.strip()]

    # Fixed-size chunking (default and Markdown fallback)
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks
