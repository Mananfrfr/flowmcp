
from flowmcp.rag.chunker import chunk_text


def test_markdown_chunking_with_headings():
    text = """# Introduction
This is intro.

## Installation
Install the package.

### Usage
Run the application.
"""

    chunks = chunk_text(text, strategy="markdown")

    assert len(chunks) == 3
    assert chunks[0].startswith("# Introduction")
    assert chunks[1].startswith("## Installation")
    assert chunks[2].startswith("### Usage")


def test_markdown_fallback_without_headings():
    text = " ".join(["word"] * 600)

    chunks = chunk_text(
        text,
        chunk_size=500,
        overlap=50,
        strategy="markdown",
    )

    assert len(chunks) == 2


def test_markdown_chunking_with_mixed_content():
    text = """Introduction before heading.

# First Section
First section content.

## Second Section
Second section content.
"""

    chunks = chunk_text(text, strategy="markdown")

    assert len(chunks) == 3
    assert chunks[0] == "Introduction before heading."
    assert chunks[1].startswith("# First Section")
    assert chunks[2].startswith("## Second Section")


def test_rag_uses_markdown_chunk_strategy():
    from flowmcp.rag.loader import RAG

    text = """# Introduction
Introduction content.

## Installation
Installation content.
"""

    def fake_embedder(texts):
        return [[1.0, 0.0] for _ in texts]

    rag = RAG(
        text,
        embedder=fake_embedder,
        chunk_strategy="markdown",
    )

    chunks = rag.retriever.chunks

    assert len(chunks) == 2
    assert chunks[0].startswith("# Introduction")
    assert chunks[1].startswith("## Installation")
