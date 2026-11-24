
from app.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    """Empty input should produce an empty list."""

    chunks = chunk_text("")

    assert chunks == []


def test_chunker_preserves_document_content():
    """Important document information should remain in the chunks."""

    text = (
        "The company uses PostgreSQL for its primary database.\n\n"
        "Employees receive 25 annual leave days."
    )

    chunks = chunk_text(text, chunk_size=500, overlap=50)

    combined_text = " ".join(chunks)

    assert len(chunks) > 0
    assert "PostgreSQL" in combined_text
    assert "25 annual leave days" in combined_text


def test_long_text_is_split_into_chunks():
    """Long paragraphs should be divided into smaller chunks."""

    text = " ".join(["enterprise"] * 200)

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunking_is_deterministic():
    """The same input should produce the same chunks."""

    text = (
        "Introduction\n"
        "This is a document about artificial intelligence."
    )

    first_result = chunk_text(text)
    second_result = chunk_text(text)

    assert first_result == second_result


def test_section_heading_is_preserved():
    """Section headings should remain attached to their content."""

    text = (
        "Technical Skills\n"
        "Python, SQL, PyTorch, FastAPI and Docker."
    )

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].startswith("Technical Skills\n")


def test_section_heading_counts_toward_chunk_size():
    """The heading must be included in the maximum chunk size."""

    text = (
        "Technical Skills\n"
        + " ".join(["Python"] * 100)
    )

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(
        chunk.startswith("Technical Skills\n")
        for chunk in chunks
    )
