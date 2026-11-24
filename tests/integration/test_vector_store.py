
import uuid

import chromadb
import numpy as np
import pytest

from app import vector_store


@pytest.fixture
def isolated_collection(monkeypatch):
    """Provide a temporary ChromaDB collection for each test."""

    test_client = chromadb.EphemeralClient()
    collection_name = f"test_{uuid.uuid4().hex}"

    test_collection = test_client.create_collection(
        name=collection_name
    )

    monkeypatch.setattr(
        vector_store,
        "collection",
        test_collection
    )

    yield test_collection

    test_client.delete_collection(name=collection_name)


def test_store_document_saves_chunks_and_metadata(isolated_collection):
    """Document chunks and their metadata should be stored."""

    chunks = [
        "Technical Skills\nPython and SQL.",
        "Experience\nWorked with FastAPI and Docker."
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    document_id = str(uuid.uuid4())

    vector_store.store_document(
        chunks=chunks,
        embeddings=embeddings,
        document_name="Test Document.pdf",
        document_id=document_id,
        file_hash="test-hash-123"
    )

    results = isolated_collection.get(
        include=["documents", "metadatas"]
    )

    assert len(results["ids"]) == 2
    assert results["documents"] == chunks

    for metadata in results["metadatas"]:
        assert metadata["document_id"] == document_id
        assert metadata["document_name"] == "Test Document.pdf"
        assert metadata["file_hash"] == "test-hash-123"


def test_store_document_rejects_empty_chunks():
    """Empty documents should not be stored."""

    with pytest.raises(ValueError):
        vector_store.store_document(
            chunks=[],
            embeddings=np.array([]),
            document_name="Empty.pdf",
            document_id=str(uuid.uuid4()),
            file_hash="empty-test-hash"
        )


def test_store_document_rejects_mismatched_embeddings():
    """Every chunk must have a corresponding embedding."""

    with pytest.raises(ValueError):
        vector_store.store_document(
            chunks=["First chunk", "Second chunk"],
            embeddings=np.array([[1.0, 0.0, 0.0]]),
            document_name="Test.pdf",
            document_id=str(uuid.uuid4()),
            file_hash="mismatch-test-hash"
        )


def test_hybrid_search_only_returns_selected_document(
    isolated_collection
):
    """Document filtering should prevent cross-document retrieval."""

    document_a = str(uuid.uuid4())
    document_b = str(uuid.uuid4())

    vector_store.store_document(
        chunks=[
            "PostgreSQL is the primary company database."
        ],
        embeddings=np.array([[1.0, 0.0, 0.0]]),
        document_name="IT Documentation.pdf",
        document_id=document_a,
        file_hash="document-a-hash"
    )

    vector_store.store_document(
        chunks=[
            "MongoDB is used for the customer application."
        ],
        embeddings=np.array([[0.0, 1.0, 0.0]]),
        document_name="Application Documentation.pdf",
        document_id=document_b,
        file_hash="document-b-hash"
    )

    results = vector_store.hybrid_search(
        query="What database does the company use?",
        query_embedding=np.array([1.0, 0.0, 0.0]),
        n_results=5,
        document_id=document_a
    )

    assert len(results["documents"][0]) == 1

    assert "PostgreSQL" in results["documents"][0][0]

    assert all(
        metadata["document_id"] == document_a
        for metadata in results["metadatas"][0]
    )


def test_hybrid_search_handles_empty_collection(isolated_collection):
    """Searching an empty collection should return empty results."""

    results = vector_store.hybrid_search(
        query="What information is available?",
        query_embedding=np.array([1.0, 0.0, 0.0])
    )

    assert results["documents"] == [[]]
    assert results["metadatas"] == [[]]
    assert results["distances"] == [[]]
    assert results["bm25_scores"] == [[]]
    assert results["hybrid_scores"] == [[]]


def test_section_metadata_is_detected(isolated_collection):
    """Recognized section headings should be stored correctly."""

    vector_store.store_document(
        chunks=[
            "Technical Skills\nPython, SQL and Docker."
        ],
        embeddings=np.array([[1.0, 0.0, 0.0]]),
        document_name="Skills.pdf",
        document_id=str(uuid.uuid4()),
        file_hash="skills-test-hash"
    )

    results = isolated_collection.get(
        include=["metadatas"]
    )

    assert results["metadatas"][0]["section"] == "Technical Skills"
