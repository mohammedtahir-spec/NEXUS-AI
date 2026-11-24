
import uuid

import chromadb
import numpy as np
import pytest

from app import vector_store


@pytest.fixture
def isolated_collection(monkeypatch):
    """Create an isolated ChromaDB collection for testing."""

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


def test_bm25_retrieves_chunk_missed_by_semantic_search(
    isolated_collection
):
    """
    BM25 should retrieve a relevant chunk even when
    semantic search does not include it in its candidates.
    """

    document_id = str(uuid.uuid4())

    vector_store.store_document(
        chunks=[
            "Technical Skills\nPostgreSQL is the primary company database.",
            "Experience\nAurora Serverless handles monthly compensation payroll.",
            "Projects\nDocker is used to deploy internal applications."
        ],
        embeddings=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]),
        document_name="Company Documentation.pdf",
        document_id=document_id,
        file_hash="test_hash_001"
    )

    results = vector_store.hybrid_search(
        query="Which system handles monthly compensation payroll?",
        query_embedding=np.array([1.0, 0.0, 0.0]),
        n_results=2,
        document_id=document_id,
        candidate_limit=1
    )

    retrieved_documents = results["documents"][0]
    retrieved_distances = results["distances"][0]

    target_index = next(
        (
            i
            for i, document in enumerate(retrieved_documents)
            if "Aurora Serverless" in document
        ),
        None
    )

    # BM25 must retrieve the target chunk.
    assert target_index is not None

    # The target was not returned by semantic retrieval.
    assert retrieved_distances[target_index] is None

    # The result must have a valid hybrid score.
    assert results["hybrid_scores"][0][target_index] > 0

    # The target must have a positive BM25 score.
    assert results["bm25_scores"][0][target_index] > 0


def test_bm25_does_not_add_unmatched_lexical_candidates(
    isolated_collection
):
    """
    BM25 must not introduce candidates when no query
    terms match any document.
    """

    document_id = str(uuid.uuid4())

    vector_store.store_document(
        chunks=[
            "Technical Skills\nPython programming and SQL databases.",
            "Experience\nDocker containers and Kubernetes orchestration."
        ],
        embeddings=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]),
        document_name="Technical Documentation.pdf",
        document_id=document_id,
        file_hash="test_hash_002"
    )

    results = vector_store.hybrid_search(
        query="quantum astronomy telescope",
        query_embedding=np.array([1.0, 0.0, 0.0]),
        n_results=5,
        document_id=document_id,
        candidate_limit=1
    )

    # Semantic retrieval returns only one candidate.
    assert len(results["documents"][0]) == 1

    # That candidate came from semantic retrieval.
    assert results["distances"][0][0] is not None

    # BM25 did not find any lexical matches.
    assert results["bm25_scores"][0][0] == 0.0

    # No additional lexical-only candidates were introduced.
    assert len(results["documents"][0]) == 1
