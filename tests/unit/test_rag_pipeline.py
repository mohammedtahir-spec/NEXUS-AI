
import numpy as np

from app import rag_pipeline


def mock_retrieval_results():
    return {
        "documents": [
            [
                "The company uses PostgreSQL.",
                "PostgreSQL is the primary database."
            ]
        ],
        "distances": [[0.12, 0.18]],
        "metadatas": [[
            {
                "document_name": "IT Documentation.pdf",
                "document_id": "doc-123",
                "section": "Database",
                "chunk_index": 0
            },
            {
                "document_name": "IT Documentation.pdf",
                "document_id": "doc-123",
                "section": "Database",
                "chunk_index": 1
            }
        ]],
        "bm25_scores": [[2.5, 1.8]],
        "hybrid_scores": [[0.032, 0.031]]
    }


def test_answer_question_generates_query_embedding(monkeypatch):
    """The question should be converted into an embedding."""

    captured = {}

    def fake_embeddings(texts):
        captured["texts"] = texts
        return np.array([[1.0, 0.0, 0.0]])

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        fake_embeddings
    )

    monkeypatch.setattr(
        rag_pipeline,
        "hybrid_search",
        lambda **kwargs: {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "bm25_scores": [[]],
            "hybrid_scores": [[]]
        }
    )

    result = rag_pipeline.answer_question(
        "What database does the company use?"
    )

    assert captured["texts"] == [
        "What database does the company use?"
    ]

    assert "could not find relevant information" in result


def test_answer_question_passes_document_filter(monkeypatch):
    """Retrieval should be restricted to the requested document."""

    captured = {}

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda texts: np.array([[1.0, 0.0, 0.0]])
    )

    def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "bm25_scores": [[]],
            "hybrid_scores": [[]]
        }

    monkeypatch.setattr(
        rag_pipeline,
        "hybrid_search",
        fake_hybrid_search
    )

    rag_pipeline.answer_question(
        question="What database is used?",
        document_id="doc-123"
    )

    assert captured["document_id"] == "doc-123"
    assert captured["n_results"] == 5


def test_answer_question_returns_answer_and_sources(monkeypatch):
    """The final response should include the answer and its sources."""

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda texts: np.array([[1.0, 0.0, 0.0]])
    )

    monkeypatch.setattr(
        rag_pipeline,
        "hybrid_search",
        lambda **kwargs: mock_retrieval_results()
    )

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, context: "The company uses PostgreSQL."
    )

    result = rag_pipeline.answer_question(
        "What database does the company use?"
    )

    assert "The company uses PostgreSQL." in result
    assert "SOURCES:" in result
    assert "IT Documentation.pdf (Database)" in result


def test_answer_question_deduplicates_sources(monkeypatch):
    """Repeated chunks from the same section should produce one source."""

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda texts: np.array([[1.0, 0.0, 0.0]])
    )

    monkeypatch.setattr(
        rag_pipeline,
        "hybrid_search",
        lambda **kwargs: mock_retrieval_results()
    )

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, context: "PostgreSQL is used."
    )

    result = rag_pipeline.answer_question("Which database is used?")

    source_line = "- IT Documentation.pdf (Database)"

    assert result.count(source_line) == 1


def test_answer_question_does_not_call_llm_without_results(monkeypatch):
    """The LLM should not be called when retrieval finds nothing."""

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda texts: np.array([[1.0, 0.0, 0.0]])
    )

    monkeypatch.setattr(
        rag_pipeline,
        "hybrid_search",
        lambda **kwargs: {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "bm25_scores": [[]],
            "hybrid_scores": [[]]
        }
    )

    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError("LLM should not be called.")

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        unexpected_llm_call
    )

    result = rag_pipeline.answer_question("Unknown information?")

    assert result == (
        "I could not find relevant information "
        "in the selected document."
    )
