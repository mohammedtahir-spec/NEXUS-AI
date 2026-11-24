
import uuid

import pytest
from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


def test_health_endpoint():
    """Health endpoint should return a healthy status."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_list_documents(monkeypatch):
    """Document endpoint should return stored document information."""

    fake_metadata = [
        {
            "document_id": "doc-123",
            "document_name": "Company Policies.pdf"
        },
        {
            "document_id": "doc-123",
            "document_name": "Company Policies.pdf"
        },
        {
            "document_id": "doc-456",
            "document_name": "IT Documentation.pdf"
        }
    ]

    monkeypatch.setattr(
        main.collection,
        "get",
        lambda **kwargs: {"metadatas": fake_metadata}
    )

    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert data["total_documents"] == 2
    assert len(data["documents"]) == 2


def test_upload_rejects_unsupported_file():
    """Only PDF files should be accepted."""

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "document.txt",
                b"This is a text file.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_ask_rejects_invalid_document_id():
    """Malformed document IDs should be rejected."""

    response = client.post(
        "/documents/not-a-valid-uuid/ask",
        json={"question": "What is this document about?"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid document ID."


def test_ask_rejects_nonexistent_document(monkeypatch):
    """Unknown document IDs should return 404."""

    monkeypatch.setattr(
        main.collection,
        "get",
        lambda **kwargs: {
            "ids": [],
            "metadatas": []
        }
    )

    document_id = str(uuid.uuid4())

    response = client.post(
        f"/documents/{document_id}/ask",
        json={"question": "What is this document about?"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_ask_rejects_whitespace_question():
    """Questions containing only whitespace should be rejected."""

    document_id = str(uuid.uuid4())

    response = client.post(
        f"/documents/{document_id}/ask",
        json={"question": "   "}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_ask_rejects_empty_question():
    """An empty question should fail request validation."""

    document_id = str(uuid.uuid4())

    response = client.post(
        f"/documents/{document_id}/ask",
        json={"question": ""}
    )

    assert response.status_code == 422
