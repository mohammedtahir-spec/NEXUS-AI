
import os
import tempfile
import uuid
import hashlib

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pdf_reader import extract_text
from app.chunker import chunk_text
from app.embeddings import generate_embeddings, EmbeddingServiceError
from app.vector_store import store_document, collection, VectorStoreError
from app.rag_pipeline import answer_question
from app.llm import LLMServiceError


MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


app = FastAPI(
    title="NEXUS AI",
    description="Enterprise Knowledge Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {"message": "NEXUS AI is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/about")
def about():
    return {
        "project": "NEXUS AI",
        "version": "1.0",
        "developer": "Mohammed"
    }


@app.get("/documents")
def list_documents():
    results = collection.get(include=["metadatas"])

    documents = {}

    for metadata in results["metadatas"]:
        if not metadata:
            continue

        document_id = metadata.get("document_id")
        document_name = metadata.get("document_name")

        if not document_id or not document_name:
            continue

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "document_name": document_name,
                "chunks": 0
            }

        documents[document_id]["chunks"] += 1

    return {
        "documents": sorted(
            documents.values(),
            key=lambda document: document["document_name"].lower()
        ),
        "total_documents": len(documents)
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    document_name = Path(file.filename).name

    file_content = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(file_content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the maximum upload size of 20 MB."
        )

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    file_hash = hashlib.sha256(file_content).hexdigest()

    existing_documents = collection.get(
        where={"file_hash": file_hash},
        include=["metadatas"]
    )

    if existing_documents["ids"]:
        existing_name = existing_documents["metadatas"][0].get(
            "document_name",
            "an existing document"
        )

        raise HTTPException(
            status_code=409,
            detail=f"This PDF has already been uploaded as '{existing_name}'."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        try:
            extracted_text = extract_text(temp_path)

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc)
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not read this PDF. Make sure it is valid."
            ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        await file.close()

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in this PDF. Scanned PDFs require OCR."
        )

    chunks = chunk_text(extracted_text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not create chunks from this document."
        )

    document_id = str(uuid.uuid4())

    try:
        embeddings = generate_embeddings(chunks)

        store_document(
            chunks=chunks,
            embeddings=embeddings,
            document_name=document_name,
            document_id=document_id,
            file_hash=file_hash
        )

    except EmbeddingServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the document."
        ) from exc

    return {
        "message": "Document uploaded successfully",
        "document_id": document_id,
        "document_name": document_name,
        "chunks_stored": len(chunks)
    }


@app.post("/documents/{document_id}/ask")
def ask_document(
    document_id: str,
    request: QuestionRequest
):

    try:
        document_id = str(uuid.UUID(document_id))

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID."
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    document = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    if not document["ids"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        answer = answer_question(
            question=question,
            document_id=document_id
        )

    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    return {
        "document_id": document_id,
        "document_name": document["metadatas"][0]["document_name"],
        "question": question,
        "answer": answer
    }
