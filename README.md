
# NEXUS AI

### Enterprise Knowledge Assistant powered by Retrieval-Augmented Generation

NEXUS AI is a document-grounded question-answering application that allows users to upload PDF documents and ask questions about their contents.

It combines semantic search, keyword-based retrieval, and a locally hosted Large Language Model (LLM) to generate answers grounded in retrieved document content.

The project demonstrates an end-to-end RAG architecture, including document ingestion, embedding generation, vector storage, hybrid retrieval, API development, frontend integration, and automated testing.

---

## Features

- **PDF ingestion:** Upload text-based PDF documents.
- **Text extraction:** Extract readable text using PyPDF.
- **Document chunking:** Split extracted content into smaller chunks for retrieval.
- **Semantic search:** Generate embeddings using Sentence Transformers.
- **Vector storage:** Store document chunks and metadata in ChromaDB.
- **Hybrid retrieval:** Combine semantic retrieval with BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Document isolation:** Restrict questions and retrieval to a selected document.
- **Grounded answers:** Generate responses using retrieved document context.
- **Source attribution:** Associate answers with the uploaded document.
- **Duplicate detection:** Detect previously uploaded PDFs using SHA-256 hashes.
- **Persistent storage:** Preserve vector database data using Docker volumes.
- **Local LLM inference:** Use Ollama to run the language model locally.

---

## Architecture

```text
                  USER
                   |
                   v
            React Frontend
                   |
                   v
             FastAPI Backend
                   |
          +--------+---------+
          |                  |
          v                  v
     PDF Upload         User Question
          |                  |
          v                  v
    Text Extraction     Query Embedding
          |                  |
          v                  v
    Text Chunking       Hybrid Retrieval
          |             /            \
          v            v              v
    Embeddings      Semantic Search   BM25
          |            \              /
          v             \            /
       ChromaDB       Reciprocal Rank Fusion
          |                  |
          +------------------+
                             |
                             v
                    Retrieved Context
                             |
                             v
                       Ollama LLM
                             |
                             v
                    Grounded Answer
                             |
                             v
                       React UI
```

### RAG Pipeline

1. A user uploads a PDF document.
2. The backend validates the file and checks for duplicates.
3. PyPDF extracts readable text.
4. The text is divided into smaller chunks.
5. Sentence Transformers generates embeddings.
6. ChromaDB stores the chunks, embeddings, and metadata.
7. A user submits a question for a specific document.
8. The question is embedded and passed to the hybrid retrieval system.
9. Semantic and BM25 retrieval results are combined using Reciprocal Rank Fusion.
10. The most relevant chunks are passed to Ollama as context.
11. The LLM generates an answer using the supplied document evidence.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React 19 |
| Frontend tooling | Vite |
| Backend | Python 3.12, FastAPI |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Ranking | Reciprocal Rank Fusion |
| LLM runtime | Ollama |
| Default LLM | llama3.2:3b |
| PDF processing | PyPDF |
| Containerization | Docker, Docker Compose |
| Testing | Pytest, HTTPX |

---

## Project Structure

```text
enterprise-rag/
|
├── app/
│   ├── main.py
│   ├── pdf_reader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── llm.py
|
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
|
├── tests/
│   ├── unit/
│   └── integration/
|
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .dockerignore
└── .gitignore
```

---

## Getting Started

### Prerequisites

Install the following:

- Docker and Docker Compose
- Ollama
- Git

### 1. Clone the repository

```bash
git clone https://github.com/mohammedtahir-spec/NEXUS-AI.git
cd NEXUS-AI
```

### 2. Download the LLM

Install Ollama and download the default model:

```bash
ollama pull llama3.2:3b
```

Make sure the Ollama service is running.

### 3. Start the application

Build and start both services:

```bash
docker compose up -d --build
```

The first build may take some time because the backend installs its dependencies and downloads the embedding model when first needed.

### 4. Access the application

| Service | URL |
|---|---|
| Frontend | http://localhost:8081 |
| Backend API | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

### 5. Stop the application

```bash
docker compose down
```

Docker volumes and the local ChromaDB directory are preserved when using this command.

---

## Configuration

The backend supports the following environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama service address |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model used for answer generation |
| `HF_HOME` | Hugging Face default | Embedding model cache directory |

The Docker Compose configuration uses `host.docker.internal` to connect the backend container to Ollama running on the host.

---

## API Reference

### Health Check

`GET /health`

Checks whether the backend is responding.

Example response:

```json
{
  "status": "healthy"
}
```

### List Documents

`GET /documents`

Returns uploaded documents and their stored chunk counts.

### Upload a PDF

`POST /documents/upload`

Content type: `multipart/form-data`

Parameter:

- `file`: PDF document

Example response:

```json
{
  "message": "Document uploaded successfully",
  "document_id": "generated-document-uuid",
  "document_name": "company-policy.pdf",
  "chunks_stored": 15
}
```

### Ask a Question

`POST /documents/{document_id}/ask`

Request:

```json
{
  "question": "What is the company's annual leave policy?"
}
```

Example response:

```json
{
  "document_id": "generated-document-uuid",
  "document_name": "company-policy.pdf",
  "question": "What is the company's annual leave policy?",
  "answer": "The document states that..."
}
```

### Other Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/about` | Project information |

Interactive API documentation is available at `/docs`.

---

## Testing

The project includes unit and integration tests covering the main application components.

Run the complete test suite locally:

```bash
python -m pytest -v
```

Run unit tests:

```bash
python -m pytest tests/unit -v
```

Run integration tests:

```bash
python -m pytest tests/integration -v
```

The tests cover:

- PDF text extraction
- Text chunking
- Vector storage and metadata
- Hybrid retrieval
- Document filtering
- RAG answer generation
- API endpoints
- Error handling
- Duplicate document detection
- Empty and unsupported files
- Storage persistence behavior

---

## Current Limitations

- Only text-based PDF documents are supported.
- Scanned PDFs require OCR, which is not currently implemented.
- DOCX, XLSX, and other document formats are not supported.
- The application does not currently implement user authentication or authorization.
- The default deployment is intended for local demonstration, not unrestricted production exposure.
- Answer quality depends on document quality, retrieval relevance, and the selected language model.
- Prompt-injection instructions in the system prompt are a mitigation, not a complete security boundary.

---

## Future Improvements

- OCR support for scanned PDFs.
- Support for additional document formats.
- User authentication and access control.
- Improved document-level metadata and filtering.
- Retrieval evaluation with a reusable benchmark dataset.
- More robust answer-quality evaluation.
- Production deployment and monitoring.
- Improved security controls and rate limiting.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
