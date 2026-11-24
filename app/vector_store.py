
import chromadb
import re

from rank_bm25 import BM25Okapi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEFAULT_CANDIDATE_LIMIT = 20
RRF_K = 60


# --------------------------------------------------
# ChromaDB initialization
# --------------------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="cv_documents"
)


# --------------------------------------------------
# Custom exception
# --------------------------------------------------

class VectorStoreError(Exception):
    """Raised when a vector database operation fails."""


# --------------------------------------------------
# Metadata generation
# --------------------------------------------------

def build_metadata(
    chunks,
    document_names,
    document_ids=None,
    file_hash=None
):
    """Build metadata for every document chunk."""

    metadata = []

    for i, chunk in enumerate(chunks):
        section = "General"

        lines = chunk.splitlines()

        for line in lines:
            cleaned = line.strip()

            if cleaned:
                section = cleaned[:100]
                break

        item = {
            "document_name": document_names[i],
            "section": section,
            "chunk_index": i
        }

        if document_ids is not None:
            item["document_id"] = document_ids[i]

        if file_hash is not None:
            item["file_hash"] = file_hash

        metadata.append(item)

    return metadata


# --------------------------------------------------
# Document storage
# --------------------------------------------------

def store_document(
    chunks,
    embeddings,
    document_name,
    document_id,
    file_hash
):
    """Store or update a document in ChromaDB."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Each chunk must have a corresponding embedding."
        )

    if not chunks:
        raise ValueError("No chunks were provided.")

    ids = [
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = build_metadata(
        chunks=chunks,
        document_names=[document_name] * len(chunks),
        document_ids=[document_id] * len(chunks),
        file_hash=file_hash
    )

    try:
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    except Exception as exc:
        raise VectorStoreError(
            "Failed to store the document in the vector database."
        ) from exc


# --------------------------------------------------
# Legacy storage compatibility
# --------------------------------------------------

def store_chunks(chunks, embeddings, document_names):
    """Store or update chunks using the legacy interface."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Each chunk must have a corresponding embedding."
        )

    metadata = build_metadata(
        chunks=chunks,
        document_names=document_names
    )

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata
    )


def rebuild_store(chunks, embeddings, document_names):
    """Legacy function for rebuilding the entire collection."""

    global collection

    if not chunks:
        raise ValueError("No chunks were provided.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Each chunk must have a corresponding embedding."
        )

    metadata = build_metadata(chunks, document_names)

    client.delete_collection(name="cv_documents")

    collection = client.create_collection(
        name="cv_documents"
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata
    )


# --------------------------------------------------
# Semantic search
# --------------------------------------------------

def search_chunks(query_embedding, n_results=2):
    """Perform semantic search."""

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


# --------------------------------------------------
# Shared helpers
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric tokens."""

    return re.findall(
        r"\b[a-z0-9]+\b",
        text.lower()
    )


def empty_search_results():
    """Return a consistent empty search result structure."""

    return {
        "documents": [[]],
        "distances": [[]],
        "metadatas": [[]],
        "bm25_scores": [[]],
        "hybrid_scores": [[]]
    }


# --------------------------------------------------
# Hybrid search: Semantic + BM25 + RRF
# --------------------------------------------------

def hybrid_search(
    query,
    query_embedding,
    n_results=2,
    document_name=None,
    document_id=None,
    candidate_limit=DEFAULT_CANDIDATE_LIMIT
):
    """
    Combine independent semantic and lexical retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    if candidate_limit < 1:
        raise ValueError("candidate_limit must be at least 1.")

    if n_results < 1:
        raise ValueError("n_results must be at least 1.")

    # Step 1: Build document filters.
    filters = []

    if document_id is not None:
        filters.append({
            "document_id": document_id
        })

    if document_name is not None:
        filters.append({
            "document_name": document_name
        })

    where_filter = None

    if len(filters) == 1:
        where_filter = filters[0]

    elif len(filters) > 1:
        where_filter = {
            "$and": filters
        }

    # Step 2: Retrieve the selected document corpus.
    corpus_results = collection.get(
        where=where_filter,
        include=[
            "documents",
            "metadatas"
        ]
    )

    corpus_ids = corpus_results["ids"]
    corpus_documents = corpus_results["documents"]
    corpus_metadatas = corpus_results["metadatas"]

    if not corpus_ids:
        return empty_search_results()

    # Step 3: Independent semantic retrieval.
    semantic_results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=min(candidate_limit, len(corpus_ids)),
        where=where_filter,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    semantic_ids = semantic_results["ids"][0]
    semantic_distances = semantic_results["distances"][0]

    # Step 4: Independent BM25 retrieval.
    query_tokens = tokenize(query)

    tokenized_corpus = []
    lexical_ids = []

    for chunk_id, document in zip(
        corpus_ids,
        corpus_documents
    ):
        tokens = tokenize(document)

        if tokens:
            lexical_ids.append(chunk_id)
            tokenized_corpus.append(tokens)

    bm25_scores_by_id = {}
    lexical_ranks = {}

    if query_tokens and tokenized_corpus:

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)

        bm25_ranking = sorted(
            range(len(lexical_ids)),
            key=lambda i: scores[i],
            reverse=True
        )

        for index, score in enumerate(scores):
            bm25_scores_by_id[lexical_ids[index]] = float(score)

        # Keep only candidates with positive BM25 scores.
        positive_matches = [
            index
            for index in bm25_ranking
            if scores[index] > 0
        ][:candidate_limit]

        for rank, index in enumerate(
            positive_matches,
            start=1
        ):
            lexical_ranks[lexical_ids[index]] = rank

    # Step 5: Build semantic ranks.
    semantic_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            semantic_ids,
            start=1
        )
    }

    distances_by_id = dict(
        zip(semantic_ids, semantic_distances)
    )

    # Step 6: Merge candidates using RRF.
    candidate_ids = list(dict.fromkeys(
        semantic_ids + list(lexical_ranks.keys())
    ))

    if not candidate_ids:
        return empty_search_results()

    hybrid_scores_by_id = {}

    for chunk_id in candidate_ids:

        score = 0.0

        if chunk_id in semantic_ranks:
            score += 1 / (
                RRF_K + semantic_ranks[chunk_id]
            )

        if chunk_id in lexical_ranks:
            score += 1 / (
                RRF_K + lexical_ranks[chunk_id]
            )

        hybrid_scores_by_id[chunk_id] = score

    # Step 7: Select final results.
    final_ranking = sorted(
        candidate_ids,
        key=lambda chunk_id: hybrid_scores_by_id[chunk_id],
        reverse=True
    )

    selected_ids = final_ranking[:n_results]

    documents_by_id = dict(
        zip(corpus_ids, corpus_documents)
    )

    metadatas_by_id = dict(
        zip(corpus_ids, corpus_metadatas)
    )

    # Step 8: Return results in the existing format.
    return {
        "documents": [[
            documents_by_id[chunk_id]
            for chunk_id in selected_ids
        ]],
        "distances": [[
            distances_by_id.get(chunk_id)
            for chunk_id in selected_ids
        ]],
        "metadatas": [[
            metadatas_by_id[chunk_id]
            for chunk_id in selected_ids
        ]],
        "bm25_scores": [[
            bm25_scores_by_id.get(chunk_id, 0.0)
            for chunk_id in selected_ids
        ]],
        "hybrid_scores": [[
            hybrid_scores_by_id[chunk_id]
            for chunk_id in selected_ids
        ]]
    }
