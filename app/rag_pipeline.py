
from app.embeddings import generate_embeddings
from app.vector_store import hybrid_search
from app.llm import generate_answer


def answer_question(
    question: str,
    document_name: str = None,
    document_id: str = None
) -> str:

    # Step 1: Generate question embedding
    query_embedding = generate_embeddings([question])[0]

    # Step 2: Retrieve relevant chunks
    results = hybrid_search(
        query=question,
        query_embedding=query_embedding,
        n_results=5,
        document_name=document_name,
        document_id=document_id
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    bm25_scores = results["bm25_scores"][0]
    hybrid_scores = results["hybrid_scores"][0]

    # Step 3: Handle empty retrieval results
    if not documents:
        return (
            "I could not find relevant information "
            "in the selected document."
        )

    # Step 4: Log retrieval metadata without document contents
    print("\nRETRIEVED SOURCES:")
    print("-" * 60)

    for i, metadata in enumerate(metadatas):
        print(f"\nSource {i + 1}")
        print(f"Document: {metadata['document_name']}")
        print(f"Document ID: {metadata.get('document_id', 'Legacy')}")
        print(f"Section: {metadata['section']}")
        print(f"Chunk index: {metadata['chunk_index']}")

        distance = distances[i]
        distance_text = (
            f"{distance:.4f}"
            if distance is not None
            else "N/A"
        )

        print(f"Semantic distance: {distance_text}")
        print(f"BM25 score: {bm25_scores[i]:.4f}")
        print(f"Hybrid score: {hybrid_scores[i]:.4f}")

    # Step 5: Build source-aware context
    context_parts = []

    for i, document in enumerate(documents):
        metadata = metadatas[i]

        source = (
            f"[Source {i + 1}]\n"
            f"Document: {metadata['document_name']}\n"
            f"Section: {metadata['section']}\n"
            f"Content:\n{document}"
        )

        context_parts.append(source)

    context = "\n\n".join(context_parts)

    # Step 6: Generate answer
    answer = generate_answer(question, context)

    # Step 7: Build deterministic source list
    unique_sources = {}

    for metadata in metadatas:
        source_name = metadata["document_name"]
        section = metadata["section"]
        unique_sources[(source_name, section)] = True

    source_lines = []

    for source_name, section in unique_sources:
        source_lines.append(f"- {source_name} ({section})")

    sources_text = "\n".join(source_lines)

    # Step 8: Return answer with sources
    return f"{answer}\n\nSOURCES:\n{sources_text}"
