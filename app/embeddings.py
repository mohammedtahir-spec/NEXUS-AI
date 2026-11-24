
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


class EmbeddingServiceError(Exception):
    """Raised when embedding generation fails."""


def generate_embeddings(chunks: list[str]):
    if not chunks:
        raise ValueError("Cannot generate embeddings from empty input.")

    if not all(isinstance(chunk, str) and chunk.strip() for chunk in chunks):
        raise ValueError("All chunks must contain non-empty text.")

    try:
        embeddings = model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        if len(embeddings) != len(chunks):
            raise EmbeddingServiceError(
                "Embedding count does not match chunk count."
            )

        return embeddings

    except Exception as exc:
        if isinstance(exc, EmbeddingServiceError):
            raise

        raise EmbeddingServiceError(
            "Failed to generate document embeddings."
        ) from exc
