from fastembed import TextEmbedding
import numpy

class EmbeddingClient:
    def __init__(self):
        """Initialize the embedding model using a multilingual paraphrase model."""
        self.model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def embed(self, text: str) -> list[float]:
        """Convert a text string into a dense vector representation.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        return next(self.model.embed([text])).tolist()

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute the cosine similarity between two embedding vectors.

        Returns a value between -1 and 1, where 1 means identical direction,
        0 means orthogonal, and -1 means opposite direction.

        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.

        Returns:
            Cosine similarity score as a float.
        """
        a = numpy.array(vec_a)
        b = numpy.array(vec_b)
        return float(numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b)))
    

if __name__ == "__main__":
    client = EmbeddingClient()

    phrases = [
        "je veux créer un platformer 2d",
        "J'ai une idée de jeu de plateforme",
        "Les tortues mangent du sable",
        "Je souhaite développer un jeu de plateforme en 2 dimensions",
        "Une pizza 8 fromages"
    ]

    embeddings = [client.embed(p) for p in phrases]

    print(f"Vector dimension: {len(embeddings[0])}\n")

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            sim = client.cosine_similarity(embeddings[i], embeddings[j])
            print(f"[{sim:.4f}] \"{phrases[i]}\"  ↔  \"{phrases[j]}\"")
