def embed_single(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text.
            model: Embedding model identifier.

        Returns:
            Embedding vector for the provided text.
        """
        embeddings = self.embed_texts([text], model)
        return embeddings[0]