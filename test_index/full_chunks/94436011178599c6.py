def embed_texts(self, texts: Sequence[str], model: str, options: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Iterable of text inputs.
            model: Embedding model identifier.
            options: Additional OpenRouter parameters.

        Returns:
            Embedding vectors corresponding to each input text.
        """
        payload = {"model": model, "input": list(texts)}
        if options:
            payload.update(options)
        response_json = self._post("/embeddings", payload)
        data = response_json.get("data")
        if not isinstance(data, list):
            raise OpenRouterError("Embeddings response missing data list.")
        embeddings: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise OpenRouterError("Embeddings response contained invalid item.")
            embeddings.append([float(value) for value in embedding])
        return embeddings