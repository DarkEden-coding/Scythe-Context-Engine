def save_index(
    index, chunks: List[Dict], repo_path: str, output_prefix: str, embedding_dim: int
):
    """Save the FAISS index and metadata.

    Args:
        index: FAISS index object to save.
        chunks: List of chunk dictionaries.
        repo_path: Original repository path.
        output_prefix: Directory prefix for output files.
        embedding_dim: Dimension of the embeddings.
    """
    os.makedirs(output_prefix, exist_ok=True)

    faiss.write_index(index, f"{output_prefix}/index.faiss")  # type: ignore

    with open(f"{output_prefix}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(f"{output_prefix}/meta.json", "w") as f:
        json.dump(
            {
                "repo_path": repo_path,
                "total_chunks": len(chunks),
                "embedding_dim": embedding_dim,
                "model": "nomic-embed-text-v1.5",
            },
            f,
            indent=2,
        )