def collect_embedding_result(future):
        """Collect results from completed embedding futures."""
        batch_idx, batch_embs, success = future.result()
        if success and batch_embs is not None:
            start_idx = batch_idx * batch_size
            with embeddings_lock:
                for j, emb in enumerate(batch_embs):
                    embeddings[start_idx + j] = emb