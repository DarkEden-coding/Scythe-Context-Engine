def collect_results(future):
        """Collect results from completed futures."""
        file_chunks, file_summary, summary_chunk, error = future.result()

        if file_chunks:
            with chunks_lock:
                chunks.extend(file_chunks)

        if file_summary:
            rel_path, summary = file_summary
            with summaries_lock:
                file_summaries[rel_path] = summary
            with chunks_lock:
                chunks.append(summary_chunk)

        if error:
            with errors_lock:
                errors.append(error)