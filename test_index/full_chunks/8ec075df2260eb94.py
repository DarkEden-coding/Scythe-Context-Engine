def collect_folder_result(future):
        """Collect results from completed folder futures."""
        folder_chunk = future.result()
        with chunks_lock:
            chunks.append(folder_chunk)