def process_files(files_to_process: List[Path], repo_path: str, output_prefix: Optional[str] = None) -> tuple:
    """Process files to extract chunks and file summaries using multithreading.

    Args:
        files_to_process: List of file paths to process.
        repo_path: Root path of the repository.
        output_prefix: Directory prefix for output files (for saving full chunks).

    Returns:
        Tuple containing (chunks, file_summaries) where chunks is a list of all extracted chunks
        and file_summaries is a dict mapping file paths to their summaries.
    """
    chunks = []
    file_summaries = {}
    errors = []

    # Thread-safe data structures
    chunks_lock = threading.Lock()
    summaries_lock = threading.Lock()
    errors_lock = threading.Lock()

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

    # Process files with 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_single_file, file_path, repo_path, output_prefix)
            for file_path in files_to_process
        ]

        # Use tqdm to track progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting chunks"
        ):
            collect_results(future)

    # Print any errors that occurred
    for error in errors:
        print(error)

    return chunks, file_summaries