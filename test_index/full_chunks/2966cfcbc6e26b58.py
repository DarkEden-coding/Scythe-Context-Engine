def generate_folder_summaries(
    file_summaries: Dict[str, str], chunks: List[Dict]
) -> List[Dict]:
    """Generate folder summaries and add to chunks using multithreading.

    Args:
        file_summaries: Dictionary mapping file paths to their summaries.
        chunks: List of existing chunks to append folder summaries to.

    Returns:
        Updated chunks list with folder summary chunks added.
    """
    folder_groups = defaultdict(list)

    for path, summary in file_summaries.items():
        folder = str(Path(path).parent)
        folder_groups[folder].append((path, summary))

    # Prepare folders to process (skip root)
    folders_to_process = [
        (folder, files) for folder, files in folder_groups.items() if folder != "."
    ]

    if not folders_to_process:
        return chunks

    # Thread-safe chunks list
    chunks_lock = threading.Lock()

    def process_folder(folder_data):
        """Process a single folder summary."""
        folder, files = folder_data
        folder_sum = summarize_folder(files)
        folder_chunk = {
            "text": f"FOLDER: {folder}\n{folder_sum}",
            "metadata": {
                "folder": folder,
                "level": "folder_summary",
                "location": {"folder": folder},
            },
        }
        return folder_chunk

    def collect_folder_result(future):
        """Collect results from completed folder futures."""
        folder_chunk = future.result()
        with chunks_lock:
            chunks.append(folder_chunk)

    # Process folders with 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_folder, folder_data)
            for folder_data in folders_to_process
        ]

        # Use tqdm to track progress
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating folder summaries",
        ):
            collect_folder_result(future)

    return chunks