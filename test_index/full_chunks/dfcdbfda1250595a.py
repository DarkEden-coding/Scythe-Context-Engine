def collect_files_to_process(repo_path: str) -> List[Path]:
    """Collect all supported code files from the repository.

    Args:
        repo_path: Root path of the repository to index.

    Returns:
        List of Path objects for supported code files, excluding ignored directories and files.
    """
    files_to_process = []

    for root, _, files in os.walk(repo_path):
        # Skip ignored directories (check if any path component matches)
        path_parts = Path(root).parts
        if any(part in IGNORED_DIRS for part in path_parts):
            continue

        for file in files:
            # Skip ignored files
            if file in IGNORED_FILES:
                continue

            ext = Path(file).suffix
            if ext in SUPPORTED_LANGS:
                files_to_process.append(Path(root) / file)

    return files_to_process