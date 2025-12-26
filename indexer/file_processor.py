"""
File collection and processing logic.
"""

import os
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from line_profiler import profile
from .ast_parser import extract_chunks
from config import IGNORED_DIRS, IGNORED_FILES, SUPPORTED_LANGS
from .summarizer import summarize_file, summarize_folder
from .chunk_storage import generate_chunk_id, save_full_chunk


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


@profile
def process_single_file(file_path: Path, repo_path: str, output_prefix: Optional[str] = None) -> tuple:
    """Process a single file to extract chunks and summary.

    Args:
        file_path: Path to the file to process.
        repo_path: Root path of the repository.
        output_prefix: Directory prefix for output files (for saving full chunks).

    Returns:
        Tuple containing (chunks, file_summary, summary_chunk, error).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        rel_path = str(file_path.relative_to(repo_path))
        lang = SUPPORTED_LANGS[file_path.suffix]

        # Extract chunks based on file type
        file_chunks = _extract_file_chunks(code, lang, rel_path)

        # Process each chunk: generate IDs and save content
        _process_chunks(file_chunks, code, rel_path, file_path, output_prefix)

        # Generate file summary if file is substantial
        file_summary, summary_chunk = _generate_file_summary(code, rel_path)

        return file_chunks, file_summary, summary_chunk, None

    except Exception:
        return [], None, None, f"Error processing {file_path}: {traceback.format_exc()}"


def _extract_file_chunks(code: str, lang: str, rel_path: str) -> List[Dict]:
    """Extract chunks from file based on language type."""
    if lang == "markdown":
        return [{
            "text": code,
            "metadata": {
                "level": "document",
                "file": rel_path,
                "type": "markdown",
                "location": {"file": rel_path},
            },
        }]
    else:
        # Extract code chunks for programming languages
        return extract_chunks(code, lang, rel_path)


def _process_chunks(file_chunks: List[Dict], code: str, rel_path: str, file_path: Path, output_prefix: Optional[str]) -> None:
    """Process chunks to generate IDs and save content."""
    for chunk in file_chunks:
        metadata = chunk["metadata"]
        level = metadata.get("level")

        if level == "code_chunk":
            _process_code_chunk(chunk, metadata, rel_path, file_path, output_prefix)
        elif level == "document":
            _process_document_chunk(chunk, metadata, code, rel_path, file_path, output_prefix)


def _process_code_chunk(chunk: Dict, metadata: Dict, rel_path: str, file_path: Path, output_prefix: Optional[str]) -> None:
    """Process a code chunk."""
    function_name = metadata.get("function_name", "unknown")
    docstring = metadata.get("docstring")
    start_line = metadata.get("start_line")
    end_line = metadata.get("end_line")

    # Ensure line numbers are integers
    if start_line is not None and end_line is not None:
        chunk_id = generate_chunk_id(rel_path, int(start_line), int(end_line))
        metadata["chunk_id"] = chunk_id

        if output_prefix:
            extension = file_path.suffix if file_path.suffix else ".txt"
            save_full_chunk(chunk_id, chunk["text"], output_prefix, extension)
            metadata["full_code_path"] = f"full_chunks/{chunk_id}{extension}"

        # Build searchable text from metadata
        metadata_text_parts = [f"Function: {function_name}"]
        metadata_text_parts.append(f"File: {rel_path}")
        metadata_text_parts.append(f"Lines: {start_line}-{end_line}")
        if docstring:
            metadata_text_parts.append(f"Docstring: {docstring}")

        chunk["text"] = "\n".join(metadata_text_parts)


def _process_document_chunk(chunk: Dict, metadata: Dict, code: str, rel_path: str, file_path: Path, output_prefix: Optional[str]) -> None:
    """Process a document chunk."""
    lines = code.split('\n')
    chunk_id = generate_chunk_id(rel_path, 1, len(lines))

    metadata["chunk_id"] = chunk_id

    if output_prefix:
        extension = file_path.suffix if file_path.suffix else ".txt"
        save_full_chunk(chunk_id, code, output_prefix, extension)
        metadata["full_code_path"] = f"full_chunks/{chunk_id}{extension}"

    # For documents, keep the full content as searchable text
    chunk["text"] = f"DOCUMENT: {rel_path}\n\n{code}"


def _generate_file_summary(code: str, rel_path: str) -> tuple:
    """Generate file summary if file is substantial enough."""
    if len(code) <= 100:
        return None, None

    summary = summarize_file(code, rel_path)
    file_summary = (rel_path, summary)
    summary_chunk = {
        "text": f"FILE: {rel_path}\n{summary}",
        "metadata": {
            "file": rel_path,
            "level": "file_summary",
            "location": {"file": rel_path},
        },
    }

    return file_summary, summary_chunk


@profile
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


@profile
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
