"""
File collection and processing logic.
"""

import hashlib
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from line_profiler import profile
from .ast_parser import extract_chunks
from config.config import (
    IGNORE_PATTERN_MATCHER,
    SUPPORTED_LANGS,
    USE_BATCH_FOR_INDEXING,
    USE_BATCH_FOR_MCP_INCREMENTAL_INDEXING,
    get_groq_batch_client,
)
from .summarizer import (
    batch_summarize_files,
    batch_summarize_folders,
    summarize_file,
    summarize_folder,
)
from .chunk_storage import generate_chunk_id, save_full_chunk
from utils.logger import log_event


def hash_file(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64kb chunks
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_files_to_process(repo_path: str) -> List[Path]:
    """Collect all supported code files, excluding ignored patterns.

    Args:
        repo_path: Root path of the repository to index.

    Returns:
        List of Path objects for supported code files, excluding ignored patterns.
    """
    files_to_process = []
    repo_path_obj = Path(repo_path)

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)
        rel_root = root_path.relative_to(repo_path_obj)

        # Prune directories in-place to avoid traversing ignored trees
        dirs_to_remove = []
        for dir_name in dirs:
            dir_rel_path = str(rel_root / dir_name)
            if IGNORE_PATTERN_MATCHER.matches(dir_rel_path):
                dirs_to_remove.append(dir_name)

        for dir_name in dirs_to_remove:
            dirs.remove(dir_name)

        # Filter files
        for file in files:
            file_rel_path = str(rel_root / file)

            if IGNORE_PATTERN_MATCHER.matches(file_rel_path):
                continue

            ext = Path(file).suffix
            if ext in SUPPORTED_LANGS:
                files_to_process.append(root_path / file)

    return files_to_process


@profile
def process_single_file(
    file_path: Path,
    repo_path: str,
    output_prefix: Optional[str] = None,
    skip_summary: bool = False,
) -> tuple:
    """Process a single file to extract chunks and summary.

    Args:
        file_path: Path to the file to process.
        repo_path: Root path of the repository.
        output_prefix: Directory prefix for output files (for saving full chunks).
        skip_summary: If True, skip individual file summarization (for batch mode).

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

        # Generate file summary if file is substantial (skip in batch mode)
        if skip_summary:
            file_summary, summary_chunk = None, None
        else:
            file_summary, summary_chunk = _generate_file_summary(code, rel_path)

        return file_chunks, file_summary, summary_chunk, None

    except Exception:
        return [], None, None, f"Error processing {file_path}: {traceback.format_exc()}"


def _extract_file_chunks(code: str, lang: str, rel_path: str) -> List[Dict]:
    """Extract chunks from file based on language type."""
    if lang == "markdown":
        return [
            {
                "text": code,
                "metadata": {
                    "level": "document",
                    "file": rel_path,
                    "type": "markdown",
                    "location": {"file": rel_path},
                },
            }
        ]
    else:
        # Extract code chunks for programming languages
        return extract_chunks(code, lang, rel_path)


def _process_chunks(
    file_chunks: List[Dict],
    code: str,
    rel_path: str,
    file_path: Path,
    output_prefix: Optional[str],
) -> None:
    """Process chunks to generate IDs and save content."""
    for chunk in file_chunks:
        metadata = chunk["metadata"]
        level = metadata.get("level")

        if level == "code_chunk":
            _process_code_chunk(chunk, metadata, rel_path, file_path, output_prefix)
        elif level == "document":
            _process_document_chunk(
                chunk, metadata, code, rel_path, file_path, output_prefix
            )


def _process_code_chunk(
    chunk: Dict,
    metadata: Dict,
    rel_path: str,
    file_path: Path,
    output_prefix: Optional[str],
) -> None:
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

        # Build searchable text from metadata ONLY
        # Embedding models work best with natural language descriptions, not raw code
        # The full code is saved separately and loaded during the refinement phase
        metadata_text_parts = []
        if function_name != "unknown":
            metadata_text_parts.append(f"Function: {function_name}")
        metadata_text_parts.append(f"File: {rel_path}")
        metadata_text_parts.append(f"Lines: {start_line}-{end_line}")
        metadata_text_parts.append(f"Type: {metadata.get('type', 'code')}")
        if docstring:
            metadata_text_parts.append(f"Documentation: {docstring}")

        chunk["text"] = "\n".join(metadata_text_parts)


def _process_document_chunk(
    chunk: Dict,
    metadata: Dict,
    code: str,
    rel_path: str,
    file_path: Path,
    output_prefix: Optional[str],
) -> None:
    """Process a document chunk."""
    lines = code.split("\n")
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
def process_files(
    files_to_process: List[Path],
    repo_path: str,
    output_prefix: Optional[str] = None,
    quiet: bool = False,
    for_mcp_query: bool = False,
) -> tuple:
    """Process files to extract chunks and file summaries using multithreading.

    Args:
        files_to_process: List of file paths to process.
        repo_path: Root path of the repository.
        output_prefix: Directory prefix for output files (for saving full chunks).
        quiet: If True, suppress progress bars.
        for_mcp_query: If True, indicates processing is triggered by MCP query.
                       Uses MCP-specific batch setting instead of general setting.

    Returns:
        Tuple containing (chunks, file_summaries) where chunks is a list of all extracted chunks
        and file_summaries is a dict mapping file paths to their summaries.
    """
    file_processing_start_time = time.time()
    chunks = []
    file_summaries = {}
    errors = []

    # Check if batch mode is configured BEFORE processing
    batch_client = get_groq_batch_client()
    use_batch = (
        (USE_BATCH_FOR_MCP_INCREMENTAL_INDEXING if for_mcp_query else USE_BATCH_FOR_INDEXING)
        and batch_client is not None
    )

    # Log batch mode decision
    provider = "groq" if batch_client else None
    log_event(
        event="batch_mode_decision",
        level="INFO",
        phase="indexing",
        component="file_processor",
        message=f"Batch mode {'enabled' if use_batch else 'disabled'}",
        data={
            "use_batch": use_batch,
            "request_count": len(files_to_process),
            "provider": provider,
        },
    )

    if use_batch:
        if not quiet:
            print("Batch mode enabled - will use Groq Batch API for summarization")

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

        # Only collect individual summaries if NOT using batch mode
        if not use_batch and file_summary:
            rel_path, summary = file_summary
            with summaries_lock:
                file_summaries[rel_path] = summary
            with chunks_lock:
                chunks.append(summary_chunk)

        if error:
            with errors_lock:
                errors.append(error)

    # Process files with 8 threads (extracting chunks, skip summaries if using batch)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                process_single_file, file_path, repo_path, output_prefix, use_batch
            )
            for file_path in files_to_process
        ]

        # Use tqdm to track progress (unless quiet mode)
        if quiet:
            for future in as_completed(futures):
                collect_results(future)
        else:
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Extracting chunks"
            ):
                collect_results(future)

    # Print any errors that occurred
    for error in errors:
        print(error, file=sys.stderr)

    # If batch mode is enabled, do batch summarization now
    if use_batch:
        if not quiet:
            print("Using Groq Batch API for file summarization...")

        # Collect file data for batch processing
        file_data_for_batch = []
        for file_path in files_to_process:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                if len(code) > 100:  # Only summarize substantial files
                    rel_path = str(file_path.relative_to(repo_path))
                    file_data_for_batch.append((rel_path, code))
            except Exception as e:
                if not quiet:
                    print(f"Error reading {file_path} for batch: {e}", file=sys.stderr)

        if file_data_for_batch:
            try:
                # Use batch summarization
                batch_summaries = batch_summarize_files(
                    file_data_for_batch, batch_client=batch_client, quiet=quiet
                )

                # Update file_summaries with batch results
                file_summaries.update(batch_summaries)

                # Remove old individual summary chunks and add batch summary chunks
                chunks = [
                    c
                    for c in chunks
                    if c.get("metadata", {}).get("level") != "file_summary"
                ]
                for rel_path, summary in batch_summaries.items():
                    summary_chunk = {
                        "text": f"FILE: {rel_path}\n{summary}",
                        "metadata": {
                            "file": rel_path,
                            "level": "file_summary",
                            "location": {"file": rel_path},
                        },
                    }
                    chunks.append(summary_chunk)

            except Exception as e:
                if not quiet:
                    print(
                        f"Batch summarization failed, keeping individual summaries: {e}",
                        file=sys.stderr,
                    )

    # Log batch processing completion
    file_processing_duration_ms = (time.time() - file_processing_start_time) * 1000
    log_event(
        event="batch_processing_complete",
        level="INFO",
        phase="indexing",
        component="file_processor",
        message="File processing completed",
        data={
            "total_files_processed": len(files_to_process),
            "total_chunks": len(chunks),
            "total_summaries": len(file_summaries),
            "errors": len(errors),
            "use_batch": use_batch,
        },
        duration_ms=file_processing_duration_ms,
    )

    return chunks, file_summaries


@profile
def generate_folder_summaries(
    file_summaries: Dict[str, str], chunks: List[Dict], quiet: bool = False
) -> List[Dict]:
    """Generate folder summaries and add to chunks using multithreading.

    Args:
        file_summaries: Dictionary mapping file paths to their summaries.
        chunks: List of existing chunks to append folder summaries to.
        quiet: If True, suppress progress bars.

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

    # Check if batch processing is configured
    batch_client = get_groq_batch_client()

    if USE_BATCH_FOR_INDEXING and batch_client:
        if not quiet:
            print("Using Groq Batch API for folder summarization...")

        try:
            # Use batch summarization for folders
            folder_summaries = batch_summarize_folders(
                folders_to_process, batch_client=batch_client, quiet=quiet
            )

            # Add batch folder summaries to chunks
            for folder, folder_sum in folder_summaries.items():
                folder_chunk = {
                    "text": f"FOLDER: {folder}\n{folder_sum}",
                    "metadata": {
                        "folder": folder,
                        "level": "folder_summary",
                        "location": {"folder": folder},
                    },
                }
                chunks.append(folder_chunk)

            return chunks

        except Exception as e:
            if not quiet:
                print(
                    f"Batch folder summarization failed, using individual processing: {e}",
                    file=sys.stderr,
                )

    # Fall back to individual processing (or if batch is disabled)
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

        # Use tqdm to track progress (unless quiet mode)
        if quiet:
            for future in as_completed(futures):
                collect_folder_result(future)
        else:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating folder summaries",
            ):
                collect_folder_result(future)

    return chunks
