"""
File and folder summarization using LLM.
"""

import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config.config import (
    SUMMARIZATION_MODEL,
    build_structured_output_format,
    chat_completion,
    extract_chat_content,
    get_batch_config,
    get_groq_batch_client,
)
from groq_batch_client import GroqBatchClient, GroqBatchError
from groq_batch_formatter import (
    create_batch_jsonl,
    create_file_summary_batch,
    create_folder_summary_batch,
    extract_content_from_result,
    parse_batch_results,
)
from .models import FileSummary, FolderSummary


def summarize_file(code: str, file_path: str) -> str:
    """Generate file summary via LLM.

    Args:
        code: The source code content as a string.
        file_path: Relative path to the file being summarized.

    Returns:
        One to two sentence summary of the file's purpose and key components.
    """
    try:
        prompt = f"""Summarize this {Path(file_path).suffix} file in 1-2 sentences based on the code provided.

Focus on: main purpose, key functions/classes, and specific technologies or patterns you can identify.
Be factual - only describe what you actually see in the code.

{code[:3500]}

Summary:"""

        response_format = build_structured_output_format(
            FileSummary.model_json_schema(), schema_name="file_summary"
        )
        resp = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=SUMMARIZATION_MODEL,
            response_format=response_format,
            options={"temperature": 0.3},
        )

        message_content = extract_chat_content(resp)
        if message_content:
            try:
                summary_data = FileSummary.model_validate_json(message_content)
                return summary_data.summary
            except Exception:
                # If JSON parsing fails, use the raw response as summary
                # Remove common prefixes that might indicate non-JSON response
                cleaned_content = message_content.strip()
                if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
                    cleaned_content = cleaned_content[1:-1]
                return cleaned_content
        else:
            return f"File: {Path(file_path).name} (summary failed: empty response)"

    except Exception:
        return (
            f"File: {Path(file_path).name} (summary failed: {traceback.format_exc()})"
        )


def summarize_folder(file_summaries: List[tuple]) -> str:
    """Aggregate file summaries into folder overview.

    Args:
        file_summaries: List of tuples containing (file_path, summary) pairs.

    Returns:
        One sentence description of the folder's purpose.
    """
    if not file_summaries:
        return "Empty folder"

    try:
        # Limit to first 8 files and format as list
        limited_summaries = file_summaries[:8]
        formatted_list = "\n".join(
            [f"- {Path(p).name}: {s}" for p, s in limited_summaries]
        )

        prompt = f"""Summarize this folder from file overviews (1 sentence):

{formatted_list}

Provide the folder purpose."""

        response_format = build_structured_output_format(
            FolderSummary.model_json_schema(), schema_name="folder_summary"
        )
        resp = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=SUMMARIZATION_MODEL,
            response_format=response_format,
            options={"temperature": 0.3},
        )

        message_content = extract_chat_content(resp)
        if message_content:
            try:
                folder_data = FolderSummary.model_validate_json(message_content)
                return folder_data.purpose
            except Exception:
                # If JSON parsing fails, use the raw response as summary
                # Remove common prefixes that might indicate non-JSON response
                cleaned_content = message_content.strip()
                if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
                    cleaned_content = cleaned_content[1:-1]
                return cleaned_content
        else:
            return "Multiple code files"

    except Exception:
        return "Multiple code files"


def batch_summarize_files(
    file_data: List[Tuple[str, str]],
    batch_client: Optional[GroqBatchClient] = None,
    model: Optional[str] = None,
    callback: Optional[Callable] = None,
    quiet: bool = False,
) -> Dict[str, str]:
    """Summarize multiple files using Groq Batch API.

    Args:
        file_data: List of (file_path, code) tuples.
        batch_client: Configured GroqBatchClient (uses config default if None).
        model: Groq model to use (uses config default if None).
        callback: Optional progress callback called with BatchJob on each poll.
        quiet: If True, suppress progress messages.

    Returns:
        Dict mapping file_path to summary string.

    Raises:
        GroqBatchError: If batch processing fails.
    """
    if not file_data:
        return {}

    # Use configured client if not provided
    if batch_client is None:
        batch_client = get_groq_batch_client()

    if batch_client is None:
        raise ValueError("Groq batch client not configured")

    # Get configuration
    batch_config = get_batch_config()
    if model is None:
        model = batch_config.get("chat_model", "llama-3.3-70b-versatile")

    completion_window = batch_config.get("batch_completion_window", "24h")
    poll_interval = batch_config.get("poll_interval_seconds", 30)

    if not quiet:
        print(f"Preparing batch summarization for {len(file_data)} files...")

    try:
        # 1. Format all requests
        response_format = build_structured_output_format(
            FileSummary.model_json_schema(), schema_name="file_summary"
        )

        requests = create_file_summary_batch(
            file_data, model=model, response_format=response_format, temperature=0.3
        )

        # 2. Create and upload batch
        if not quiet:
            print("Uploading batch file...")
        jsonl = create_batch_jsonl(requests)
        file_id = batch_client.upload_batch_file(jsonl, filename="file_summaries.jsonl")

        # 3. Create batch job
        if not quiet:
            print(f"Creating batch job (completion window: {completion_window})...")
        batch = batch_client.create_batch(
            file_id,
            completion_window=completion_window,
            metadata={"type": "file_summarization", "count": len(file_data)},
        )

        if not quiet:
            print(f"Batch job created: {batch.id}")
            print(f"Status: {batch.status}")
            print("Waiting for completion (this may take several minutes to hours)...")

        # 4. Wait for completion with progress updates
        def progress_callback(job):
            if callback:
                callback(job)
            if not quiet:
                completed = job.request_counts.get("completed", 0)
                total = job.request_counts.get("total", len(file_data))
                print(
                    f"Status: {job.status} | Progress: {completed}/{total}",
                    file=sys.stderr,
                )

        completed = batch_client.wait_for_batch(
            batch.id, poll_interval=poll_interval, callback=progress_callback
        )

        if not quiet:
            print(f"Batch completed! Processing results...")

        # 5. Download and parse results
        if not completed.output_file_id:
            raise GroqBatchError("Batch completed but no output file available")

        output_content = batch_client.download_file(completed.output_file_id)
        results = parse_batch_results(output_content)

        # 6. Map back to file paths and extract summaries
        summaries = {}
        failed_count = 0

        for i, (file_path, _) in enumerate(file_data):
            custom_id = f"file_{i}_{file_path}"

            if custom_id in results:
                result = results[custom_id]

                if "error" in result:
                    summaries[file_path] = f"File: {Path(file_path).name} (batch error)"
                    failed_count += 1
                else:
                    content = extract_content_from_result(result)
                    if content:
                        try:
                            summary_data = FileSummary.model_validate_json(content)
                            summaries[file_path] = summary_data.summary
                        except Exception:
                            # Use raw content if JSON parsing fails
                            cleaned = content.strip()
                            if cleaned.startswith('"') and cleaned.endswith('"'):
                                cleaned = cleaned[1:-1]
                            summaries[file_path] = cleaned
                    else:
                        summaries[file_path] = (
                            f"File: {Path(file_path).name} (empty response)"
                        )
                        failed_count += 1
            else:
                summaries[file_path] = f"File: {Path(file_path).name} (missing result)"
                failed_count += 1

        # 7. Cleanup
        if not quiet:
            print("Cleaning up batch files...")
        try:
            batch_client.delete_file(file_id)
            if completed.output_file_id:
                batch_client.delete_file(completed.output_file_id)
            if completed.error_file_id:
                batch_client.delete_file(completed.error_file_id)
        except Exception as e:
            if not quiet:
                print(f"Warning: Failed to cleanup files: {e}", file=sys.stderr)

        if not quiet:
            success_count = len(summaries) - failed_count
            print(
                f"Batch summarization complete: {success_count}/{len(file_data)} successful"
            )

        return summaries

    except GroqBatchError as e:
        # Clean up on error if possible
        try:
            if "file_id" in locals():
                batch_client.delete_file(file_id)
        except Exception:
            pass
        raise GroqBatchError(f"Batch summarization failed: {e}") from e


def batch_summarize_folders(
    folder_data: List[Tuple[str, List[Tuple[str, str]]]],
    batch_client: Optional[GroqBatchClient] = None,
    model: Optional[str] = None,
    callback: Optional[Callable] = None,
    quiet: bool = False,
) -> Dict[str, str]:
    """Summarize multiple folders using Groq Batch API.

    Args:
        folder_data: List of (folder_path, file_summaries) tuples.
            file_summaries is a list of (file_path, summary) tuples.
        batch_client: Configured GroqBatchClient (uses config default if None).
        model: Groq model to use (uses config default if None).
        callback: Optional progress callback called with BatchJob on each poll.
        quiet: If True, suppress progress messages.

    Returns:
        Dict mapping folder_path to summary string.

    Raises:
        GroqBatchError: If batch processing fails.
    """
    if not folder_data:
        return {}

    # Use configured client if not provided
    if batch_client is None:
        batch_client = get_groq_batch_client()

    if batch_client is None:
        raise ValueError("Groq batch client not configured")

    # Get configuration
    batch_config = get_batch_config()
    if model is None:
        model = batch_config.get("chat_model", "llama-3.3-70b-versatile")

    completion_window = batch_config.get("batch_completion_window", "24h")
    poll_interval = batch_config.get("poll_interval_seconds", 30)

    if not quiet:
        print(f"Preparing batch summarization for {len(folder_data)} folders...")

    try:
        # 1. Format all requests
        response_format = build_structured_output_format(
            FolderSummary.model_json_schema(), schema_name="folder_summary"
        )

        requests = create_folder_summary_batch(
            folder_data, model=model, response_format=response_format, temperature=0.3
        )

        # 2. Create and upload batch
        jsonl = create_batch_jsonl(requests)
        file_id = batch_client.upload_batch_file(
            jsonl, filename="folder_summaries.jsonl"
        )

        # 3. Create batch job
        batch = batch_client.create_batch(
            file_id,
            completion_window=completion_window,
            metadata={"type": "folder_summarization", "count": len(folder_data)},
        )

        # 4. Wait for completion
        def progress_callback(job):
            if callback:
                callback(job)
            if not quiet:
                completed = job.request_counts.get("completed", 0)
                total = job.request_counts.get("total", len(folder_data))
                print(
                    f"Folder batch status: {job.status} | Progress: {completed}/{total}",
                    file=sys.stderr,
                )

        completed = batch_client.wait_for_batch(
            batch.id, poll_interval=poll_interval, callback=progress_callback
        )

        # 5. Download and parse results
        if not completed.output_file_id:
            raise GroqBatchError("Batch completed but no output file available")

        output_content = batch_client.download_file(completed.output_file_id)
        results = parse_batch_results(output_content)

        # 6. Map back to folder paths
        summaries = {}

        for i, (folder_path, _) in enumerate(folder_data):
            custom_id = f"folder_{i}_{folder_path}"

            if custom_id in results:
                result = results[custom_id]

                if "error" not in result:
                    content = extract_content_from_result(result)
                    if content:
                        try:
                            folder_summary = FolderSummary.model_validate_json(content)
                            summaries[folder_path] = folder_summary.purpose
                        except Exception:
                            cleaned = content.strip()
                            if cleaned.startswith('"') and cleaned.endswith('"'):
                                cleaned = cleaned[1:-1]
                            summaries[folder_path] = cleaned
                    else:
                        summaries[folder_path] = "Multiple code files"
                else:
                    summaries[folder_path] = "Multiple code files"
            else:
                summaries[folder_path] = "Multiple code files"

        # 7. Cleanup
        try:
            batch_client.delete_file(file_id)
            if completed.output_file_id:
                batch_client.delete_file(completed.output_file_id)
            if completed.error_file_id:
                batch_client.delete_file(completed.error_file_id)
        except Exception:
            pass

        return summaries

    except GroqBatchError as e:
        try:
            if "file_id" in locals():
                batch_client.delete_file(file_id)
        except Exception:
            pass
        raise GroqBatchError(f"Batch folder summarization failed: {e}") from e
