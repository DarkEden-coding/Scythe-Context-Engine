"""
File and folder summarization using LLM.
"""

import traceback
from pathlib import Path
from typing import List

from config.config import (
    SUMMARIZATION_MODEL,
    build_structured_output_format,
    chat_completion,
    extract_chat_content,
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