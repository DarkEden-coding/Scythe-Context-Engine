"""
Utilities for formatting requests for Groq Batch API.
"""

import json
from typing import Any, Dict, List, Optional, Sequence


def format_chat_completion_request(
    custom_id: str,
    messages: Sequence[Dict[str, Any]],
    model: str,
    response_format: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Format a single chat completion request for batch processing.

    Args:
        custom_id: Unique identifier for correlating responses.
        messages: Chat messages in OpenAI format.
        model: Groq model ID (e.g., "llama-3.3-70b-versatile").
        response_format: Optional JSON schema format for structured output.
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        JSONL-ready request dictionary.
    """
    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }

    if response_format:
        body["response_format"] = response_format

    # Add any additional parameters
    body.update(kwargs)

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def create_batch_jsonl(requests: List[Dict[str, Any]]) -> str:
    """Convert list of requests to JSONL string.

    Args:
        requests: List of formatted request dictionaries.

    Returns:
        JSONL string ready for upload to Groq Batch API.
    """
    if not requests:
        raise ValueError("Cannot create JSONL from empty request list")

    return "\n".join(json.dumps(req, ensure_ascii=False) for req in requests)


def parse_batch_results(jsonl_content: str) -> Dict[str, Any]:
    """Parse batch output JSONL into dict keyed by custom_id.

    Args:
        jsonl_content: JSONL output from completed batch job.

    Returns:
        Dict mapping custom_id to response body or error.

    Example output format:
        {
            "file_0_path/to/file.py": {
                "choices": [{"message": {"content": "..."}}],
                ...
            },
            "file_1_another.py": {
                "error": {"message": "Rate limit exceeded", "code": 429}
            }
        }
    """
    results = {}

    for line in jsonl_content.strip().split("\n"):
        if not line:
            continue

        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            # Skip malformed lines
            continue

        custom_id = result.get("custom_id")
        if not custom_id:
            continue

        # Check for errors in the result
        if result.get("error"):
            results[custom_id] = {"error": result["error"]}
        else:
            # Extract the response body
            response = result.get("response", {})
            body = response.get("body", {})
            results[custom_id] = body

    return results


def extract_content_from_result(result: Dict[str, Any]) -> Optional[str]:
    """Extract content from a batch result dictionary.

    Args:
        result: Single result from parse_batch_results.

    Returns:
        Content string if successful, None if error or missing.
    """
    if "error" in result:
        return None

    choices = result.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    return message.get("content")


def create_file_summary_batch(
    file_data: List[tuple],
    model: str,
    response_format: Optional[Dict[str, Any]] = None,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Create batch requests for file summarization.

    Args:
        file_data: List of (file_path, code) tuples.
        model: Groq model to use.
        response_format: Optional structured output format.
        temperature: Sampling temperature.

    Returns:
        List of formatted batch requests.
    """
    requests = []

    for i, (file_path, code) in enumerate(file_data):
        # Build the prompt
        prompt = f"""Summarize this file in 1-2 sentences based on the code provided.

Focus on: main purpose, key functions/classes, and specific technologies or patterns you can identify.
Be factual - only describe what you actually see in the code.

{code[:3500]}

Summary:"""

        # Create the request
        req = format_chat_completion_request(
            custom_id=f"file_{i}_{file_path}",
            messages=[{"role": "user", "content": prompt}],
            model=model,
            response_format=response_format,
            temperature=temperature,
        )

        requests.append(req)

    return requests


def create_folder_summary_batch(
    folder_data: List[tuple],
    model: str,
    response_format: Optional[Dict[str, Any]] = None,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Create batch requests for folder summarization.

    Args:
        folder_data: List of (folder_path, file_summaries) tuples.
            file_summaries is a list of (file_path, summary) tuples.
        model: Groq model to use.
        response_format: Optional structured output format.
        temperature: Sampling temperature.

    Returns:
        List of formatted batch requests.
    """
    requests = []

    for i, (folder_path, file_summaries) in enumerate(folder_data):
        # Limit to first 8 files
        limited_summaries = file_summaries[:8]

        # Format as list
        from pathlib import Path

        formatted_list = "\n".join(
            [f"- {Path(p).name}: {s}" for p, s in limited_summaries]
        )

        # Build the prompt
        prompt = f"""Summarize this folder from file overviews (1 sentence):

{formatted_list}

Provide the folder purpose."""

        # Create the request
        req = format_chat_completion_request(
            custom_id=f"folder_{i}_{folder_path}",
            messages=[{"role": "user", "content": prompt}],
            model=model,
            response_format=response_format,
            temperature=temperature,
        )

        requests.append(req)

    return requests
