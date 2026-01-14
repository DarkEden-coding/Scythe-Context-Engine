# Groq Batch API Implementation Plan

## Overview

This document outlines the implementation plan for adding Groq batch request support as an alternative provider option alongside the existing OpenRouter integration in the Scythe Context Engine.

## Current Architecture Analysis

### Existing Provider Pattern

The codebase currently supports two providers via [`config/config.py`](config/config.py:1):

- **OpenRouter**: Remote API with embeddings and chat completions
- **Ollama**: Local inference server

The provider abstraction uses:

- [`ProviderType`](config/config.py:12) literal type for provider selection
- [`OpenRouterClient`](openrouter_client.py:12) class for API interactions
- Provider-agnostic wrapper functions: [`embed_texts()`](config/config.py:68), [`chat_completion()`](config/config.py:82), [`generate_text()`](config/config.py:107)

### Current Batch Processing

Currently, batch processing is handled **synchronously with threading**:

- [`embedder.py`](indexer/embedder.py:1) uses `ThreadPoolExecutor` with 32 workers
- [`embed_batch_with_retry()`](indexer/embedder.py:18) processes batches with exponential backoff
- Summarization in [`file_processor.py`](indexer/file_processor.py:1) uses 8 threads

### Key Insight: Groq Batch API Difference

The Groq Batch API is **asynchronous** and fundamentally different:

1. Upload a JSONL file with all requests
2. Create a batch job (returns immediately)
3. Poll for completion (24h-7d completion window)
4. Download results from output file

This is ideal for **cost savings** (typically 50% discount) but not for real-time processing.

---

## Implementation Plan

### Phase 1: Create Groq Client Module

#### New File: `groq_batch_client.py`

```python
"""
Groq Batch API client for asynchronous bulk processing.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path
import json
import time
import requests
from requests.adapters import HTTPAdapter

BatchStatus = Literal[
    "validating", "failed", "in_progress", "finalizing",
    "completed", "expired", "cancelling", "cancelled"
]

@dataclass
class BatchJob:
    """Represents a Groq batch job."""
    id: str
    status: BatchStatus
    input_file_id: str
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    request_counts: Dict[str, int]
    created_at: int
    expires_at: int
    completed_at: Optional[int]


class GroqBatchError(Exception):
    """Raised when Groq Batch API requests fail."""


class GroqBatchClient:
    """Client for Groq Batch API operations."""

    API_BASE = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

        if session:
            self.session = session
        else:
            self.session = requests.Session()
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
            self.session.mount("https://", adapter)

    # === File Operations ===

    def upload_batch_file(self, jsonl_content: str, filename: str = "batch.jsonl") -> str:
        """Upload a JSONL file for batch processing.

        Args:
            jsonl_content: JSONL formatted string with batch requests.
            filename: Name for the uploaded file.

        Returns:
            File ID for use in batch creation.
        """
        # Implementation details...

    def download_file(self, file_id: str) -> str:
        """Download file content by ID.

        Returns:
            File content as string.
        """

    def delete_file(self, file_id: str) -> bool:
        """Delete a file by ID."""

    # === Batch Operations ===

    def create_batch(
        self,
        input_file_id: str,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Create and start a batch job.

        Args:
            input_file_id: ID of uploaded JSONL file.
            completion_window: "24h" to "7d".
            metadata: Optional key-value metadata.

        Returns:
            BatchJob with initial status.
        """

    def get_batch(self, batch_id: str) -> BatchJob:
        """Get current batch job status."""

    def list_batches(self) -> List[BatchJob]:
        """List all batch jobs."""

    def cancel_batch(self, batch_id: str) -> BatchJob:
        """Cancel a running batch job."""

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        callback: Optional[callable] = None,
    ) -> BatchJob:
        """Poll until batch completes or fails.

        Args:
            batch_id: Batch job ID.
            poll_interval: Seconds between status checks.
            timeout: Max seconds to wait (None = no limit).
            callback: Optional function called on each poll with BatchJob.

        Returns:
            Completed BatchJob.

        Raises:
            GroqBatchError: If batch fails or times out.
        """
```

### Phase 2: Batch Request Formatter

#### New File: `groq_batch_formatter.py`

```python
"""
Utilities for formatting requests for Groq Batch API.
"""

import json
from typing import Any, Dict, List, Sequence

def format_chat_completion_request(
    custom_id: str,
    messages: Sequence[Dict[str, Any]],
    model: str,
    response_format: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Format a single chat completion request for batch.

    Args:
        custom_id: Unique identifier for correlating responses.
        messages: Chat messages.
        model: Groq model ID.
        response_format: Optional JSON schema format.
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        JSONL-ready request dict.
    """
    body = {
        "model": model,
        "messages": list(messages),
    }
    if response_format:
        body["response_format"] = response_format
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
        requests: List of formatted request dicts.

    Returns:
        JSONL string ready for upload.
    """
    return "\n".join(json.dumps(req) for req in requests)


def parse_batch_results(jsonl_content: str) -> Dict[str, Any]:
    """Parse batch output JSONL into dict keyed by custom_id.

    Args:
        jsonl_content: JSONL output from completed batch.

    Returns:
        Dict mapping custom_id to response body.
    """
    results = {}
    for line in jsonl_content.strip().split("\n"):
        if not line:
            continue
        result = json.loads(line)
        custom_id = result.get("custom_id")
        if result.get("error"):
            results[custom_id] = {"error": result["error"]}
        else:
            results[custom_id] = result.get("response", {}).get("body", {})
    return results
```

### Phase 3: Integrate with Summarization Pipeline

#### Modify: `indexer/summarizer.py`

Add batch summarization support:

```python
# New function for batch summarization
def batch_summarize_files(
    file_data: List[tuple[str, str]],  # (file_path, code)
    batch_client: GroqBatchClient,
    model: str,
    callback: Optional[callable] = None,
) -> Dict[str, str]:
    """Summarize multiple files using Groq Batch API.

    Args:
        file_data: List of (file_path, code) tuples.
        batch_client: Configured GroqBatchClient.
        model: Groq model to use.
        callback: Optional progress callback.

    Returns:
        Dict mapping file_path to summary.
    """
    # 1. Format all requests
    requests = []
    for i, (file_path, code) in enumerate(file_data):
        prompt = _build_file_summary_prompt(code, file_path)
        req = format_chat_completion_request(
            custom_id=f"file_{i}_{file_path}",
            messages=[{"role": "user", "content": prompt}],
            model=model,
            response_format=build_structured_output_format(
                FileSummary.model_json_schema(), "file_summary"
            ),
            temperature=0.3,
        )
        requests.append(req)

    # 2. Create and upload batch
    jsonl = create_batch_jsonl(requests)
    file_id = batch_client.upload_batch_file(jsonl)
    batch = batch_client.create_batch(file_id)

    # 3. Wait for completion
    completed = batch_client.wait_for_batch(
        batch.id,
        callback=callback,
    )

    # 4. Download and parse results
    output_content = batch_client.download_file(completed.output_file_id)
    results = parse_batch_results(output_content)

    # 5. Map back to file paths
    summaries = {}
    for i, (file_path, _) in enumerate(file_data):
        custom_id = f"file_{i}_{file_path}"
        if custom_id in results and "error" not in results[custom_id]:
            content = extract_chat_content(results[custom_id])
            try:
                summary_data = FileSummary.model_validate_json(content)
                summaries[file_path] = summary_data.summary
            except Exception:
                summaries[file_path] = content.strip()
        else:
            summaries[file_path] = f"File: {Path(file_path).name} (batch failed)"

    # 6. Cleanup
    batch_client.delete_file(file_id)
    if completed.output_file_id:
        batch_client.delete_file(completed.output_file_id)

    return summaries
```

### Phase 4: Update Configuration

#### Modify: `config/config.py`

```python
# Add Groq to provider types
ProviderType = Literal["openrouter", "ollama", "groq"]

# Add Groq configuration section in config.json:
{
    "groq": {
        "api_key": "gsk_...",
        "chat_model": "llama-3.3-70b-versatile",
        "batch_completion_window": "24h",
        "use_batch_for_indexing": true,
        "poll_interval_seconds": 30
    }
}

# Add Groq client initialization
_groq_batch_client: Optional[GroqBatchClient] = None
if _config.get("groq", {}).get("api_key"):
    _groq_batch_client = GroqBatchClient(
        api_key=_config["groq"]["api_key"]
    )
```

#### Modify: `config/create_config.py`

Add Groq option to the interactive setup:

```python
DEFAULTS = {
    # ... existing ...
    "groq": {
        "chat_model": "llama-3.3-70b-versatile",
        "batch_completion_window": "24h",
        "use_batch_for_indexing": True,
        "poll_interval_seconds": 30,
    },
}

# In main():
while provider not in ["openrouter", "ollama", "groq"]:
    provider = input("Select provider (openrouter/ollama/groq) [openrouter]: ")...

if provider == "groq":
    key = ""
    while not key:
        key = input("Enter Groq API key: ").strip()

    p_def = DEFAULTS["groq"]
    config["groq"] = {
        "api_key": key,
        "chat_model": ask("Chat Model", p_def["chat_model"], skip),
        "batch_completion_window": ask("Batch Window (24h-7d)", p_def["batch_completion_window"], skip),
        "use_batch_for_indexing": ask("Use batch API for indexing?", p_def["use_batch_for_indexing"], skip),
        "poll_interval_seconds": int(ask("Poll Interval", p_def["poll_interval_seconds"], skip)),
    }
```

### Phase 5: Hybrid Processing Mode

The key architectural decision is **when to use batch vs. real-time**:

| Operation              | Batch (Groq)                             | Real-time (OpenRouter/Ollama) |
| ---------------------- | ---------------------------------------- | ----------------------------- |
| Indexing summarization | ✅ Ideal (many files, not time-critical) | ✅ Works but slower           |
| Folder summarization   | ✅ Can batch                             | ✅ Works                      |
| Query refinement       | ❌ Too slow                              | ✅ Required                   |
| Interactive queries    | ❌ Not suitable                          | ✅ Required                   |

#### Recommended Hybrid Architecture

```python
# In config.json, support mixed providers:
{
    "provider": "openrouter",  # Primary for real-time
    "batch_provider": "groq",   # Optional batch provider for indexing
    "groq": {
        "api_key": "...",
        "use_for_indexing": true,
    },
    "openrouter": {
        "api_key": "...",
        # Used for queries and real-time ops
    }
}
```

### Phase 6: Modify Indexing Pipeline

#### Update: `indexer/file_processor.py`

```python
def process_files_with_batch(
    files_to_process: List[Path],
    repo_path: str,
    output_prefix: Optional[str] = None,
    quiet: bool = False,
    batch_client: Optional[GroqBatchClient] = None,
) -> tuple:
    """Process files with optional batch summarization.

    If batch_client is provided, uses Groq Batch API for summarization.
    Otherwise falls back to threaded real-time summarization.
    """
    # Phase 1: Extract chunks (always local, fast)
    chunks = []
    file_data_for_summary = []

    for file_path in files_to_process:
        # ... extract chunks as before ...
        if len(code) > 100:
            file_data_for_summary.append((rel_path, code))

    # Phase 2: Summarize files
    if batch_client and file_data_for_summary:
        # Use batch API
        summaries = batch_summarize_files(
            file_data_for_summary,
            batch_client,
            model=config["groq"]["chat_model"],
            callback=lambda b: print(f"Batch status: {b.status}") if not quiet else None,
        )
    else:
        # Use existing threaded approach
        summaries = _threaded_summarize_files(file_data_for_summary)

    # ... rest of processing ...
```

---

## File Structure After Implementation

```
Scythe-Context-Engine/
├── config/
│   ├── config.py          # Updated with Groq support
│   └── create_config.py   # Updated with Groq option
├── groq_batch_client.py   # NEW: Groq Batch API client
├── groq_batch_formatter.py # NEW: Request formatting utilities
├── openrouter_client.py   # Existing (unchanged)
├── indexer/
│   ├── summarizer.py      # Updated with batch_summarize_files()
│   ├── file_processor.py  # Updated with batch processing option
│   └── embedder.py        # Unchanged (embeddings still use OpenRouter/Ollama)
└── ...
```

---

## API Compatibility Notes

### Groq Batch API Specifics

1. **File Size Limit**: 100 MB max for input JSONL
2. **Completion Window**: 24h to 7d (required parameter)
3. **Endpoint Support**: Only `/v1/chat/completions` currently
4. **No Embeddings**: Groq Batch API does not support embeddings - continue using OpenRouter/Ollama for this

### Request Format

Each line in the JSONL must be:

```json
{
  "custom_id": "unique-request-id",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "llama-3.3-70b-versatile",
    "messages": [{ "role": "user", "content": "..." }],
    "temperature": 0.3
  }
}
```

### Response Format

Output JSONL contains:

```json
{
  "id": "response-id",
  "custom_id": "unique-request-id",
  "response": {
    "status_code": 200,
    "body": {
      "choices": [{ "message": { "content": "..." } }]
    }
  }
}
```

---

## Implementation Checklist

- [ ] Create `groq_batch_client.py` with full API implementation
- [ ] Create `groq_batch_formatter.py` for JSONL handling
- [ ] Add Groq to `ProviderType` in `config/config.py`
- [ ] Add Groq configuration section handling
- [ ] Update `config/create_config.py` with Groq setup wizard
- [ ] Add `batch_summarize_files()` to `indexer/summarizer.py`
- [ ] Update `process_files()` to support batch mode
- [ ] Add CLI flag `--use-batch` to `index_repo.py`
- [ ] Add progress reporting for batch jobs
- [ ] Handle batch errors and partial failures gracefully
- [ ] Add cleanup for uploaded/output files
- [ ] Write unit tests for batch client
- [ ] Update README.md with Groq batch documentation

---

## Cost/Performance Trade-offs

| Approach             | Cost             | Latency      | Best For                    |
| -------------------- | ---------------- | ------------ | --------------------------- |
| OpenRouter real-time | $$               | Low          | Queries, small repos        |
| Ollama local         | Free (compute)   | Medium       | Development, privacy        |
| Groq Batch           | $ (50% discount) | High (hours) | Large repos, cost-sensitive |

### Recommended Usage Pattern

1. **Initial large repo index**: Use Groq Batch (cost-effective for thousands of files)
2. **Incremental updates**: Use real-time (only few files changed)
3. **Query processing**: Always real-time (user is waiting)

---

## Error Handling Strategy

```python
class BatchProcessingError(Exception):
    """Raised when batch processing fails."""
    def __init__(self, message: str, partial_results: Optional[Dict] = None):
        super().__init__(message)
        self.partial_results = partial_results

# In batch processing:
try:
    results = batch_summarize_files(...)
except BatchProcessingError as e:
    if e.partial_results:
        # Use what we got, fall back to real-time for rest
        remaining = [f for f in files if f not in e.partial_results]
        realtime_results = threaded_summarize_files(remaining)
        results = {**e.partial_results, **realtime_results}
    else:
        # Full fallback to real-time
        results = threaded_summarize_files(files)
```

---

## Migration Path

For existing users:

1. **No breaking changes**: Existing `openrouter` and `ollama` configs continue to work
2. **Opt-in**: Groq batch is only used if explicitly configured
3. **Graceful degradation**: If batch fails, falls back to real-time processing
