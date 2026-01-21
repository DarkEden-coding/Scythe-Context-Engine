"""
Groq Batch API client for asynchronous bulk processing.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import requests
from requests.adapters import HTTPAdapter

from utils.logger import log_event

BatchStatus = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
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
    endpoint: str
    completion_window: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchJob":
        """Create BatchJob from API response dictionary."""
        return cls(
            id=data["id"],
            status=data["status"],
            input_file_id=data["input_file_id"],
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
            request_counts=data.get("request_counts", {}),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            completed_at=data.get("completed_at"),
            endpoint=data["endpoint"],
            completion_window=data["completion_window"],
        )


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
        """Initialize the Groq Batch API client.

        Args:
            api_key: Authentication key for Groq API.
            timeout_seconds: Request timeout in seconds.
            session: Optional requests session for reuse.
        """
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

        if session:
            self.session = session
        else:
            self.session = requests.Session()
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    # === File Operations ===

    def upload_batch_file(
        self, jsonl_content: str, filename: str = "batch.jsonl"
    ) -> str:
        """Upload a JSONL file for batch processing.

        Args:
            jsonl_content: JSONL formatted string with batch requests.
            filename: Name for the uploaded file.

        Returns:
            File ID for use in batch creation.

        Raises:
            GroqBatchError: If upload fails.
        """
        url = f"{self.API_BASE}/files"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Create a temporary file-like object
        files = {
            "file": (filename, jsonl_content.encode("utf-8"), "application/jsonl"),
            "purpose": (None, "batch"),
        }

        try:
            response = self.session.post(
                url, headers=headers, files=files, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise GroqBatchError(f"File upload request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GroqBatchError(
                f"File upload failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqBatchError("File upload response is not valid JSON.") from exc

        file_id = data.get("id")
        if not file_id:
            raise GroqBatchError("File upload response missing 'id' field.")

        return file_id

    def download_file(self, file_id: str) -> str:
        """Download file content by ID.

        Args:
            file_id: ID of the file to download.

        Returns:
            File content as string.

        Raises:
            GroqBatchError: If download fails.
        """
        url = f"{self.API_BASE}/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = self.session.get(
                url, headers=headers, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise GroqBatchError(f"File download request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GroqBatchError(
                f"File download failed ({response.status_code}): {response.text}"
            )

        return response.text

    def delete_file(self, file_id: str) -> bool:
        """Delete a file by ID.

        Args:
            file_id: ID of the file to delete.

        Returns:
            True if deletion was successful.

        Raises:
            GroqBatchError: If deletion fails.
        """
        url = f"{self.API_BASE}/files/{file_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.session.delete(
                url, headers=headers, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise GroqBatchError(f"File deletion request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GroqBatchError(
                f"File deletion failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqBatchError("File deletion response is not valid JSON.") from exc

        return data.get("deleted", False)

    def list_files(self) -> List[Dict[str, Any]]:
        """List all uploaded files.

        Returns:
            List of file metadata dictionaries.

        Raises:
            GroqBatchError: If request fails.
        """
        url = f"{self.API_BASE}/files"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.session.get(
                url, headers=headers, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise GroqBatchError(f"List files request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GroqBatchError(
                f"List files failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqBatchError("List files response is not valid JSON.") from exc

        return data.get("data", [])

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
            completion_window: Time frame for processing ("24h" to "7d").
            metadata: Optional key-value metadata.

        Returns:
            BatchJob with initial status.

        Raises:
            GroqBatchError: If batch creation fails.
        """
        url = f"{self.API_BASE}/batches"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": completion_window,
        }

        if metadata:
            payload["metadata"] = metadata

        try:
            response = self.session.post(
                url, headers=headers, json=payload, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            log_event(
                event="batch_job_error",
                level="ERROR",
                phase="indexing",
                component="groq_batch_client",
                message=f"Batch creation request failed: {str(exc)}",
                error=exc,
            )
            raise GroqBatchError(f"Batch creation request failed: {exc}") from exc

        if response.status_code >= 400:
            log_event(
                event="batch_job_error",
                level="ERROR",
                phase="indexing",
                component="groq_batch_client",
                message=f"Batch creation failed with status {response.status_code}",
                data={
                    "status_code": response.status_code,
                },
            )
            raise GroqBatchError(
                f"Batch creation failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            log_event(
                event="batch_job_error",
                level="ERROR",
                phase="indexing",
                component="groq_batch_client",
                message="Batch creation response is not valid JSON",
                error=exc,
            )
            raise GroqBatchError("Batch creation response is not valid JSON.") from exc

        batch_job = BatchJob.from_dict(data)

        log_event(
            event="batch_job_created",
            level="INFO",
            phase="indexing",
            component="groq_batch_client",
            message="Batch job created successfully",
            data={
                "job_id": batch_job.id,
                "status": batch_job.status,
                "completion_window": completion_window,
                "expires_at": batch_job.expires_at,
            },
        )

        return batch_job

    def get_batch(self, batch_id: str) -> BatchJob:
        """Get current batch job status.

        Args:
            batch_id: ID of the batch job.

        Returns:
            Current BatchJob state.

        Raises:
            GroqBatchError: If request fails.
        """
        url = f"{self.API_BASE}/batches/{batch_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.session.get(
                url, headers=headers, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            log_event(
                event="batch_job_error",
                level="ERROR",
                phase="indexing",
                component="groq_batch_client",
                message=f"Get batch request failed: {str(exc)}",
                error=exc,
            )
            raise GroqBatchError(f"Get batch request failed: {exc}") from exc

        if response.status_code >= 400:
            log_event(
                event="batch_job_error",
                level="ERROR",
                phase="indexing",
                component="groq_batch_client",
                message=f"Get batch failed with status {response.status_code}",
                data={
                    "batch_id": batch_id,
                    "status_code": response.status_code,
                },
            )
            raise GroqBatchError(
                f"Get batch failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            log_event(
                event="batch_job_error",
                level="ERROR",
                phase="indexing",
                component="groq_batch_client",
                message="Get batch response is not valid JSON",
                error=exc,
            )
            raise GroqBatchError("Get batch response is not valid JSON.") from exc

        return BatchJob.from_dict(data)

    def list_batches(self) -> List[BatchJob]:
        """List all batch jobs.

        Returns:
            List of BatchJob objects.

        Raises:
            GroqBatchError: If request fails.
        """
        url = f"{self.API_BASE}/batches"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.session.get(
                url, headers=headers, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise GroqBatchError(f"List batches request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GroqBatchError(
                f"List batches failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqBatchError("List batches response is not valid JSON.") from exc

        batches_data = data.get("data", [])
        return [BatchJob.from_dict(batch) for batch in batches_data]

    def cancel_batch(self, batch_id: str) -> BatchJob:
        """Cancel a running batch job.

        Args:
            batch_id: ID of the batch to cancel.

        Returns:
            Updated BatchJob with cancelling status.

        Raises:
            GroqBatchError: If cancellation fails.
        """
        url = f"{self.API_BASE}/batches/{batch_id}/cancel"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.session.post(
                url, headers=headers, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise GroqBatchError(f"Cancel batch request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GroqBatchError(
                f"Cancel batch failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqBatchError("Cancel batch response is not valid JSON.") from exc

        return BatchJob.from_dict(data)

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[BatchJob], None]] = None,
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
            GroqBatchError: If batch fails, expires, or times out.
        """
        start_time = time.time()

        while True:
            batch = self.get_batch(batch_id)

            if callback:
                callback(batch)

            # Check terminal states
            if batch.status == "completed":
                return batch
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise GroqBatchError(
                    f"Batch {batch_id} ended with status: {batch.status}"
                )

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise GroqBatchError(
                        f"Batch {batch_id} timed out after {elapsed:.1f}s"
                    )

            # Wait before next poll
            time.sleep(poll_interval)
