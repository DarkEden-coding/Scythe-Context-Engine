"""
Groq API client for chat completions.
"""

import json
import time
from typing import Any, Dict, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter

from utils.logger import log_event


class GroqError(Exception):
    """Raised when Groq API requests fail."""


class GroqClient:
    """Client for Groq chat completion API."""

    API_BASE = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 60.0,
        session: Optional[requests.Session] = None,
        pool_size: int = 100,
    ) -> None:
        """Initialize the Groq client.

        Args:
            api_key: Authentication key for Groq API.
            timeout_seconds: Request timeout in seconds.
            session: Optional requests session for reuse.
            pool_size: Connection pool size for concurrent requests.
        """
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

        if session:
            self.session = session
        else:
            self.session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=pool_size,
                pool_maxsize=pool_size,
            )
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    def chat_completion(
        self,
        messages: Sequence[Dict[str, Any]],
        model: str,
        response_format: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request to Groq.

        Args:
            messages: Conversation messages following OpenAI schema.
            model: Chat model identifier.
            response_format: Optional response format schema.
            options: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            Parsed JSON response from the API.
        """
        payload: Dict[str, Any] = {"model": model, "messages": list(messages)}
        if response_format is not None:
            payload["response_format"] = response_format
        if options:
            payload.update(options)
        return self._post("/chat/completions", payload, options=options)

    def generate_text(
        self,
        prompt: str,
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text using a single-turn prompt.

        Args:
            prompt: User prompt string.
            model: Chat model identifier.
            options: Additional parameters.

        Returns:
            Generated text content.
        """
        messages = [{"role": "user", "content": prompt}]
        response_json = self.chat_completion(messages, model, options=options)
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GroqError("Chat completion response missing choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise GroqError("Chat completion choice missing message.")
        content = message.get("content")
        if not isinstance(content, str):
            raise GroqError("Chat completion message missing content.")
        return content

    def _post(
        self, path: str, payload: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute an authenticated POST request."""
        request_start_time = time.time()
        url = f"{self.API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Log LLM request
        try:
            request_size = len(json.dumps(payload))
            prompt_tokens_estimate = len(str(payload.get("messages", []))) // 4
            log_event(
                event="llm_request",
                level="INFO",
                phase="query",
                component="groq_client",
                message="Sending request to Groq API",
                data={
                    "provider": "groq",
                    "model": payload.get("model"),
                    "endpoint": path,
                    "request_size_bytes": request_size,
                    "prompt_tokens_estimate": prompt_tokens_estimate,
                    "options": options or {},
                },
            )
        except Exception:
            pass  # Silently ignore logging errors

        try:
            response = self.session.post(
                url, headers=headers, json=payload, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            request_duration_ms = (time.time() - request_start_time) * 1000
            log_event(
                event="llm_error",
                level="ERROR",
                phase="query",
                component="groq_client",
                message=f"Groq request error: {str(exc)}",
                duration_ms=request_duration_ms,
                error=exc,
            )
            raise GroqError(f"Groq request error: {exc}") from exc

        if response.status_code >= 400:
            request_duration_ms = (time.time() - request_start_time) * 1000
            log_event(
                event="llm_error",
                level="ERROR",
                phase="query",
                component="groq_client",
                message=f"Groq request failed with status {response.status_code}",
                data={
                    "status_code": response.status_code,
                },
                duration_ms=request_duration_ms,
            )
            raise GroqError(
                f"Groq request failed ({response.status_code}): {response.text}"
            )

        try:
            response_json = response.json()

            # Log LLM response
            request_duration_ms = (time.time() - request_start_time) * 1000
            try:
                completion_tokens = response_json.get("usage", {}).get("completion_tokens", 0)
                prompt_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
                total_tokens = response_json.get("usage", {}).get("total_tokens", 0)

                log_event(
                    event="llm_response",
                    level="INFO",
                    phase="query",
                    component="groq_client",
                    message="Received response from Groq API",
                    data={
                        "status_code": response.status_code,
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": total_tokens,
                    },
                    duration_ms=request_duration_ms,
                )
            except Exception:
                pass  # Silently ignore logging errors

            return response_json
        except ValueError as exc:
            request_duration_ms = (time.time() - request_start_time) * 1000
            log_event(
                event="llm_error",
                level="ERROR",
                phase="query",
                component="groq_client",
                message="Groq response is not valid JSON",
                duration_ms=request_duration_ms,
                error=exc,
            )
            raise GroqError("Groq response is not valid JSON.") from exc
