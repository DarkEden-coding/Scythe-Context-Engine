"""
OpenRouter API client utilities.
"""

from typing import Any, Dict, List, Optional, Sequence

import requests


class OpenRouterError(Exception):
    """Raised when OpenRouter API requests fail."""


class OpenRouterClient:
    """Simple wrapper around the OpenRouter REST API."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: Authentication key for OpenRouter.
            api_base: Base URL for the OpenRouter API.
            timeout_seconds: Request timeout in seconds.
            session: Optional requests session for reuse.
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def embed_texts(self, texts: Sequence[str], model: str, options: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Iterable of text inputs.
            model: Embedding model identifier.
            options: Additional OpenRouter parameters.

        Returns:
            Embedding vectors corresponding to each input text.
        """
        payload = {"model": model, "input": list(texts)}
        if options:
            payload.update(options)
        response_json = self._post("/embeddings", payload)
        data = response_json.get("data")
        if not isinstance(data, list):
            raise OpenRouterError("Embeddings response missing data list.")
        embeddings: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise OpenRouterError("Embeddings response contained invalid item.")
            embeddings.append([float(value) for value in embedding])
        return embeddings

    def embed_single(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text.
            model: Embedding model identifier.

        Returns:
            Embedding vector for the provided text.
        """
        embeddings = self.embed_texts([text], model)
        return embeddings[0]

    def chat_completion(
        self,
        messages: Sequence[Dict[str, Any]],
        model: str,
        response_format: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: Conversation messages following OpenAI schema.
            model: Chat model identifier.
            response_format: Optional response format schema.
            options: Additional OpenRouter parameters.

        Returns:
            Parsed JSON response from the API.
        """
        payload: Dict[str, Any] = {"model": model, "messages": list(messages)}
        if response_format is not None:
            payload["response_format"] = response_format
        if options:
            payload.update(options)
        return self._post("/chat/completions", payload)

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
            options: Additional OpenRouter parameters.

        Returns:
            Generated text content.
        """
        messages = [{"role": "user", "content": prompt}]
        response_json = self.chat_completion(messages, model, options=options)
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenRouterError("Chat completion response missing choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise OpenRouterError("Chat completion choice missing message.")
        content = message.get("content")
        if not isinstance(content, str):
            raise OpenRouterError("Chat completion message missing content.")
        return content

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an authenticated POST request."""
        url = f"{self.api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self.session.post(
                url, headers=headers, json=payload, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise OpenRouterError(f"OpenRouter request error: {exc}") from exc
        if response.status_code >= 400:
            raise OpenRouterError(
                f"OpenRouter request failed ({response.status_code}): {response.text}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter response is not valid JSON.") from exc
