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