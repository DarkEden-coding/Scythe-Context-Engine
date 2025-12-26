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