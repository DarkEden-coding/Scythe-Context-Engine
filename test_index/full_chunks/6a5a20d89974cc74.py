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