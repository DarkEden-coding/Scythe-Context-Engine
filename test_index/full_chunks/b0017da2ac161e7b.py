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