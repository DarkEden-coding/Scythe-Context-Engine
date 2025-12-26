def __init__(self, db_path: str = "context_cache.db"):
        """Initialize cache with SQLite database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_db()