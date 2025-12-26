class Cache:
    """SQLite-based cache with lazy TTL expiration."""

    def __init__(self, db_path: str = "context_cache.db"):
        """Initialize cache with SQLite database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database and create cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )
            """)

            # Create index for efficient expiration cleanup
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
            """)

    def get(self, key: str) -> Optional[str]:
        """Retrieve cached value with lazy TTL check.

        Args:
            key: Cache key to retrieve.

        Returns:
            Cached value if found and not expired, None otherwise.
        """
        current_time = int(time.time())

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()

            # Get value and check expiration
            cursor.execute("""
                SELECT value, expires_at FROM cache WHERE key = ?
            """, (key,))

            result = cursor.fetchone()

            if result is None:
                return None

            value, expires_at = result

            # Check if expired
            if current_time > expires_at:
                # Lazy cleanup - remove expired entry
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                connection.commit()
                return None

            return value

    def set(self, key: str, value: str, ttl: int):
        """Store value in cache with expiration timestamp.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds from now.
        """
        expires_at = int(time.time()) + ttl

        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
            """, (key, value, expires_at))

    def clear_expired(self):
        """Manually clear all expired entries.

        This is optional - lazy cleanup happens automatically on get().
        """
        current_time = int(time.time())

        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                DELETE FROM cache WHERE expires_at < ?
            """, (current_time,))

    def clear_all(self):
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM cache")