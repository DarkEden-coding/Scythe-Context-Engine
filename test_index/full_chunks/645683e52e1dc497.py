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