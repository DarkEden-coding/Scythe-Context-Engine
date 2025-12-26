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