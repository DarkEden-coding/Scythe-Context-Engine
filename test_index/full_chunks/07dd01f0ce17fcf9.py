def clear_expired(self):
        """Manually clear all expired entries.

        This is optional - lazy cleanup happens automatically on get().
        """
        current_time = int(time.time())

        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                DELETE FROM cache WHERE expires_at < ?
            """, (current_time,))