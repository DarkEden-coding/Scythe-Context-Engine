def clear_all(self):
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM cache")