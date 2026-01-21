"""Pattern matching utilities for file filtering with .gitignore-style patterns."""

from pathlib import Path
from typing import List, Set, Tuple
import pathspec


class FilePatternMatcher:
    """Two-tier matcher: exact matches + gitignore patterns."""

    def __init__(self, patterns: List[str]):
        self.exact_matches, pattern_list = self._split_patterns(patterns)
        self.pattern_spec = (
            pathspec.PathSpec.from_lines("gitwildmatch", pattern_list)
            if pattern_list
            else None
        )

    def _split_patterns(self, patterns: List[str]) -> Tuple[Set[str], List[str]]:
        """Separate exact matches from wildcard patterns for performance."""
        exact = set()
        wildcards = []

        for pattern in patterns:
            if any(char in pattern for char in ["*", "?", "[", "!"]) or "/" in pattern:
                wildcards.append(pattern)
            else:
                exact.add(pattern)

        return exact, wildcards

    def matches(self, path: str) -> bool:
        """Check if path should be ignored."""
        path_obj = Path(path)

        # Fast path: exact filename match
        if path_obj.name in self.exact_matches:
            return True

        # Fast path: exact directory component match
        if any(part in self.exact_matches for part in path_obj.parts):
            return True

        # Pattern path: gitignore-style matching
        if self.pattern_spec:
            normalized = path.replace("\\", "/")
            return self.pattern_spec.match_file(normalized)

        return False
