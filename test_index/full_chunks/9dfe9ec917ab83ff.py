class ParserManager:
    """Manager for caching tree-sitter parsers and queries across file processing."""

    _parsers = {}
    _languages = {}
    _queries = {}

    @classmethod
    def get_parser(cls, lang: str):
        """Get or create a cached parser for the given language.

        Args:
            lang: Programming language identifier.

        Returns:
            Parser instance for the language, or None if language not supported.
        """
        if lang not in cls._parsers:
            if lang not in LANGUAGE_MODULES:
                return None

            module = LANGUAGE_MODULES[lang]
            cls._languages[lang] = Language(module.language())
            parser = Parser()
            parser.language = cls._languages[lang]
            cls._parsers[lang] = parser

        return cls._parsers[lang]

    @classmethod
    def get_query(cls, lang: str):
        """Get or create a cached query for the given language.

        Args:
            lang: Programming language identifier.

        Returns:
            Query instance for finding code chunks in the language.
        """
        if lang not in cls._queries:
            targets = {
                "python": ["function_definition", "class_definition"],
                "javascript": [
                    "function_declaration",
                    "class_declaration",
                    "method_definition",
                ],
                "typescript": [
                    "function_declaration",
                    "class_declaration",
                    "method_definition",
                ],
                "java": ["method_declaration", "class_declaration"],
                "cpp": ["function_definition", "class_specifier"],
                "c": ["function_definition"],
                "go": ["function_declaration", "method_declaration"],
                "rust": ["function_item", "impl_item"],
            }

            node_types = targets.get(lang, ["function_definition", "class_definition"])
            patterns = [f"({node_type}) @node" for node_type in node_types]
            query_string = "\n".join(patterns)
            cls._queries[lang] = cls._languages[lang].query(query_string)

        return cls._queries[lang]