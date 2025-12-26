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