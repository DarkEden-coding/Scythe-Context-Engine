def extract_chunks(code: str, lang: str, file_path: str) -> List[Dict]:
    """Extract functions/classes via tree-sitter queries - optimized.

    Args:
        code: The source code content as a string.
        lang: Programming language identifier.
        file_path: Relative path to the file being processed.

    Returns:
        List of chunk dictionaries containing code snippets and metadata.
    """
    try:
        # Get cached parser
        parser = ParserManager.get_parser(lang)
        if parser is None:
            return []

        # Parse once
        tree = parser.parse(bytes(code, "utf8"))

        chunks = []

        # Language-specific node types to extract
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

        target_types = set(
            targets.get(lang, ["function_definition", "class_definition"])
        )

        def traverse(node: Node):
            """Recursively traverse the tree to find target nodes."""
            if node.type in target_types:
                start_byte, end_byte = node.start_byte, node.end_byte
                chunk_text = code[start_byte:end_byte]

                function_name = extract_function_name(node, code, lang)
                docstring = extract_docstring(node, code, lang)

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "file": file_path,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "type": node.type,
                            "level": "code_chunk",
                            "function_name": function_name,
                            "docstring": docstring,
                            "location": {
                                "file": file_path,
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                            },
                        },
                    }
                )

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        # Fallback: line windows if no chunks found
        if not chunks:
            lines = code.split("\n")

            for i in range(0, len(lines), 30):  # 30-line windows
                chunk_text = "\n".join(lines[i : i + 30])

                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": {
                                "file": file_path,
                                "start_line": i + 1,  # Convert to 1-indexed
                                "end_line": min(i + 30, len(lines))
                                + 1,  # Convert to 1-indexed
                                "type": "line_window",
                                "level": "code_chunk",
                                "location": {
                                    "file": file_path,
                                    "start_line": i + 1,  # Convert to 1-indexed
                                    "end_line": min(i + 30, len(lines))
                                    + 1,  # Convert to 1-indexed
                                },
                            },
                        }
                    )

        return chunks

    except Exception:
        print(f"Parse error {file_path}: {traceback.format_exc()}")
        return []