def extract_function_name(node: Node, code: str, lang: str) -> str:
    """Extract the function name from a tree-sitter node.

    Args:
        node: Tree-sitter node representing a function/method/class.
        code: The source code content as a string.
        lang: Programming language identifier.

    Returns:
        The function name, or "unknown" if not found.
    """
    try:
        name_field_map = {
            "python": "name",
            "javascript": "name",
            "typescript": "name",
            "java": "name",
            "cpp": "name",
            "c": "declarator",
            "go": "name",
            "rust": "name",
        }

        name_field = name_field_map.get(lang, "name")
        name_node = node.child_by_field_name(name_field)

        if name_node:
            return code[name_node.start_byte : name_node.end_byte]

        for child in node.children:
            if child.type == "identifier":
                return code[child.start_byte : child.end_byte]

        return "unknown"
    except Exception:
        return "unknown"