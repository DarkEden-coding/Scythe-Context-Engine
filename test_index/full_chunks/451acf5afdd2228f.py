def extract_docstring(node: Node, code: str, lang: str) -> Optional[str]:
    """Extract the docstring from a function node.

    Args:
        node: Tree-sitter node representing a function/method/class.
        code: The source code content as a string.
        lang: Programming language identifier.

    Returns:
        The docstring text, or None if not found.
    """
    try:
        if lang == "python":
            body_node = node.child_by_field_name("body")
            if body_node and body_node.child_count > 0:
                first_child = body_node.children[0]
                if first_child.type == "expression_statement":
                    string_node = (
                        first_child.children[0] if first_child.child_count > 0 else None
                    )
                    if string_node and string_node.type == "string":
                        docstring_text = code[
                            string_node.start_byte : string_node.end_byte
                        ]
                        return docstring_text.strip('"""').strip("'''").strip()

        elif lang in ["javascript", "typescript"]:
            for child in node.children:
                if child.type == "comment":
                    comment_text = code[child.start_byte : child.end_byte]
                    if comment_text.startswith("/**"):
                        return comment_text.strip("/**").strip("*/").strip()

        elif lang == "java":
            prev_sibling = node.prev_sibling
            if prev_sibling and prev_sibling.type == "comment":
                comment_text = code[prev_sibling.start_byte : prev_sibling.end_byte]
                if comment_text.startswith("/**"):
                    return comment_text.strip("/**").strip("*/").strip()

        return None
    except Exception:
        return None