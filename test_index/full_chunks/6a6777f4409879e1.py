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