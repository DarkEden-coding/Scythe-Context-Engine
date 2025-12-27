"""
AST parsing and chunk extraction using tree-sitter.
"""

import traceback
from typing import Dict, List, Optional

import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_cpp
import tree_sitter_c
import tree_sitter_go
import tree_sitter_rust
from tree_sitter import Language, Parser, Node

from line_profiler import profile


# Language to tree-sitter module mapping
LANGUAGE_MODULES = {
    "python": tree_sitter_python,
    "javascript": tree_sitter_javascript,
    "typescript": tree_sitter_typescript,
    "java": tree_sitter_java,
    "cpp": tree_sitter_cpp,
    "c": tree_sitter_c,
    "go": tree_sitter_go,
    "rust": tree_sitter_rust,
}


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


@profile
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
            "python": [
                "function_definition",
                "class_definition",
                "decorated_definition",
            ],
            "javascript": [
                "function_declaration",
                "class_declaration",
                "method_definition",
                "export_statement",
                "lexical_declaration",
                "variable_declaration",
                "arrow_function",
            ],
            "typescript": [
                "function_declaration",
                "class_declaration",
                "method_definition",
                "export_statement",
                "lexical_declaration",
                "variable_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
                "arrow_function",
            ],
            "java": [
                "method_declaration",
                "class_declaration",
                "interface_declaration",
            ],
            "cpp": ["function_definition", "class_specifier", "struct_specifier"],
            "c": ["function_definition", "struct_specifier"],
            "go": ["function_declaration", "method_declaration", "type_declaration"],
            "rust": [
                "function_item",
                "impl_item",
                "struct_item",
                "trait_item",
                "enum_item",
            ],
        }

        target_types = set(
            targets.get(lang, ["function_definition", "class_definition"])
        )

        covered_ranges = []

        def traverse(node: Node):
            """Recursively traverse the tree to find target nodes."""
            # Special handling for export statements in JS/TS
            # export_statement wraps the actual declaration, so unwrap it
            if node.type == "export_statement" and lang in ["javascript", "typescript"]:
                # Look for the declaration inside the export
                for child in node.children:
                    if child.type in [
                        "function_declaration",
                        "class_declaration",
                        "lexical_declaration",
                        "variable_declaration",
                    ]:
                        # Process the actual declaration with the export's full range
                        start_byte, end_byte = node.start_byte, node.end_byte
                        chunk_text = code[start_byte:end_byte]

                        if len(chunk_text.strip()) < 20:
                            continue

                        # Extract name from the child declaration
                        function_name = extract_function_name(child, code, lang)
                        docstring = extract_docstring(child, code, lang)

                        start_line = node.start_point[0] + 1
                        end_line = node.end_point[0] + 1

                        chunks.append(
                            {
                                "text": chunk_text,
                                "metadata": {
                                    "file": file_path,
                                    "start_line": start_line,
                                    "end_line": end_line,
                                    "type": child.type,
                                    "level": "code_chunk",
                                    "function_name": function_name,
                                    "docstring": docstring,
                                    "location": {
                                        "file": file_path,
                                        "start_line": start_line,
                                        "end_line": end_line,
                                    },
                                },
                            }
                        )
                        covered_ranges.append((start_line, end_line))
                        return  # Don't traverse further

                # If no declaration found, continue traversing
                for child in node.children:
                    traverse(child)
                return

            if node.type in target_types:
                start_byte, end_byte = node.start_byte, node.end_byte
                chunk_text = code[start_byte:end_byte]

                # Skip very small chunks (e.g. empty declarations)
                if len(chunk_text.strip()) < 20:
                    for child in node.children:
                        traverse(child)
                    return

                function_name = extract_function_name(node, code, lang)
                docstring = extract_docstring(node, code, lang)

                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "file": file_path,
                            "start_line": start_line,
                            "end_line": end_line,
                            "type": node.type,
                            "level": "code_chunk",
                            "function_name": function_name,
                            "docstring": docstring,
                            "location": {
                                "file": file_path,
                                "start_line": start_line,
                                "end_line": end_line,
                            },
                        },
                    }
                )
                covered_ranges.append((start_line, end_line))
                # Do not traverse children of a target node to avoid nested chunks
                # (unless it's a class, where we might want methods - but current logic
                # handles methods separately if they are targets)
                if "class" not in node.type:
                    return

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        # Fill gaps with line windows
        lines = code.split("\n")
        total_lines = len(lines)

        # Sort and merge covered ranges
        covered_ranges.sort()
        merged_ranges = []
        if covered_ranges:
            curr_start, curr_end = covered_ranges[0]
            for next_start, next_end in covered_ranges[1:]:
                if next_start <= curr_end + 1:
                    curr_end = max(curr_end, next_end)
                else:
                    merged_ranges.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_ranges.append((curr_start, curr_end))

        # Find gaps
        gaps = []
        last_end = 0
        for start, end in merged_ranges:
            if start > last_end + 1:
                gaps.append((last_end + 1, start - 1))
            last_end = end
        if last_end < total_lines:
            gaps.append((last_end + 1, total_lines))

        # Add chunks for gaps
        for gap_start, gap_end in gaps:
            gap_size = gap_end - gap_start + 1
            if gap_size <= 0:
                continue

            # For small gaps, just add one chunk
            # For large gaps, split into windows
            window_size = 50
            for i in range(gap_start, gap_end + 1, window_size):
                win_end = min(i + window_size - 1, gap_end)
                chunk_text = "\n".join(lines[i - 1 : win_end])

                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": {
                                "file": file_path,
                                "start_line": i,
                                "end_line": win_end,
                                "type": "gap_window",
                                "level": "code_chunk",
                                "function_name": "top-level",
                                "location": {
                                    "file": file_path,
                                    "start_line": i,
                                    "end_line": win_end,
                                },
                            },
                        }
                    )

        return chunks

    except Exception:
        print(f"Parse error {file_path}: {traceback.format_exc()}")
        return []
