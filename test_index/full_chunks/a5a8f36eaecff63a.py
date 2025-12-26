def process_single_file(file_path: Path, repo_path: str, output_prefix: Optional[str] = None) -> tuple:
    """Process a single file to extract chunks and summary.

    Args:
        file_path: Path to the file to process.
        repo_path: Root path of the repository.
        output_prefix: Directory prefix for output files (for saving full chunks).

    Returns:
        Tuple containing (chunks, file_summary, summary_chunk, error).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        rel_path = str(file_path.relative_to(repo_path))
        lang = SUPPORTED_LANGS[file_path.suffix]

        # Handle markdown files as full documents
        if lang == "markdown":
            file_chunks = [{
                "text": code,
                "metadata": {
                    "level": "document",
                    "file": rel_path,
                    "type": "markdown",
                    "location": {"file": rel_path},
                },
            }]
        else:
            # Extract code chunks for programming languages
            file_chunks = extract_chunks(code, lang, rel_path)
        
        # Process each chunk: generate chunk_id and save full content
        for chunk in file_chunks:
            metadata = chunk["metadata"]
            if metadata.get("level") == "code_chunk":
                function_name = metadata.get("function_name", "unknown")
                docstring = metadata.get("docstring")
                start_line = metadata.get("start_line")
                end_line = metadata.get("end_line")

                chunk_id = generate_chunk_id(rel_path, start_line, end_line)

                metadata["chunk_id"] = chunk_id

                if output_prefix:
                    extension = file_path.suffix if file_path.suffix else ".txt"
                    save_full_chunk(chunk_id, chunk["text"], output_prefix, extension)
                    metadata["full_code_path"] = f"full_chunks/{chunk_id}{extension}"

                metadata_text_parts = [f"Function: {function_name}"]
                metadata_text_parts.append(f"File: {rel_path}")
                metadata_text_parts.append(f"Lines: {start_line}-{end_line}")
                if docstring:
                    metadata_text_parts.append(f"Docstring: {docstring}")

                chunk["text"] = "\n".join(metadata_text_parts)
            elif metadata.get("level") == "document":
                # For document files like markdown, save the full content
                lines = code.split('\n')
                chunk_id = generate_chunk_id(rel_path, 1, len(lines))

                metadata["chunk_id"] = chunk_id

                if output_prefix:
                    extension = file_path.suffix if file_path.suffix else ".txt"
                    save_full_chunk(chunk_id, code, output_prefix, extension)  # Save full document content
                    metadata["full_code_path"] = f"full_chunks/{chunk_id}{extension}"

                # For documents, keep the full content as searchable text
                chunk["text"] = f"DOCUMENT: {rel_path}\n\n{code}"

        # File summary (skip tiny files)
        file_summary = None
        summary_chunk = None
        if len(code) > 100:
            summary = summarize_file(code, rel_path)
            file_summary = (rel_path, summary)
            summary_chunk = {
                "text": f"FILE: {rel_path}\n{summary}",
                "metadata": {
                    "file": rel_path,
                    "level": "file_summary",
                    "location": {"file": rel_path},
                },
            }

        return file_chunks, file_summary, summary_chunk, None

    except Exception:
        return [], None, None, f"Error processing {file_path}: {traceback.format_exc()}"