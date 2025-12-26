def summarize_folder(file_summaries: List[tuple]) -> str:
    """Aggregate file summaries into folder overview.

    Args:
        file_summaries: List of tuples containing (file_path, summary) pairs.

    Returns:
        One sentence description of the folder's purpose.
    """
    if not file_summaries:
        return "Empty folder"

    try:
        # Limit to first 8 files and format as list
        limited_summaries = file_summaries[:8]
        formatted_list = "\n".join(
            [f"- {Path(p).name}: {s}" for p, s in limited_summaries]
        )

        prompt = f"""Summarize this folder from file overviews (1 sentence):

{formatted_list}

Provide the folder purpose."""

        response_format = build_structured_output_format(
            FolderSummary.model_json_schema(), schema_name="folder_summary"
        )
        resp = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=SUMMARIZATION_MODEL,
            response_format=response_format,
            options={"temperature": 0.3},
        )

        message_content = extract_chat_content(resp)
        if message_content:
            try:
                folder_data = FolderSummary.model_validate_json(message_content)
                return folder_data.purpose
            except Exception:
                # If JSON parsing fails, use the raw response as summary
                # Remove common prefixes that might indicate non-JSON response
                cleaned_content = message_content.strip()
                if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
                    cleaned_content = cleaned_content[1:-1]
                return cleaned_content
        else:
            return "Multiple code files"

    except Exception:
        return "Multiple code files"