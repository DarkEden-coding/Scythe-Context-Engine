def summarize_file(code: str, file_path: str) -> str:
    """Generate file summary via LLM.

    Args:
        code: The source code content as a string.
        file_path: Relative path to the file being summarized.

    Returns:
        One to two sentence summary of the file's purpose and key components.
    """
    try:
        prompt = f"""Summarize this {Path(file_path).suffix} file in 1-2 sentences.

Focus on: purpose, key functions/classes, dependencies.



{code[:3000]}



Summary:"""

        response_format = build_structured_output_format(
            FileSummary.model_json_schema(), schema_name="file_summary"
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
                summary_data = FileSummary.model_validate_json(message_content)
                return summary_data.summary
            except Exception:
                # If JSON parsing fails, use the raw response as summary
                # Remove common prefixes that might indicate non-JSON response
                cleaned_content = message_content.strip()
                if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
                    cleaned_content = cleaned_content[1:-1]
                return cleaned_content
        else:
            return f"File: {Path(file_path).name} (summary failed: empty response)"

    except Exception:
        return (
            f"File: {Path(file_path).name} (summary failed: {traceback.format_exc()})"
        )