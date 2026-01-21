import hashlib
import sys
import time
import uuid
from pathlib import Path
import tiktoken

# Force UTF-8 encoding for stdout/stderr to handle Unicode characters
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Add project root to sys.path to allow imports from indexer and query_context
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp.server.fastmcp import FastMCP
from index_repo import index_repo
from query_context.query import query_context
from utils.logger import (
    init_logging_system,
    set_query_context,
    create_query_logger,
    log_event,
)

# Initialize logging system before FastMCP
init_logging_system()

# Initialize FastMCP server
mcp = FastMCP("Scythe Context Engine")


def get_project_identifier(project_path: str) -> str:
    """Generate a unique identifier for a project based on its absolute path.

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        A unique identifier string for the project.
    """
    # Use SHA256 hash of the absolute path to create a unique identifier
    return hashlib.sha256(project_path.encode()).hexdigest()[:16]


def _strip_non_ascii(text: str) -> str:
    """Remove all non-ASCII characters from a string.

    Args:
        text: The string to clean.

    Returns:
        String with only ASCII characters preserved.
    """
    if not isinstance(text, str):
        return str(text)
    return "".join(char for char in text if ord(char) < 128)


def _truncate_to_token_limit(text: str, token_limit: int) -> tuple:
    """Truncate text to a maximum token count using tiktoken.

    Args:
        text: The text to truncate.
        token_limit: Maximum number of tokens to keep.

    Returns:
        A tuple of (truncated_text, was_truncated)
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= token_limit:
        return text, False

    truncated_tokens = tokens[:token_limit]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text, True


@mcp.tool()
def query(query_text: str, project_location: str, token_limit: int = 15000) -> str:
    """
    Search the project codebase using a semantic RAG engine. For optimal retrieval, follow these guidelines:

    *   **Prioritize Semantic Intent**: Use descriptive, conceptual phrases that define the mechanism, data flow, or architectural bridge you are investigating.
    *   **Avoid Keyword Stuffing**: Do not provide a list of disconnected terms (e.g., `auth login user token`). Instead, describe the interaction (e.g., `process for validating JWT tokens during user login`).
    *   **Utilize Multi-Querying**: For complex tasks spanning multiple architectural layers or distinct logic paths, perform multiple targeted queries rather than one broad one.
        *   *Example*: Run one query for `frontend state management of operation configs` and a separate query for `backend persistence and validation logic`.
    *   **Focus on Relationships**: Describe how components interact.
        *   **Good**: `Mapping and synchronization of operation IDs between frontend state and backend persistence`.
        *   **Bad**: `operation IDs identification frontend backend saving loading`.
    *   **Provide Technical Context**: This system aggregates and rescores results across a large token window (15k); detailed descriptions of specific logic or interactions will produce higher-quality context than generic terms.

    Args:
        query_text: The search query or question about the codebase.
        project_location: The absolute path to the project root directory on the local machine.
        token_limit: Maximum token count for the output (default 5000 tokens). Results exceeding this limit will be truncated.
    """
    # Generate unique query_id and initialize query context
    query_id = f"{uuid.uuid4().hex[:16]}"
    set_query_context(query_id, query_text=query_text, project_location=project_location)

    # Create query-specific log file
    create_query_logger(query_id)

    query_start_time = time.time()

    try:
        # Log query entry
        log_event(
            event="query_start",
            level="INFO",
            phase="server",
            component="mcp_server",
            message="Query started",
            data={
                "query_preview": query_text[:100],
                "project_location": project_location,
                "token_limit": token_limit,
            },
        )

        # Strip non-ASCII characters from inputs
        query_text = _strip_non_ascii(query_text)
        project_location = _strip_non_ascii(project_location)

        # 1. Determine index path (store in context engine's indexes folder)
        project_path = Path(project_location).absolute()
        project_id = get_project_identifier(str(project_path))
        context_engine_path = Path(__file__).parent.parent.absolute()
        index_path = context_engine_path / "indexes" / project_id

        # Ensure the index directory exists
        index_path.mkdir(parents=True, exist_ok=True)

        # 2. Run incremental indexing
        indexing_start_time = time.time()
        log_event(
            event="indexing_phase_start",
            level="INFO",
            phase="indexing",
            component="server",
            message="Starting indexing phase",
        )

        try:
            index_repo(
                str(project_path), str(index_path), auto_confirm=True, quiet=True, for_mcp_query=True
            )
            indexing_duration_ms = (time.time() - indexing_start_time) * 1000
            log_event(
                event="indexing_phase_complete",
                level="INFO",
                phase="indexing",
                component="server",
                message="Indexing phase completed successfully",
                duration_ms=indexing_duration_ms,
            )
        except Exception as e:
            # Log indexing errors to stderr but continue to query if possible
            indexing_duration_ms = (time.time() - indexing_start_time) * 1000
            log_event(
                event="indexing_phase_error",
                level="WARNING",
                phase="indexing",
                component="server",
                message=f"Indexing error (non-fatal): {str(e)}",
                duration_ms=indexing_duration_ms,
                error=e,
            )
            print(f"Indexing error (non-fatal): {e}", file=sys.stderr)

        # 3. Perform the query
        query_phase_start_time = time.time()
        log_event(
            event="query_phase_start",
            level="INFO",
            phase="query",
            component="server",
            message="Starting query phase",
        )

        try:
            result = query_context(
                query=query_text,
                index_prefix=str(index_path),
                top_k=20,
                output_k=10,
                no_cache=False,
                token_limit=token_limit,
                quiet=True,
            )
            query_phase_duration_ms = (time.time() - query_phase_start_time) * 1000
            log_event(
                event="query_phase_complete",
                level="INFO",
                phase="query",
                component="server",
                message="Query phase completed successfully",
                duration_ms=query_phase_duration_ms,
            )
        except UnicodeEncodeError as ue:
            # Handle encoding errors by returning stripped result or error message
            query_phase_duration_ms = (time.time() - query_phase_start_time) * 1000
            log_event(
                event="query_phase_error",
                level="ERROR",
                phase="query",
                component="server",
                message=f"Encoding error during query: {str(ue)}",
                duration_ms=query_phase_duration_ms,
                error=ue,
            )
            return f"Query completed but encountered encoding issues while processing: {_strip_non_ascii(str(ue))}"
        except Exception as ex:
            # Handle other exceptions
            query_phase_duration_ms = (time.time() - query_phase_start_time) * 1000
            log_event(
                event="query_phase_error",
                level="ERROR",
                phase="query",
                component="server",
                message=f"Error during query: {str(ex)}",
                duration_ms=query_phase_duration_ms,
                error=ex,
            )
            return f"Query failed: {_strip_non_ascii(str(ex))}"

        # Strip non-ASCII characters from result before returning
        cleaned_result = _strip_non_ascii(result)

        # Apply token limit truncation
        truncated_result, was_truncated = _truncate_to_token_limit(
            cleaned_result, token_limit
        )

        total_duration_ms = (time.time() - query_start_time) * 1000
        log_event(
            event="query_complete",
            level="INFO",
            phase="server",
            component="mcp_server",
            message="Query completed successfully",
            data={
                "result_length": len(cleaned_result),
                "was_truncated": was_truncated,
                "token_limit": token_limit,
            },
            duration_ms=total_duration_ms,
        )

        if was_truncated:
            truncated_result += (
                f"\n\n[Result truncated: output exceeded {token_limit} token limit]"
            )
        return truncated_result
    except Exception as e:
        total_duration_ms = (time.time() - query_start_time) * 1000
        error_msg = _strip_non_ascii(str(e))
        log_event(
            event="query_error",
            level="ERROR",
            phase="server",
            component="mcp_server",
            message=f"Fatal error during query: {error_msg}",
            duration_ms=total_duration_ms,
            error=e,
        )
        return f"Error during query: {error_msg}"


if __name__ == "__main__":
    mcp.run()
