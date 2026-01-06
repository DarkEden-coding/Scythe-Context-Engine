import hashlib
import sys
from pathlib import Path
import tiktoken

# Force UTF-8 encoding for stdout/stderr to handle Unicode characters
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Add project root to sys.path to allow imports from indexer and query_context
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp.server.fastmcp import FastMCP
from index_repo import index_repo
from query_context.query import query_context

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
    return ''.join(char for char in text if ord(char) < 128)


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
    Search the project for relevant code context.
    This tool will automatically index the project (incremental) before searching.

    Args:
        query_text: The search query or question about the codebase. Make it targeted and specific. ex: "Frontend user authentication" And target semantic matching, ie not "show me the code for user authentication" instead just be "user authentication frontend and backend"
        project_location: The absolute path to the project root directory on the local machine.
        token_limit: Maximum token count for the output (default 5000 tokens). Results exceeding this limit will be truncated.
    """
    try:
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
        try:
            index_repo(str(project_path), str(index_path), auto_confirm=True, quiet=True)
        except Exception as e:
            # Log indexing errors to stderr but continue to query if possible
            print(f"Indexing error (non-fatal): {e}", file=sys.stderr)

        # 3. Perform the query
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
        except UnicodeEncodeError as ue:
            # Handle encoding errors by returning stripped result or error message
            return f"Query completed but encountered encoding issues while processing: {_strip_non_ascii(str(ue))}"
        except Exception as ex:
            # Handle other exceptions
            return f"Query failed: {_strip_non_ascii(str(ex))}"

        # Strip non-ASCII characters from result before returning
        cleaned_result = _strip_non_ascii(result)

        # Apply token limit truncation
        truncated_result, was_truncated = _truncate_to_token_limit(cleaned_result, token_limit)

        if was_truncated:
            truncated_result += f"\n\n[Result truncated: output exceeded {token_limit} token limit]"
        return truncated_result
    except Exception as e:
        error_msg = _strip_non_ascii(str(e))
        return f"Error during query: {error_msg}"


if __name__ == "__main__":
    mcp.run()
