import hashlib
import sys
from pathlib import Path

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


@mcp.tool()
def query(query_text: str, project_location: str) -> str:
    """
    Search the project for relevant code context.
    This tool will automatically index the project (incremental) before searching.

    Args:
        query_text: The search query or question about the codebase. Make it targeted and specific. ex: "Frontend user authentication" And target semantic matching, ie not "show me the code for user authentication" instead just be "user authentication frontend and backend"
        project_location: The absolute path to the project root directory on the local machine.
    """
    try:
        # 1. Determine index path (store in context engine's indexes folder)
        project_path = Path(project_location).absolute()
        project_id = get_project_identifier(str(project_path))
        context_engine_path = Path(__file__).parent.parent.absolute()
        index_path = context_engine_path / "indexes" / project_id

        # Ensure the index directory exists
        index_path.mkdir(parents=True, exist_ok=True)

        # 2. Run incremental indexing
        # index_repo(repo_path, output_prefix, auto_confirm=True)
        # Note: index_repo prints to stdout, we might want to capture it or just let it flow
        print(f"Indexing {project_path} into {index_path}...")
        index_repo(str(project_path), str(index_path), auto_confirm=True, quiet=True)

        # 3. Perform the query
        # query_context(query, index_prefix, top_k=20, output_k=5, no_cache=False)
        print(f"Querying: {query_text}")
        result = query_context(
            query=query_text,
            index_prefix=str(index_path),
            top_k=20,
            output_k=10,
            no_cache=False,
        )

        return result
    except Exception as e:
        return f"Error during query: {str(e)}"


if __name__ == "__main__":
    mcp.run()
