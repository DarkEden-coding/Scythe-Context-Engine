from mcp.server.fastmcp import FastMCP
from index_repo import index_repo
from query_context import query_context
import os
import sys
import contextlib

# Initialize FastMCP server
mcp = FastMCP("Scythe Context Engine")


@mcp.tool()
def query(query: str, project_location: str) -> str:
    """
    Query the codebase for context. This tool will:
    1. Incrementally index the project at the given location.
    2. Search the index for the most relevant context.
    3. Return the refined context.

    Args:
        query: The search query or question about the codebase.
        project_location: The absolute path to the project directory to index and query.
    """
    repo_path = os.path.abspath(project_location)
    # Use a subdirectory in the project for the index to avoid cluttering the root
    index_path = os.path.join(repo_path, ".scythe_index")

    # We redirect stdout to stderr because MCP uses stdout for its protocol
    with contextlib.redirect_stdout(sys.stderr):
        try:
            # 1. Incremental Indexing
            print(f"Indexing repository at {repo_path}...")
            index_repo(repo_path, index_path, auto_confirm=True, quiet=True)

            # 2. Query Context
            print(f"Querying context for: {query}")
            context = query_context(
                query=query,
                index_prefix=index_path,
                top_k=30,
                output_k=10,
                no_cache=False,
            )
            return context
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
