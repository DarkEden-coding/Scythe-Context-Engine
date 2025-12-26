"""

Query indexed repo for context with reranking + semantic cache.

Usage: python query_context.py "fix login rate limiting" --index repo_index

"""

import argparse
from rich.console import Console
from rich.markdown import Markdown

from query_context import query_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query context from indexed repo")

    parser.add_argument("query", help="Search query")

    parser.add_argument("--index", default="repo_index", help="Index prefix")

    parser.add_argument("--top-k", type=int, default=20, help="Initial retrieval count")

    parser.add_argument("--output-k", type=int, default=5, help="Final context chunks")

    parser.add_argument("--no-cache", action="store_true", help="Do not use cache")

    args = parser.parse_args()

    context = query_context(
        args.query, args.index, args.top_k, args.output_k, args.no_cache
    )

    console = Console()
    console.print("\n[bold blue]" + "=" * 80 + "[/bold blue]")
    console.print("[bold cyan]CONTEXT OUTPUT:[/bold cyan]")
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]\n")

    # Render the markdown content richly
    markdown = Markdown(context)
    console.print(markdown)
