"""

Index a code repository: AST chunks + file/folder summaries + embeddings.

Usage: python index_repo.py /path/to/repo --output repo_index

"""

import argparse
from collections import defaultdict

from line_profiler import profile

from indexer.file_processor import (
    collect_files_to_process,
    generate_folder_summaries,
    process_files,
)
from indexer.embedder import create_faiss_index, save_index


@profile
def index_repo(repo_path: str, output_prefix: str):
    """Main indexing pipeline for creating searchable repository index.

    Args:
        repo_path: Path to the repository to index.
        output_prefix: Directory prefix where index files will be saved.
    """
    print(f"📁 Indexing {repo_path}...")

    # 1. Collect files to process
    files_to_process = collect_files_to_process(repo_path)
    print(f"📄 Found {len(files_to_process)} code files")

    # 1.5. Show folder overview and get confirmation
    folder_counts = defaultdict(int)
    for file_path in files_to_process:
        rel_path = file_path.relative_to(repo_path)
        folder = str(rel_path.parent)
        if folder == ".":
            folder = "(root)"
        folder_counts[folder] += 1

    print("\n📂 Repository Overview:")
    print("=" * 50)
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder}: {count} code files")
    print("=" * 50)

    confirm = input("\nProceed with indexing? (y/N): ").lower().strip()
    if confirm not in ["y", "yes"]:
        print("Indexing cancelled.")
        return

    # 2. Process files to extract chunks and summaries
    chunks, file_summaries = process_files(files_to_process, repo_path, output_prefix)

    # 3. Generate folder summaries
    print("📂 Generating folder summaries...")
    chunks = generate_folder_summaries(file_summaries, chunks)

    print(f"✅ Extracted {len(chunks)} total chunks")

    # 4. Create FAISS index
    print("🔢 Embedding chunks...")
    index, embedding_dim = create_faiss_index(chunks)

    # 5. Save index
    print("💾 Saving index...")
    save_index(index, chunks, repo_path, output_prefix, embedding_dim)

    print(f"✅ Index saved to {output_prefix}/ ({len(chunks)} chunks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index code repository")

    parser.add_argument("repo_path", help="Path to repository")

    parser.add_argument("--output", default="repo_index", help="Output prefix")

    args = parser.parse_args()

    index_repo(args.repo_path, args.output)
