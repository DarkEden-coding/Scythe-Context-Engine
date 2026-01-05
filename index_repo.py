"""

Index a code repository: AST chunks + file/folder summaries + embeddings.

Usage: python index_repo.py /path/to/repo --output repo_index

"""

import argparse
import json
import pickle
from pathlib import Path

from line_profiler import profile

from indexer.file_processor import (
    collect_files_to_process,
    generate_folder_summaries,
    hash_file,
    process_files,
)
from indexer.embedder import create_faiss_index, save_index


@profile
def index_repo(
    repo_path: str, output_prefix: str, auto_confirm: bool = False, quiet: bool = False
):
    """Main indexing pipeline for creating searchable repository index.

    Args:
        repo_path: Path to the repository to index.
        output_prefix: Directory prefix where index files will be saved.
        auto_confirm: If True, bypass the confirmation prompt.
        quiet: If True, suppress progress bars and reduce output verbosity.
    """

    print(f"Indexing {repo_path}...")
    repo_path_obj = Path(repo_path)
    output_path = Path(output_prefix)

    # 1. Load existing state if available
    old_chunks = []
    old_hashes = {}
    if (output_path / "chunks.pkl").exists() and (output_path / "meta.json").exists():
        if not quiet:
            print("Loading existing index for incremental update...")
        try:
            with open(output_path / "chunks.pkl", "rb") as f:
                old_chunks = pickle.load(f)
            with open(output_path / "meta.json", "r") as f:
                meta = json.load(f)
                old_hashes = meta.get("file_hashes", {})
        except Exception as e:
            if not quiet:
                print(
                    f"Could not load existing index: {e}. Performing full re-index."
                )
            old_chunks = []
            old_hashes = {}

    # 2. Collect files and identify changes
    all_files = collect_files_to_process(repo_path)
    current_hashes = {str(f.relative_to(repo_path)): hash_file(f) for f in all_files}

    added_files = []
    modified_files = []
    unchanged_files = []
    deleted_files = [f for f in old_hashes if f not in current_hashes]

    for rel_path, current_hash in current_hashes.items():
        if rel_path not in old_hashes:
            added_files.append(repo_path_obj / rel_path)
        elif old_hashes[rel_path] != current_hash:
            modified_files.append(repo_path_obj / rel_path)
        else:
            unchanged_files.append(repo_path_obj / rel_path)

    files_to_process = added_files + modified_files

    # 3. Show overview and get confirmation
    print("\nRepository Overview:")
    print("=" * 50)
    if not old_hashes:
        print(f"  Total files to index: {len(all_files)}")
    else:
        print(f"  Unchanged: {len(unchanged_files)}")
        print(f"  Added:     {len(added_files)}")
        print(f"  Modified:  {len(modified_files)}")
        print(f"  Deleted:   {len(deleted_files)}")
    print("=" * 50)

    if not files_to_process and not deleted_files:
        print("Index is already up to date. No changes detected.")
        return

    if not auto_confirm:
        confirm = input("\nProceed with indexing? (y/N): ").lower().strip()
        if confirm not in ["y", "yes"]:
            print("Indexing cancelled.")
            return
    else:
        if not quiet:
            print("\nAuto-confirming indexing...")

    # 4. Filter old chunks (keep only those from unchanged files)
    # We remove both code chunks and file summaries for modified/deleted files
    changed_rel_paths = set(
        [str(f.relative_to(repo_path)) for f in modified_files] + deleted_files
    )

    # Keep chunks that are NOT from changed files AND are not folder summaries
    # (folder summaries are regenerated)
    kept_chunks = [
        c
        for c in old_chunks
        if c.get("metadata", {}).get("file") not in changed_rel_paths
        and c.get("metadata", {}).get("level") != "folder_summary"
    ]

    # 5. Process new/modified files
    new_chunks = []
    file_summaries = {}
    if files_to_process:
        if quiet:
            print("Indexing...")
        else:
            print(f"Processing {len(files_to_process)} changed/new files...")
        new_chunks, file_summaries = process_files(
            files_to_process, repo_path, output_prefix, quiet=quiet
        )

    # Merge file summaries: old ones for unchanged files + new ones
    all_file_summaries = {}
    # Extract file summaries from kept_chunks
    for c in kept_chunks:
        if c.get("metadata", {}).get("level") == "file_summary":
            rel_path = c["metadata"]["file"]
            # Extract the summary text from the chunk text "FILE: rel_path\nsummary"
            summary = c["text"].split("\n", 1)[1] if "\n" in c["text"] else ""
            all_file_summaries[rel_path] = summary

    all_file_summaries.update(file_summaries)

    # Combine chunks
    combined_chunks = kept_chunks + new_chunks

    # 6. Generate folder summaries (always regenerate to stay accurate)
    if quiet:
        print("Summarizing...")
    else:
        print("Generating folder summaries...")
    final_chunks = generate_folder_summaries(
        all_file_summaries, combined_chunks, quiet=quiet
    )

    if not quiet:
        print(f"Total chunks: {len(final_chunks)} ({len(new_chunks)} new/updated)")

    # 7. Create FAISS index
    if quiet:
        print("Embedding...")
    else:
        print("Embedding chunks...")
    index, embedding_dim, indexed_chunks = create_faiss_index(final_chunks, quiet=quiet)

    # 8. Save index and update metadata with hashes
    if quiet:
        print("Saving...")
    else:
        print("Saving index...")
    save_index(index, indexed_chunks, repo_path, output_prefix, embedding_dim)

    # Update meta.json with file hashes
    with open(output_path / "meta.json", "r") as f:
        meta = json.load(f)

    meta["file_hashes"] = current_hashes
    with open(output_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if quiet:
        print("Indexed.")
    else:
        print(f"Index updated in {output_prefix}/ ({len(indexed_chunks)} chunks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index code repository")

    parser.add_argument("repo_path", help="Path to repository")

    parser.add_argument("--output", default="repo_index", help="Output prefix")

    parser.add_argument(
        "--yes", "-y", action="store_true", help="Auto-confirm indexing"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode: reduce output verbosity"
    )

    args = parser.parse_args()

    index_repo(args.repo_path, args.output, auto_confirm=args.yes, quiet=args.quiet)
