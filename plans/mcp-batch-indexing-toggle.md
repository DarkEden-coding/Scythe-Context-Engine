# Plan: MCP-Specific Batch Indexing Toggle

## Overview

Add a new configuration setting to control Groq Batch API usage specifically for incremental indexing triggered by MCP queries. This setting will default to **off** to avoid the long wait times (20+ minutes) associated with batch processing during interactive query sessions.

## Current Architecture

### Configuration Flow

```
config.json → config/config.py → USE_BATCH_FOR_INDEXING (global constant)
                                           ↓
                                   file_processor.py
                                           ↓
                                   process_files() checks USE_BATCH_FOR_INDEXING
```

### Indexing Call Paths

1. **CLI**: `python index_repo.py <path>` → `index_repo()` → `process_files()`
2. **MCP**: `mcp_server/server.py:query()` → `index_repo()` → `process_files()`

Both paths currently use the same `USE_BATCH_FOR_INDEXING` setting.

## Proposed Design

### New Configuration Setting

```json
{
  "use_batch_for_indexing": false, // Existing: General batch setting
  "use_batch_for_mcp_incremental_indexing": false // NEW: MCP-specific setting
}
```

### Configuration Constants (config/config.py)

```python
# Existing
USE_BATCH_FOR_INDEXING: bool = _config.get("use_batch_for_indexing", False)

# New
USE_BATCH_FOR_MCP_INCREMENTAL_INDEXING: bool = _config.get(
    "use_batch_for_mcp_incremental_indexing", False
)
```

### Decision Logic

The new logic will use **different batch settings** based on the indexing trigger source:

```
┌─────────────────────────────────────────────────────────────┐
│                    Batch Mode Decision Logic                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  index_repo(for_mcp_query=False)  ← CLI indexing           │
│       │                                                      │
│       └─→ process_files(use_batch=USE_BATCH_FOR_INDEXING)   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  index_repo(for_mcp_query=True)   ← MCP query indexing     │
│       │                                                      │
│       └─→ process_files(use_batch=USE_BATCH_FOR_MCP_       │
│                                   INCREMENTAL_INDEXING)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Separate Settings**: MCP queries get their own batch setting, independent of CLI indexing
2. **Default to OFF**: `use_batch_for_mcp_incremental_indexing` defaults to `false` for responsive queries
3. **Explicit Parameter**: `index_repo()` accepts `for_mcp_query` boolean to indicate trigger source
4. **Backward Compatible**: Existing `USE_BATCH_FOR_INDEXING` remains unchanged for CLI usage

## Implementation Steps

### 1. Update config/create_config.py

**Add to DEFAULTS dictionary:**

```python
DEFAULTS = {
    "use_batch_for_indexing": False,
    "use_batch_for_mcp_incremental_indexing": False,  # NEW
    # ... rest of defaults
}
```

**Add to wizard prompt (after Groq batch indexing question):**

```python
# After asking about general batch indexing (line 212-216)
if provider == "groq" or config.get("batch_provider") == "groq":
    print("\nMCP Query Incremental Indexing:")
    print("MCP queries can trigger incremental indexing before each query.")
    print("⚠️  WARNING: Batch indexing takes 20+ minutes per update")
    print("   (vs ~1-2 minutes for standard API calls)")
    print("   This can significantly delay query responses.")
    config["use_batch_for_mcp_incremental_indexing"] = ask(
        "Enable batch API for MCP-triggered incremental indexing?",
        DEFAULTS["use_batch_for_mcp_incremental_indexing"],
        skip,
    )
```

### 2. Update config/config.py

**Add new constant after line 56:**

```python
# Batch Indexing Setting (top-level config)
USE_BATCH_FOR_INDEXING: bool = _config.get("use_batch_for_indexing", False)

# MCP-Specific Batch Indexing Setting (for incremental indexing triggered by MCP queries)
USE_BATCH_FOR_MCP_INCREMENTAL_INDEXING: bool = _config.get(
    "use_batch_for_mcp_incremental_indexing", False
)
```

### 3. Update index_repo.py

**Modify function signature (line 26-31):**

```python
def index_repo(
    repo_path: str,
    output_prefix: str,
    auto_confirm: bool = False,
    quiet: bool = False,
    for_mcp_query: bool = False,  # NEW parameter
):
```

**Update docstring:**

```python
Args:
    repo_path: Path to the repository to index.
    output_prefix: Directory prefix where index files will be saved.
    auto_confirm: If True, bypass the confirmation prompt.
    quiet: If True, suppress progress bars and reduce output verbosity.
    for_mcp_query: If True, indicates indexing is triggered by MCP query.
                   Uses MCP-specific batch setting instead of general setting.
```

**Pass `for_mcp_query` to `process_files()` (line 131-133):**

```python
new_chunks, file_summaries = process_files(
    files_to_process, repo_path, output_prefix, quiet=quiet, for_mcp_query=for_mcp_query
)
```

### 4. Update indexer/file_processor.py

**Import new constant (line 19-25):**

```python
from config.config import (
    IGNORED_DIRS,
    IGNORED_FILES,
    SUPPORTED_LANGS,
    USE_BATCH_FOR_INDEXING,
    USE_BATCH_FOR_MCP_INCREMENTAL_INDEXING,  # NEW
    get_groq_batch_client,
)
```

**Modify `process_files()` function signature:**

```python
def process_files(
    files_to_process: List[Path],
    repo_path: Path,
    output_prefix: str,
    quiet: bool = False,
    for_mcp_query: bool = False,  # NEW parameter
):
```

**Update batch mode detection logic (line 263-265):**

```python
# Check if batch mode is configured BEFORE processing
batch_client = get_groq_batch_client()
use_batch = (
    (USE_BATCH_FOR_MCP_INCREMENTAL_INDEXING if for_mcp_query else USE_BATCH_FOR_INDEXING)
    and batch_client is not None
)
```

**Pass `for_mcp_query` to `process_single_file()` calls (line 299):**

```python
futures = [
    executor.submit(process_single_file, file_path, repo_path, output_prefix, use_batch, for_mcp_query)
    for file_path in files_to_process
]
```

**Update `process_single_file()` function signature:**

```python
def process_single_file(
    file_path: Path,
    repo_path: Path,
    output_prefix: str,
    use_batch: bool,
    for_mcp_query: bool = False,  # NEW parameter
):
```

### 5. Update mcp_server/server.py

**Pass `for_mcp_query=True` when calling `index_repo()` (line 110-112):**

```python
index_repo(
    str(project_path), str(index_path), auto_confirm=True, quiet=True, for_mcp_query=True
)
```

### 6. Update Documentation

**Add to docs/GROQ_BATCH_USAGE.md:**

````markdown
## MCP Query Incremental Indexing

The MCP query tool automatically runs incremental indexing before each query to ensure
fresh results. You can control whether this uses the Groq Batch API:

```json
{
  "use_batch_for_mcp_incremental_indexing": false
}
```
````

- **false (default)**: Uses real-time API for MCP incremental indexing

  - Pros: Fast query responses (~1-2 minutes)
  - Cons: Higher cost for incremental updates

- **true**: Uses Batch API for MCP incremental indexing
  - Pros: Lower cost (~50% savings)
  - Cons: Long wait times (20+ minutes) for first query after changes

**Recommendation**: Keep this setting `false` for interactive use cases where
query responsiveness is important. Enable it only for cost-sensitive scenarios
where you can tolerate longer initial query times.

```

## Testing Checklist

- [ ] Run `python config/create_config.py` and verify new setting appears in wizard
- [ ] Verify config.json contains new setting after wizard completion
- [ ] Test CLI indexing: `python index_repo.py <path>` uses `USE_BATCH_FOR_INDEXING`
- [ ] Test MCP query: batch mode uses `USE_BATCH_FOR_MCP_INCREMENTAL_INDEXING`
- [ ] Verify backward compatibility with existing config.json files
- [ ] Test with `use_batch_for_mcp_incremental_indexing = true` (batch mode for MCP)
- [ ] Test with `use_batch_for_mcp_incremental_indexing = false` (real-time for MCP)

## Migration Notes

Existing `config.json` files will work without modification:
- Missing `use_batch_for_mcp_incremental_indexing` defaults to `false`
- Existing `use_batch_for_indexing` behavior unchanged for CLI indexing

To enable MCP batch indexing, users can manually add the setting or re-run
`python config/create_config.py`.
```
