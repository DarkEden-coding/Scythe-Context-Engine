# Groq Batch API Usage Guide

## Overview

The Scythe Context Engine now supports using Groq's Batch API for cost-effective, asynchronous summarization during repository indexing. This feature can reduce summarization costs by up to 50% compared to real-time API calls, making it ideal for large repositories.

## When to Use Batch Mode

### ✅ Good Use Cases

- **Initial indexing of large repositories** (hundreds/thousands of files)
- **Cost-sensitive operations** where time is not critical
- **Overnight/background indexing jobs**
- **Bulk re-indexing** after configuration changes

### ❌ Not Recommended For

- **Small repositories** (< 50 files) - overhead not worth it
- **Incremental updates** - only a few files changed
- **Interactive/real-time queries** - batch takes hours to complete
- **Embeddings** - Groq Batch API doesn't support embeddings yet

## Configuration

### 1. Set Up Groq as Provider

Run the configuration wizard:

```bash
python config/create_config.py
```

Select `groq` as your provider and provide your Groq API key.

**Important:** Since Groq doesn't support embeddings yet, you'll need to configure a fallback provider (OpenRouter or Ollama) for embedding generation.

### 2. Configuration File Structure

Your `config/config.json` will look like:

```json
{
  "cache": { "ttl_seconds": 86400 },
  "provider": "groq",
  "groq": {
    "api_key": "gsk_...",
    "chat_model": "llama-3.3-70b-versatile",
    "batch_completion_window": "24h",
    "use_batch_for_indexing": true,
    "poll_interval_seconds": 30,
    "timeout_seconds": 60
  },
  "openrouter": {
    "api_key": "sk-or-...",
    "embedding_model": "openai/text-embedding-3-small"
  }
}
```

### 3. Hybrid Configuration (Recommended)

You can use OpenRouter/Ollama as your primary provider and Groq only for batch operations:

```json
{
  "provider": "openrouter",
  "batch_provider": "groq",
  "openrouter": {
    "api_key": "sk-or-...",
    "chat_model": "openai/gpt-oss-120b:exacto",
    "embedding_model": "openai/text-embedding-3-small"
  },
  "groq": {
    "api_key": "gsk_...",
    "chat_model": "llama-3.3-70b-versatile",
    "batch_completion_window": "24h",
    "use_batch_for_indexing": true
  }
}
```

This gives you:

- Fast real-time queries with OpenRouter
- Cost-effective batch indexing with Groq

## Usage

### Basic Batch Indexing

```bash
python index_repo.py /path/to/repo --output repo_index --batch
```

The `--batch` flag enables Groq Batch API usage for summarization.

### With Auto-Confirm and Quiet Mode

```bash
python index_repo.py /path/to/repo --output repo_index --batch --yes --quiet
```

This is ideal for automated scripts and cron jobs.

### Process Flow

When you run with `--batch`, here's what happens:

1. **Phase 1: Chunk Extraction** (local, fast)

   - Parses files and extracts code chunks
   - Generates chunk IDs and saves full code

2. **Phase 2: Batch Summarization** (remote, slow)

   - Creates JSONL file with all summarization requests
   - Uploads to Groq Batch API
   - Creates batch job
   - Polls every 30 seconds for completion
   - Downloads and processes results

3. **Phase 3: Embeddings** (uses fallback provider)
   - Generates embeddings for all chunks
   - Creates FAISS index

## Monitoring Batch Progress

During batch processing, you'll see real-time updates:

```
Phase 1: Extracting code chunks...
Extracting chunks: 100%|████████████| 450/450 [00:12<00:00]

Phase 2: Batch summarizing 423 files...
Uploading batch file...
Creating batch job (completion window: 24h)...
Batch job created: batch_01jh6xa7reempvjyh6n3yst2zw
Status: validating
Waiting for completion (this may take several minutes to hours)...
Status: in_progress | Progress: 123/423
Status: in_progress | Progress: 245/423
Status: in_progress | Progress: 367/423
Status: completed | Progress: 423/423
Batch completed! Processing results...
Cleaning up batch files...
Batch summarization complete: 420/423 successful
```

## Configuration Options

### Batch Completion Window

Controls how long Groq has to process your batch:

```json
"batch_completion_window": "24h"  // Options: "24h", "48h", "72h", up to "7d"
```

- Shorter windows may be processed faster
- Longer windows provide more flexibility for Groq's scheduler
- Default: `24h`

### Poll Interval

How often to check batch status:

```json
"poll_interval_seconds": 30  // Check every 30 seconds
```

- Lower values = more responsive feedback
- Higher values = fewer API calls
- Recommended: 30-60 seconds

### Use Batch for Indexing

Control whether batch mode is used by default:

```json
"use_batch_for_indexing": true
```

- If `false`, batch mode is disabled even with `--batch` flag
- Useful for temporarily disabling without changing command

## Error Handling

### Automatic Fallback

If batch processing fails, the system automatically falls back to real-time summarization:

```
Batch summarization failed: API timeout. Falling back to real-time...
Processing files with real-time summarization...
```

### Partial Success

If some batch requests fail, successful ones are used and failures get default summaries:

```
Batch summarization complete: 418/423 successful
Warning: 5 files failed to summarize
```

### File Cleanup

Batch files are automatically cleaned up after processing:

- Input JSONL file
- Output results file
- Error file (if any)

If cleanup fails, a warning is shown but indexing continues.

## Performance Comparison

### Small Repository (50 files)

| Mode      | Time   | Cost   |
| --------- | ------ | ------ |
| Real-time | 2 min  | $0.05  |
| Batch     | 15 min | $0.025 |

**Verdict:** Real-time is better (overhead not worth savings)

### Medium Repository (500 files)

| Mode      | Time   | Cost  |
| --------- | ------ | ----- |
| Real-time | 15 min | $0.50 |
| Batch     | 30 min | $0.25 |

**Verdict:** Batch worth it if cost matters more than time

### Large Repository (5,000 files)

| Mode      | Time      | Cost  |
| --------- | --------- | ----- |
| Real-time | 2.5 hours | $5.00 |
| Batch     | 3 hours   | $2.50 |

**Verdict:** Batch recommended (significant savings)

## Troubleshooting

### "Groq batch client not configured"

**Problem:** Groq API key not set in config

**Solution:** Run `python config/create_config.py` and configure Groq

### "Batch mode requested but Groq client not configured"

**Problem:** `use_batch_for_indexing` is `false` or API key missing

**Solution:** Check `config/config.json` and set:

```json
"groq": {
  "api_key": "gsk_...",
  "use_batch_for_indexing": true
}
```

### Batch Takes Too Long

**Problem:** Batch job stuck in "validating" or "in_progress" for hours

**Solution:**

1. Check Groq status page for outages
2. Try a longer completion window (e.g., "48h")
3. Cancel and restart: The system will auto-cleanup and retry

### Out of Memory During Embedding

**Problem:** Too many chunks to embed at once

**Solution:** This is unrelated to batch mode. Reduce batch size in `indexer/embedder.py`:

```python
batch_size = 16  # Instead of 32
```

## Best Practices

### 1. Hybrid Workflow

Use the best tool for each job:

```bash
# Initial index with batch (cost-effective)
python index_repo.py ~/large-project --output index --batch --yes

# Incremental updates without batch (fast)
python index_repo.py ~/large-project --output index --yes

# Queries always use real-time provider
python query_context.py index "authentication flow"
```

### 2. Scheduled Batch Jobs

Set up a cron job for overnight indexing:

```cron
# Daily at 2 AM - re-index with batch mode
0 2 * * * cd /path/to/scythe && python index_repo.py ~/projects --batch --yes --quiet
```

### 3. Monitor Costs

Enable Groq API usage tracking to monitor costs:

- Check your Groq dashboard regularly
- Set up billing alerts
- Compare batch vs. real-time costs

### 4. Test First

Before indexing a massive repo:

```bash
# Test on a subset
python index_repo.py ~/large-project/src/core --output test_index --batch
```

## API Limits

### Groq Batch API Limits

- **File Size:** Max 100 MB per JSONL file
- **Requests per Batch:** No documented limit, but practical limit ~10,000
- **Concurrent Batches:** Depends on your Groq tier
- **Completion Window:** 24h to 7d

### Rate Limiting

Batch API is rate-limited at the batch level, not per-request. This means:

- You won't hit per-request rate limits
- But you may have a limit on active batches
- Check your Groq account tier for specifics

## Advanced: Custom Batch Callbacks

You can provide custom progress callbacks:

```python
from groq_batch_client import GroqBatchClient, BatchJob

def my_callback(job: BatchJob):
    print(f"Custom tracking: {job.id} - {job.status}")
    # Log to file, update database, send notification, etc.

# Use in your code
from config.config import get_groq_batch_client
from indexer.summarizer import batch_summarize_files

client = get_groq_batch_client()
results = batch_summarize_files(
    file_data,
    batch_client=client,
    callback=my_callback
)
```

## See Also

- [Implementation Plan](GROQ_BATCH_IMPLEMENTATION_PLAN.md) - Technical architecture details
- [Groq Batch API Docs](https://console.groq.com/docs/batch) - Official Groq documentation
- [Configuration Guide](../config/create_config.py) - Interactive setup wizard
