#!/usr/bin/env python3
import json
from pathlib import Path

DEFAULTS: dict = {
    "use_batch_for_indexing": False,
    "use_batch_for_mcp_incremental_indexing": False,
    "indexing": {
        "supported_languages": {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".md": "markdown",
        },
        "ignore_patterns": [
            ".git",
            "node_modules",
            "__pycache__",
            "**/.venv/**",
            "**/venv/**",
            "**/build/**",
            "**/dist/**",
            "*.pyc",
            "*.lock",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "bun.lockb",
            "Gemfile.lock",
            "Cargo.lock",
            "static",
            ".cudavenv",
        ],
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "chat_model": "openai/gpt-oss-120b:exacto",
        "embedding_model": "openai/text-embedding-3-small",
        "timeout_seconds": 15,
        "chat_provider_whitelist": ["groq"],
        "embedding_provider_whitelist": ["openai"],
    },
    "ollama": {
        "summarization_model": "gemma3:1b",
        "embedding_model": "qwen3-embedding:0.6b",
    },
    "groq": {
        "chat_model": "openai/gpt-oss-120b",
        "batch_completion_window": "24h",
        "poll_interval_seconds": 30,
        "timeout_seconds": 60,
    },
}


def ask(prompt, default, skip_interactive=False):
    """Returns default if skip_interactive is True, else prompts user."""
    if skip_interactive:
        return default

    if isinstance(default, bool):
        choice = input(f"{prompt} ({'Y/n' if default else 'y/N'}): ").strip().lower()
        return default if not choice else choice in ["y", "yes"]

    res = input(f"{prompt} [{default}]: ").strip()
    return res if res else default


def get_list(prompt, defaults, skip_interactive):
    """Get a list of items from user or use defaults."""
    if skip_interactive or not ask(f"Customize {prompt}?", False):
        return defaults
    print(f"Enter items for {prompt} (one per line, empty to finish):")
    items = []
    while True:
        val = input("> ").strip()
        if not val:
            break
        items.append(val)
    return items


def get_openrouter_config(api_key, skip_interactive, use_defaults=True):
    """Create complete OpenRouter configuration with all required fields.

    Args:
        api_key: The OpenRouter API key
        skip_interactive: If True, use all defaults without prompting
        use_defaults: If True, use default values for all fields

    Returns:
        Complete OpenRouter configuration dictionary
    """
    p_def: dict = DEFAULTS["openrouter"]
    config = {
        "api_key": api_key,
        "api_base": p_def["api_base"],
        "timeout_seconds": p_def["timeout_seconds"],
    }

    if use_defaults or skip_interactive:
        config["chat_model"] = p_def["chat_model"]
        config["embedding_model"] = p_def["embedding_model"]
        config["chat_provider_whitelist"] = p_def["chat_provider_whitelist"]
        config["embedding_provider_whitelist"] = p_def["embedding_provider_whitelist"]
    else:
        config["chat_model"] = ask("Chat Model", p_def["chat_model"], skip_interactive)
        config["embedding_model"] = ask(
            "Embedding Model", p_def["embedding_model"], skip_interactive
        )
        config["chat_provider_whitelist"] = (
            p_def["chat_provider_whitelist"] if ask("Use chat whitelist?", True) else []
        )
        config["embedding_provider_whitelist"] = (
            p_def["embedding_provider_whitelist"]
            if ask("Use embedding whitelist?", True)
            else []
        )

    return config


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     Scythe Context Engine - Configuration Wizard         ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    print("This wizard will help you configure your LLM provider settings.\n")

    # Step 1: Choose primary provider
    print("Step 1: Choose your primary LLM provider")
    print("-" * 50)
    print("  • openrouter - Access to multiple models via OpenRouter")
    print("  • ollama     - Local models via Ollama")
    print("  • groq       - Fast inference via Groq (requires embedding fallback)")
    print()

    provider: str = ""
    while provider not in ["openrouter", "ollama", "groq"]:
        provider = (
            input("Select provider [openrouter]: ").strip().lower() or "openrouter"
        )

    skip: bool = ask("\nUse default settings for everything else?", True)

    config: dict = {
        "cache": {"ttl_seconds": 86400},
        "provider": provider,
        "use_batch_for_indexing": False,  # Will be set later based on Groq config
        "indexing": {
            "supported_languages": DEFAULTS["indexing"]["supported_languages"],
            "ignore_patterns": get_list(
                "ignore patterns (supports wildcards: *.lock, **/build/**, etc.)",
                DEFAULTS["indexing"]["ignore_patterns"],
                skip,
            ),
        },
    }

    # Step 2: Configure the chosen provider
    print(f"\nStep 2: Configure {provider.upper()}")
    print("-" * 50)

    if provider == "openrouter":
        key: str = ""
        while not key:
            key = input("Enter your OpenRouter API key: ").strip()
        config["openrouter"]: dict = get_openrouter_config(key, skip, use_defaults=skip)
        print("✓ OpenRouter configured")
    elif provider == "ollama":
        print("Make sure Ollama is running locally before proceeding.\n")
        p_def: dict = DEFAULTS["ollama"]
        config["ollama"]: dict = {
            "summarization_model": ask(
                "Summarization Model", p_def["summarization_model"], skip
            ),
            "embedding_model": ask("Embedding Model", p_def["embedding_model"], skip),
        }
        print("✓ Ollama configured")
    elif provider == "groq":
        key: str = ""
        while not key:
            key = input("Enter your Groq API key: ").strip()

        g_def: dict = DEFAULTS["groq"]
        config["groq"]: dict = {
            "api_key": key,
            "chat_model": ask("Chat Model", g_def["chat_model"], skip),
            "batch_completion_window": ask(
                "Batch Completion Window (24h/48h/72h/7d)",
                g_def["batch_completion_window"],
                skip,
            ),
            "poll_interval_seconds": int(
                ask("Poll Interval (seconds)", g_def["poll_interval_seconds"], skip)
            ),
            "timeout_seconds": int(
                ask("Timeout (seconds)", g_def["timeout_seconds"], skip)
            ),
        }
        print("✓ Groq configured")

        # Ask about batch indexing for Groq
        print("\nBatch Indexing Configuration:")
        print("Groq Batch API can reduce indexing costs by ~50%")
        print("⚠️  WARNING: Batch indexing takes 20+ minutes per index")
        print("   (vs ~1-2 minutes for standard API calls)")
        config["use_batch_for_indexing"] = ask(
            "Enable batch indexing to save money?",
            DEFAULTS["use_batch_for_indexing"],
            skip,
        )

        # Ask about batch indexing for MCP queries
        print("\nMCP Query Incremental Indexing:")
        print("MCP queries can trigger incremental indexing before each query.")
        print("⚠️  WARNING: Batch indexing takes 5+ minutes per update")
        print("   (vs <1 minute for standard API calls)")
        print("   This can significantly delay query responses.")
        config["use_batch_for_mcp_incremental_indexing"] = ask(
            "Enable batch API for MCP-triggered incremental indexing?",
            DEFAULTS["use_batch_for_mcp_incremental_indexing"],
            skip,
        )

    # Step 3: Configure embeddings (if Groq is primary or optional batch mode)
    if provider == "groq":
        print("\nStep 3: Configure Embeddings Provider")
        print("-" * 50)
        print("Groq doesn't support embeddings. Choose a fallback provider:")
        print("  • openrouter - Access to OpenAI embeddings via OpenRouter")
        print("  • ollama     - Local embeddings via Ollama")
        print()

        fallback: str = ""
        while fallback not in ["openrouter", "ollama"]:
            fallback = (
                input("Select embedding provider [openrouter]: ").strip().lower()
                or "openrouter"
            )

        config["embedding_provider"]: str = fallback

        if fallback == "openrouter":
            f_key: str = ""
            while not f_key:
                f_key = input("Enter OpenRouter API key: ").strip()
            config["openrouter"]: dict = get_openrouter_config(f_key, skip, use_defaults=True)
            print("✓ OpenRouter configured for embeddings")
        else:  # ollama
            if "ollama" not in config:
                config["ollama"]: dict = {}
            config["ollama"]["embedding_model"] = ask(
                "Embedding Model", DEFAULTS["ollama"]["embedding_model"], skip
            )
            print("✓ Ollama configured for embeddings")
    else:
        # Optional: Add Groq for batch operations
        print("\nStep 3 (Optional): Add Groq for Batch Processing")
        print("-" * 50)
        print("Groq Batch API can make indexing cheaper and faster.")
        print("This is optional - you can use your primary provider instead.")
        print()

        if ask("Configure Groq for batch operations?", False):
            key: str = ""
            while not key:
                key = input("Enter your Groq API key: ").strip()

            g_def: dict = DEFAULTS["groq"]
            config["groq"]: dict = {
                "api_key": key,
                "chat_model": ask("Chat Model", g_def["chat_model"], skip),
                "batch_completion_window": ask(
                    "Batch Completion Window (24h/48h/72h/7d)",
                    g_def["batch_completion_window"],
                    skip,
                ),
                "poll_interval_seconds": int(
                    ask("Poll Interval (seconds)", g_def["poll_interval_seconds"], skip)
                ),
                "timeout_seconds": int(
                    ask("Timeout (seconds)", g_def["timeout_seconds"], skip)
                ),
            }
            config["batch_provider"]: str = "groq"
            print("✓ Groq configured for batch operations")

            # Ask about batch indexing
            print("\nBatch Indexing Configuration:")
            print("Groq Batch API can reduce indexing costs by ~50%")
            print("⚠️  WARNING: Batch indexing takes 20+ minutes per index")
            print("   (vs ~1-2 minutes for standard API calls)")
            config["use_batch_for_indexing"] = ask(
                "Enable batch indexing to save money?",
                DEFAULTS["use_batch_for_indexing"],
                skip,
            )

            # Ask about batch indexing for MCP queries
            print("\nMCP Query Incremental Indexing:")
            print("MCP queries can trigger incremental indexing before each query.")
            print("⚠️  WARNING: Batch indexing takes 5+ minutes per update")
            print("   (vs <1 minute for standard API calls)")
            print("   This can significantly delay query responses.")
            config["use_batch_for_mcp_incremental_indexing"] = ask(
                "Enable batch API for MCP-triggered incremental indexing?",
                DEFAULTS["use_batch_for_mcp_incremental_indexing"],
                skip,
            )

    # Save configuration
    print("\n" + "=" * 60)
    Path("config/config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    print("✓ Configuration saved to config/config.json")
    print("\nConfiguration Summary:")
    print(f"  Primary Provider: {provider}")
    if "embedding_provider" in config:
        print(f"  Embedding Provider: {config['embedding_provider']}")
    if "batch_provider" in config:
        print(f"  Batch Provider: {config['batch_provider']}")

    # Show batch indexing status
    if config.get("use_batch_for_indexing"):
        print("  Batch Indexing: ENABLED (slower but cheaper)")
        print("    ⚠️  Indexing will take 20+ minutes")
    else:
        print("  Batch Indexing: Disabled (faster indexing)")

    print("\nYou're all set! Run indexing with:")
    print("  python index_repo.py <path-to-repo>")


if __name__ == "__main__":
    main()
